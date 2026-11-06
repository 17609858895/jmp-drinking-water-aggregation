"""E08: 2000-2024 global series under the JMP aggregation rule vs. the JMP country rule applied consistently
(mean of area-level minima), plus Frechet bounds and the independence value for the joint household criterion.
Missing area elements are imputed with M49 sub-region -> M49 region (default, JMP 2025 Annex 1) or SDG region -> global means;
ML imputation replaces this step in E10. Also a low/middle-income subset for comparison with survey-based estimates."""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); import features as F
from m49 import SUB2L1
HIER = sys.argv[1] if len(sys.argv) > 1 else 'm49l1'   # JMP 2025: M49 sub-region -> M49 region -> global
d = F.load()
EL = ['rprem', 'rav', 'rqual']


def frame(y):
    g = d[d.year == y]
    rows = []
    for a in ['u', 'r']:
        share = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'iso3': g.iso3.values, 'm49': g.m49.values, 'sdg': (g.m49.map(SUB2L1) if HIER == 'm49l1' else g.region_sdg).values, 'income': g.region_income.values,
                          'area': a, 'pop': (g.pop_t * share).values, 'imp': g[f'imp_{a}'].values})
        for k in EL: f[k] = g[f'{k}_{a}'].values
        rows.append(f)
    f = pd.concat(rows, ignore_index=True)
    return f[f['pop'].notna()].reset_index(drop=True)


def impute(f, col, w):
    f[col + '_obs'] = f[col].notna()
    for lvl in ['m49', 'sdg', None]:
        miss = f[col].isna()
        if not miss.any(): break
        src = f[f[col].notna()]
        if lvl is None:
            for a in ['u', 'r']:
                s = src[src.area == a]; f.loc[miss & (f.area == a), col] = np.average(s[col], weights=s[w] + 1e-9)
            break
        mu = src.groupby([lvl, 'area']).apply(lambda g: np.average(g[col], weights=g[w] + 1e-9))
        f.loc[miss, col] = [mu.get((x, a), np.nan) for x, a in zip(f.loc[miss, lvl], f.loc[miss, 'area'])]
    return f


def complete(y):
    f = frame(y)
    f = impute(f, 'imp', 'pop')
    f['ipop'] = f['pop'] * f.imp / 100
    for k in EL: f = impute(f, k, 'ipop')
    return f


def summarise(f):
    out = {}
    P = f['pop'].sum()
    # JMP rule: aggregate element ratios among improved users per area, take min, weight urban/rural
    jmp = 0.0
    for a in ['u', 'r']:
        g = f[f.area == a]
        W1 = np.average(g.imp, weights=g['pop'])
        jmp += g['pop'].sum() / P * W1 * min(np.average(g[k], weights=g.ipop) for k in EL)
    R = f[EL].values
    upper = f.imp * R.min(axis=1)
    indep = f.imp * R.prod(axis=1)
    lower = f.imp * np.maximum(0, R.sum(axis=1) - 2)
    w = f['pop']
    out.update(jmp_rule=jmp, mean_of_mins=np.average(upper, weights=w), independence=np.average(indep, weights=w),
               frechet_lower=np.average(lower, weights=w), pop_bn=P / 1e6,
               qual_obs_pop_share=f.loc[f.rqual_obs, 'pop'].sum() / P)
    for k in ['jmp_rule', 'mean_of_mins', 'independence', 'frechet_lower']:
        out['lack_bn_' + k] = (1 - out[k] / 100) * out['pop_bn']
    return out


rows = []
for y in range(2000, 2025):
    f = complete(y)
    rows.append(dict(year=y, group='World', **summarise(f)))
    lmic = f[f.income.isin(['Low income', 'Lower middle income', 'Upper middle income'])]
    rows.append(dict(year=y, group='LMIC (WB 2025 groups)', **summarise(lmic)))
S = pd.DataFrame(rows)
S['gap_pp'] = S.jmp_rule - S.mean_of_mins
S['gap_people_M'] = S.lack_bn_mean_of_mins * 1e3 - S.lack_bn_jmp_rule * 1e3
S.to_csv('results/E08_series_bounds.csv' if HIER == 'sdg' else 'results/E08_series_bounds_m49l1.csv', index=False)
pd.set_option('display.width', 250)
show = ['year', 'group', 'jmp_rule', 'mean_of_mins', 'independence', 'frechet_lower', 'gap_pp', 'lack_bn_jmp_rule', 'lack_bn_mean_of_mins', 'lack_bn_independence', 'lack_bn_frechet_lower', 'qual_obs_pop_share']
print(S[S.year.isin([2000, 2005, 2010, 2015, 2020, 2024])][show].round(3).to_string())
