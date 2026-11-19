"""E18: reconstruct the published 2015 and 2024 regional and group values (JMP 2025 Annex 8) with the aggregate-first rule.
Two versions: components completed by the M49 hierarchy of the 2025 annex ('imputed'), or each criterion averaged over the
countries reporting it ('available'). Also the local-minimum (country-consistent) value with the same completed components.
Groups: SDG regions, least developed, landlocked developing, small island developing, fragile contexts, World Bank income
groups, low- and middle-income, world. Output: results/E18_regional_reconstruction.csv and a summary of absolute differences."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
from m49 import SUB2L1
EL = ['rprem', 'rav', 'rqual']
d = F.load()


def frame(y):
    g = d[d.year == y]; fr = []
    for a in 'ur':
        sh = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'iso3': g.iso3.values, 'm49': g.m49.values, 'l1': g.m49.map(SUB2L1).values, 'area': a, 'pop': (g.pop_t * sh).values,
                          'imp': g[f'imp_{a}'].values, 'sdg': g.region_sdg.values, 'income': g.region_income.values, 'ldc': g.region_ldc.notna().values,
                          'lldc': g.region_lldc.notna().values, 'sids': g.region_sids.notna().values, 'fragile': g.region_fragileoecdany.notna().values})
        for k in EL: f[k] = g[f'{k}_{a}'].values
        fr.append(f)
    f = pd.concat(fr, ignore_index=True); return f[f['pop'].notna()].reset_index(drop=True)


def impute(f, col, w):
    for lvl in ['m49', 'l1', None]:
        miss = f[col].isna()
        if not miss.any(): break
        src = f[f[col].notna()]
        if lvl is None:
            for a in 'ur':
                s = src[src.area == a]; f.loc[miss & (f.area == a), col] = np.average(s[col], weights=s[w] + 1e-9)
            break
        mu = {k: np.average(q[col], weights=q[w] + 1e-9) for k, q in src.groupby([lvl, 'area'])}
        f.loc[miss, col] = [mu.get((x, a), np.nan) for x, a in zip(f.loc[miss, lvl], f.loc[miss, 'area'])]


def completed(y, imputed):
    f = frame(y); impute(f, 'imp', 'pop'); f['ipop'] = f['pop'] * f.imp / 100
    if imputed:
        for k in EL: impute(f, k, 'ipop')
    return f


def rule(f):
    """Aggregate-first values (rural, urban, total) and the local-minimum total for one group frame."""
    out, P = {}, f['pop'].sum(); tot = 0.0; loc = 0.0
    for a, nm in [('r', 'rural'), ('u', 'urban')]:
        s = f[f.area == a]
        m = []
        for k in EL:
            ok = s[k].notna() & (s.ipop > 0)
            if ok.any(): m.append(np.average(s[k][ok], weights=s.ipop[ok]))
        out[nm] = np.average(s.imp, weights=s['pop']) * min(m)
        tot += s['pop'].sum() / P * out[nm]
        r = s[EL].values; mn = np.nanmin(np.where(np.isnan(r), np.inf, r), axis=1); mn[np.isinf(mn)] = np.nan
        ok = ~np.isnan(mn)
        loc += s['pop'].sum() / P * np.average(s.imp[ok] * mn[ok], weights=s['pop'][ok])
    out['total'] = tot; out['local_min_total'] = loc
    return out


GROUPS = {'Australia and New Zealand': lambda f: f.sdg == 'Australia and New Zealand',
          'Central and Southern Asia': lambda f: f.sdg == 'Central and Southern Asia',
          'Eastern and South-Eastern Asia': lambda f: f.sdg == 'Eastern and South-Eastern Asia',
          'Europe and Northern America': lambda f: f.sdg == 'Europe and Northern America',
          'Latin America and the Caribbean': lambda f: f.sdg == 'Latin America and the Caribbean',
          'Northern Africa and Western Asia': lambda f: f.sdg == 'Northern Africa and Western Asia',
          'Oceania (excluding Australia and New Zealand)': lambda f: f.sdg == 'Oceania',
          'Sub-Saharan Africa': lambda f: f.sdg == 'Sub-Saharan Africa',
          'Landlocked developing countries': lambda f: f.lldc, 'Least developed countries': lambda f: f.ldc,
          'Small island developing States': lambda f: f.sids, 'Fragile contexts': lambda f: f.fragile,
          'Low- and middle-income': lambda f: f.income.isin(['Low income', 'Lower middle income', 'Upper middle income']),
          'Low-income': lambda f: f.income == 'Low income', 'Lower-middle-income': lambda f: f.income == 'Lower middle income',
          'Upper-middle-income': lambda f: f.income == 'Upper middle income', 'High-income': lambda f: f.income == 'High income',
          'World': lambda f: np.ones(len(f), bool)}
pub = pd.read_csv('results/jmp2025_annex8_sm.csv')
rows = []
for y in [2015, 2024]:
    fi, fa = completed(y, True), completed(y, False)
    for g, sel in GROUPS.items():
        ri, ra = rule(fi[sel(fi)]), rule(fa[sel(fa)])
        p = pub[(pub.group == g) & (pub.year == y)].iloc[0]
        for area in ['rural', 'urban', 'total']:
            rows.append(dict(group=g, year=y, area=area, published=p[area], aggregate_first_imputed=ri[area], aggregate_first_available=ra[area],
                             local_min_imputed=ri['local_min_total'] if area == 'total' else np.nan))
R = pd.DataFrame(rows)
R.to_csv('results/E18_regional_reconstruction.csv', index=False)
pd.set_option('display.width', 250)
v = R.dropna(subset=['published'])
v = v[v.published.between(1, 99)]
for c in ['aggregate_first_imputed', 'aggregate_first_available']:
    e = (v[c] - v.published).abs()
    print(c, 'n', len(e), 'MAE %.2f' % e.mean(), 'median %.2f' % e.median(), 'within 1 point %.0f%%' % ((e <= 1).mean() * 100), 'max %.2f' % e.max())
t = v[v.area == 'total']
print(t[['group', 'year', 'published', 'aggregate_first_imputed', 'aggregate_first_available', 'local_min_imputed']].round(1).to_string(index=False))
