"""E12: where the aggregation-order gap comes from.
Under the JMP aggregate rule each area total is W1 * R_k* with k* the criterion with the lowest population-weighted ratio, so
gap_area = sum_c pop_c * imp_c * (r_k*,c - min_k r_k,c) / sum_c pop_c, a sum of non-negative country terms. Each term is the
population-equivalent accounting difference; it does not identify individual households.
Imputation follows the JMP 2025 hierarchy (M49 sub-region -> M49 region -> global), as in figures/scripts/figdata.py.
Outputs: results/E12_gap_by_country_<year>.csv and results/E12_gap_by_group.csv"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
from m49 import SUB2L1
d = F.load()
EL = ['rprem', 'rav', 'rqual']


def area_frame(y):
    g = d[d.year == y]; rows = []
    for a in 'ur':
        sh = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'iso3': g.iso3.values, 'name': g['name'].values, 'm49': g.m49.values, 'l1': g.m49.map(SUB2L1).values, 'area': a,
                          'sdg': g.region_sdg.values, 'income': g.region_income.values, 'ldc': g.region_ldc.notna().values,
                          'fragile': g.region_fragileoecdany.notna().values, 'has_sm': g.wat_sm_t.notna().values,
                          'pop': (g.pop_t * sh).values, 'imp': g[f'imp_{a}'].values})
        for k in EL: f[k] = g[f'{k}_{a}'].values
        rows.append(f)
    f = pd.concat(rows, ignore_index=True)
    return f[f['pop'].notna()].reset_index(drop=True)


def impute(f, col, w):
    for lvl in ['m49', 'l1', None]:
        miss = f[col].isna()
        if not miss.any(): break
        src = f[f[col].notna()]
        if lvl is None:
            for a in 'ur':
                s = src[src.area == a]; f.loc[miss & (f.area == a), col] = np.average(s[col], weights=s[w] + 1e-9)
            break
        mu = src.groupby([lvl, 'area']).apply(lambda g: np.average(g[col], weights=g[w] + 1e-9))
        f.loc[miss, col] = [mu.get((x, a), np.nan) for x, a in zip(f.loc[miss, lvl], f.loc[miss, 'area'])]
    return f


grp_rows = []
for y in [2015, 2024]:
    f = area_frame(y); f = impute(f, 'imp', 'pop'); f['ipop'] = f['pop'] * f.imp / 100
    for k in EL: f = impute(f, k, 'ipop')
    f['kstar'] = ''
    for a in 'ur':
        g = f.area == a
        R = {k: np.average(f.loc[g, k], weights=f.loc[g, 'ipop']) for k in EL}
        f.loc[g, 'kstar'] = min(R, key=R.get)
    f['r_kstar'] = [r[k] for r, k in zip(f[EL].to_dict('records'), f.kstar)]
    f['binding'] = f[EL].idxmin(axis=1).str[1:]
    f['hidden_M'] = f['pop'] * f.imp / 100 * (f.r_kstar - f[EL].min(axis=1)) / 1e3   # million people
    P = f['pop'].sum()
    print(y, 'aggregate criterion per area:', f.groupby('area').kstar.first().to_dict(), '| total hidden (M):', round(f.hidden_M.sum(), 1),
          '| gap (pp):', round(f.hidden_M.sum() * 1e3 / P * 100, 2))
    C = f.groupby(['iso3', 'name', 'sdg', 'income', 'ldc', 'fragile', 'has_sm']).agg(pop_M=('pop', lambda s: s.sum() / 1e3), hidden_M=('hidden_M', 'sum')).reset_index()
    C['hidden_share_of_pop'] = C.hidden_M / C.pop_M * 100
    hb = f.pivot_table(index='iso3', columns='binding', values='hidden_M', aggfunc='sum').fillna(0)
    C = C.merge(hb.rename(columns=lambda c: 'hidden_' + c + '_M'), left_on='iso3', right_index=True, how='left')
    C = C.sort_values('hidden_M', ascending=False); C['cum_share'] = C.hidden_M.cumsum() / C.hidden_M.sum()
    C.to_csv(f'results/E12_gap_by_country_{y}.csv', index=False)
    print(C.head(12)[['name', 'pop_M', 'hidden_M', 'hidden_share_of_pop', 'cum_share']].round(2).to_string(index=False))
    print('binding criterion of hidden people (M):', f.groupby('binding').hidden_M.sum().round(1).to_dict())
    # group gaps: aggregate-then-min vs min-then-aggregate computed WITHIN each group (as a regional figure would be)
    groups = {'All': np.ones(len(f), bool), 'Low income': f.income == 'Low income', 'Lower middle income': f.income == 'Lower middle income',
              'Upper middle income': f.income == 'Upper middle income', 'High income': f.income == 'High income',
              'Least developed': f.ldc, 'Fragile': f.fragile}
    for gname, m in groups.items():
        g = f[m]
        if g.area.nunique() < 2: continue
        Pg = g['pop'].sum(); agg = 0; mm = 0
        for a in 'ur':
            h = g[g.area == a]; w = h.ipop
            agg += h['pop'].sum() / Pg * np.average(h.imp, weights=h['pop']) * min(np.average(h[k], weights=w) for k in EL)
            mm += h['pop'].sum() / Pg * np.average(h.imp * h[EL].min(axis=1), weights=h['pop'])
        grp_rows.append(dict(year=y, group=gname, pop_M=Pg / 1e3, aggregate_then_min=agg, min_then_aggregate=mm, gap_pp=agg - mm,
                             hidden_M=(agg - mm) / 100 * Pg / 1e3, relative_gap=(agg - mm) / (100 - mm)))
G = pd.DataFrame(grp_rows); G.to_csv('results/E12_gap_by_group.csv', index=False)
print(G.round(2).to_string(index=False))
