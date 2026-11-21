"""E21: the country-preserving compatibility test applied to safely managed sanitation (JMP 2025 Annex 9).
Safely managed sanitation is a sum of components (disposed in situ, emptied and treated, wastewater treated), so
averaging components across countries before combining them gives the same value as averaging national estimates.
For each region, group and the world: hold national safely managed sanitation estimates fixed, bound each country
without an estimate between zero and its at-least-basic coverage (100% if that is also missing), and compare the
bounds with the regional value in Annex 9. Output: results/E21_sanitation_control.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'

GROUPS = {
    'Australia and New Zealand': lambda f: f.region_sdg == 'Australia and New Zealand',
    'Central and Southern Asia': lambda f: f.region_sdg == 'Central and Southern Asia',
    'Eastern and South-Eastern Asia': lambda f: f.region_sdg == 'Eastern and South-Eastern Asia',
    'Europe and Northern America': lambda f: f.region_sdg == 'Europe and Northern America',
    'Latin America and the Caribbean': lambda f: f.region_sdg == 'Latin America and the Caribbean',
    'Northern Africa and Western Asia': lambda f: f.region_sdg == 'Northern Africa and Western Asia',
    'Oceania (excluding Australia and New Zealand)': lambda f: f.region_sdg == 'Oceania',
    'Sub-Saharan Africa': lambda f: f.region_sdg == 'Sub-Saharan Africa',
    'Landlocked developing countries': lambda f: f.region_lldc.notna(),
    'Least developed countries': lambda f: f.region_ldc.notna(),
    'Small island developing States': lambda f: f.region_sids.notna(),
    'Fragile contexts': lambda f: f.region_fragileoecdany.notna(),
    'Low- and middle-income': lambda f: f.region_income.isin(['Low income', 'Lower middle income', 'Upper middle income']),
    'Low-income': lambda f: f.region_income == 'Low income',
    'Lower-middle-income': lambda f: f.region_income == 'Lower middle income',
    'Upper-middle-income': lambda f: f.region_income == 'Upper middle income',
    'High-income': lambda f: f.region_income == 'High income',
    'World': lambda f: pd.Series(True, index=f.index),
}

def pooled_components(g):
    """Total obtained by averaging each component over the areas that report it, then adding (rural and urban weighted)."""
    total = 0.0
    for a in 'ur':
        w = g.pop_t * (g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100)
        area = 0.0
        for c in ('sdo_sm', 'fst_sm', 'sew_sm'):
            x = g[f'san_{c}_{a}']; ok = x.notna() & (w > 0)
            if ok.any(): area += np.average(x[ok], weights=w[ok])
        total += w.sum() / g.pop_t.sum() * area
    return total


def area_share(g, a):
    w = g.pop_t * (g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100)
    return 100 * w[g[f'san_sm_{a}'].notna()].sum() / w.sum()


data = pd.read_csv(RESULTS / 'jmp_san.csv')
pub = pd.read_csv(RESULTS / 'jmp2025_annex9_sm.csv')
rows = []
for year in (2015, 2024):
    r = data.loc[(data.year == year) & data.pop_t.notna()].copy()
    est = r.san_sm_t.notna()
    r['floor'] = r.san_sm_t.fillna(0)
    r['ceiling'] = r.san_sm_t.where(est, r.san_basal_t.fillna(100))
    for group, select in GROUPS.items():
        g = r[select(r)]
        e = g.san_sm_t.notna()
        w = g.pop_t
        p = pub[(pub.group == group) & (pub.year == year)].iloc[0]['total']
        rows.append({
            'group': group, 'year': year, 'countries': len(g), 'countries_with_estimates': int(e.sum()),
            'population_million': w.sum() / 1000,
            'population_share_with_estimates': 100 * w[e].sum() / w.sum(),
            'regional_value': p,
            'mean_of_national_estimates': np.average(g.san_sm_t[e], weights=w[e]) if e.any() else np.nan,
            'floor': np.average(g.floor, weights=w),
            'ceiling': np.average(g.ceiling, weights=w),
            'pooled_component_total': pooled_components(g),
            'rural_population_share_with_estimates': area_share(g, 'r'),
            'urban_population_share_with_estimates': area_share(g, 'u'),
        })
out = pd.DataFrame(rows)
out['excess_over_ceiling'] = out.regional_value - 0.5 - out.ceiling
out['shortfall_below_floor'] = out.floor - (out.regional_value + 0.5)
out['above_ceiling'] = out.excess_over_ceiling > 0
out['below_floor'] = out.shortfall_below_floor > 0
out['compatible'] = ~(out.above_ceiling | out.below_floor)
out['difference_from_national_mean'] = out.regional_value - out.mean_of_national_estimates
out.to_csv(RESULTS / 'E21_sanitation_control.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(2).to_string(index=False))
v = out.dropna(subset=['regional_value'])
for y in (2015, 2024):
    s = v[v.year == y]
    full = s[s.population_share_with_estimates > 95]
    print(y, 'totals', len(s), 'compatible', int(s.compatible.sum()), 'above ceiling', int(s.above_ceiling.sum()), 'below floor', int(s.below_floor.sum()),
          '| groups with >95% of population estimated:', [(a, round(b, 2)) for a, b in zip(full.group, full.difference_from_national_mean)])
