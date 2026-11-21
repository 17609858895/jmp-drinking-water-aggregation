"""E20: country-preserving ceilings for the published regional and group values (JMP 2025 Annex 8).
For every SDG region, development group, income group and the world, hold the published national safely managed
estimates fixed and give each country without an estimate (a) its national-component ceiling and (b) the permissive
ceiling that lets any area without a documented value reach 100%. Compare both with the published group total and with
the lowest value that rounds to it, and flag the groups whose published total the aggregate-first rule reproduces within
one point (E18). Output: results/E20_group_ceilings.csv."""
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


def maximum(data, area):
    non_quality = data[[f'wat_imp_prem_{area}', f'wat_imp_av_{area}']].min(axis=1)
    improved = data[f'wat_basal_{area}'] + data[f'wat_lim_{area}']
    return non_quality.fillna(improved)


def country_values(year):
    data = pd.read_csv(RESULTS / 'jmp_wat.csv')
    r = data.loc[(data.year == year) & data.pop_t.notna()].copy()
    published = r.wat_sm_t.notna()
    national = maximum(r, 't').fillna(100)
    urban = r.prop_u / 100
    area = [r[f'wat_sm_{a}'].fillna(maximum(r, a)) for a in 'ur']
    relaxed = urban * area[0].fillna(100) + (1 - urban) * area[1].fillna(100)
    r['published_flag'] = published
    r['floor'] = r.wat_sm_t.fillna(0)
    r['ceiling_national'] = r.wat_sm_t.where(published, national)
    r['ceiling_permissive'] = r.wat_sm_t.where(published, np.fmax(national, relaxed))
    return r


pub = pd.read_csv(RESULTS / 'jmp2025_annex8_sm.csv')
rows = []
for year in (2015, 2024):
    r = country_values(year)
    for group, select in GROUPS.items():
        g = r[select(r)]
        p = pub[(pub.group == group) & (pub.year == year)].iloc[0]['total']
        w = g.pop_t
        est = g.published_flag
        rows.append({
            'group': group, 'year': year, 'countries': len(g), 'countries_with_estimates': int(est.sum()),
            'population_million': w.sum() / 1000,
            'population_share_with_estimates': 100 * w[est].sum() / w.sum(),
            'published_group_value': p,
            'mean_of_national_estimates': np.average(g.wat_sm_t[est], weights=w[est]) if est.any() else np.nan,
            'floor': np.average(g.floor, weights=w),
            'ceiling_national': np.average(g.ceiling_national, weights=w),
            'ceiling_permissive': np.average(g.ceiling_permissive, weights=w),
        })
out = pd.DataFrame(rows)
rec = pd.read_csv(RESULTS / 'E18_regional_reconstruction.csv')
rec = rec[rec.area == 'total'][['group', 'year', 'aggregate_first_imputed']].rename(columns={'aggregate_first_imputed': 'aggregate_first_reconstruction'})
out = out.merge(rec, on=['group', 'year'], how='left')
out['reproduced_within_one_point'] = (out.aggregate_first_reconstruction - out.published_group_value).abs() <= 1
out['lowest_value_rounding_to_published'] = out.published_group_value - 0.5
out['excess_over_permissive_ceiling'] = out.lowest_value_rounding_to_published - out.ceiling_permissive
out['exceeds_permissive_ceiling'] = out.excess_over_permissive_ceiling > 0
out.to_csv(RESULTS / 'E20_group_ceilings.csv', index=False)
pd.set_option('display.width', 260)
print(out.round(2).to_string(index=False))
v = out.dropna(subset=['published_group_value'])
for y in (2015, 2024):
    s = v[v.year == y]
    k = s[s.reproduced_within_one_point]
    print(y, 'groups with published totals', len(s), 'exceeding permissive ceiling', int(s.exceeds_permissive_ceiling.sum()),
          '| reproduced within one point', len(k), 'of which exceeding', int(k.exceeds_permissive_ceiling.sum()),
          'range %.2f-%.2f' % (k.excess_over_permissive_ceiling[k.exceeds_permissive_ceiling].min(), k.excess_over_permissive_ceiling[k.exceeds_permissive_ceiling].max()),
          '| groups with >99.9% of population estimated', int((s.population_share_with_estimates > 99.9).sum()))
