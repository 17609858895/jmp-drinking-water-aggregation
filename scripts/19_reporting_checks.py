"""Check how the JMP ceiling changes if an unknown area is allowed full coverage."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'


def maximum(data, area):
    non_quality = data[[f'wat_imp_prem_{area}', f'wat_imp_av_{area}']].min(axis=1)
    improved = data[f'wat_basal_{area}'] + data[f'wat_lim_{area}']
    return non_quality.fillna(improved)


def audit_ceiling(file, year, headline):
    data = pd.read_csv(file)
    records = data.loc[(data.year == year) & data.pop_t.notna()].copy()
    published = records.wat_sm_t.notna()
    national = maximum(records, 't').fillna(100)
    urban = records.prop_u / 100
    assert urban[~published].between(0,1).all()
    area = [records[f'wat_sm_{a}'].fillna(maximum(records, a)) for a in 'ur']
    relaxed = urban * area[0].fillna(100) + (1 - urban) * area[1].fillna(100)
    complete_area = urban * area[0] + (1 - urban) * area[1]
    national_only = records.wat_sm_t.where(published, national)
    complete_variant = records.wat_sm_t.where(published, np.fmax(national, complete_area))
    relaxed_variant = records.wat_sm_t.where(published, np.fmax(national, relaxed))
    population = records.pop_t.sum()
    missing = data.loc[~data.iso3.isin(records.iso3) & data.pop_t.notna()].sort_values('year').drop_duplicates('iso3', keep='last')
    expanded = (relaxed_variant @ records.pop_t + 100 * missing.pop_t.sum()) / (population + missing.pop_t.sum())
    return {
        'reference_year': year,
        'headline': headline,
        'countries': len(records),
        'published_countries': int(published.sum()),
        'national_ceiling': float(national_only @ records.pop_t / population),
        'complete_pair_variant': float(complete_variant @ records.pop_t / population),
        'permissive_area_ceiling': float(relaxed_variant @ records.pop_t / population),
        'permissive_area_expanded_ceiling': float(expanded),
        'additional_countries': len(missing),
        'additional_population_millions': float(missing.pop_t.sum()/1000),
    }


def matched_household_comparison():
    data = pd.read_csv(RESULTS / 'jmp_wat.csv')
    y = data.loc[(data.year == 2024) & data.pop_t.notna()]
    domains = []
    for a in 'ur':
        improved = y[f'wat_basal_{a}'] + y[f'wat_lim_{a}']
        population = y.pop_t * (y.prop_u / 100 if a == 'u' else 1 - y.prop_u / 100)
        ratios = y[[f'wat_imp_{criterion}_{a}' for criterion in ('prem', 'av', 'qual')]].div(improved, axis=0).clip(0, 1)
        match = ratios.notna().all(axis=1) & improved.gt(0) & population.gt(0)
        domains.append(pd.DataFrame({
            'pop': population[match],
            'iso3': y.iso3[match],
            'upper': improved[match] * ratios[match].min(axis=1),
            'lower': improved[match] * (ratios[match].sum(axis=1) - 2).clip(lower=0),
            'independence': improved[match] * ratios[match].prod(axis=1),
        }))
    sample = pd.concat(domains, ignore_index=True)
    assert (sample.lower <= sample.independence + 1e-10).all()
    assert (sample.independence <= sample.upper + 1e-10).all()
    return {'domains':len(sample), 'countries':sample.iso3.nunique(),
            'population_bn':sample['pop'].sum()/1e6,
            **{k:np.average(sample[k],weights=sample['pop']) for k in ['upper','lower','independence']}}


out = {'ceiling': [audit_ceiling(RESULTS / 'jmp_wat.csv', 2024, 74),
                   audit_ceiling(RESULTS / 'vintage/jmp2021_wat.csv', 2020, 74),
                   audit_ceiling(RESULTS / 'vintage/jmp2023_wat.csv', 2022, 73)],
       'matched_household':matched_household_comparison()}
print(json.dumps(out,indent=2))
