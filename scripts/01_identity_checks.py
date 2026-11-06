import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 200)
w = pd.read_parquet('results/jmp_wat.parquet')
for a in ['t', 'r', 'u']:
    cols = [f'wat_imp_prem_{a}', f'wat_imp_av_{a}', f'wat_imp_qual_{a}']
    m = w[cols].min(axis=1, skipna=True)
    ok = w[f'wat_sm_{a}'].notna()
    diff = (w.loc[ok, f'wat_sm_{a}'] - m[ok])
    print(a, 'n', ok.sum(), 'abs diff quantiles', diff.abs().quantile([.5, .9, .99, 1]).round(4).tolist())
    # binding constraint
    arg = w.loc[ok, cols].idxmin(axis=1).str.replace(f'wat_imp_', '').str.replace(f'_{a}', '')
    print('  binding', arg[w.loc[ok, 'year'] == 2024].value_counts().to_dict())
# total vs urban/rural
pu = w.prop_u / 100
for v in ['wat_basal', 'wat_sm', 'wat_pip']:
    ok = w[[f'{v}_r', f'{v}_u', f'{v}_t']].notna().all(axis=1)
    rec = pu * w[f'{v}_u'] + (1 - pu) * w[f'{v}_r']
    print(v, 'n', ok.sum(), 'recon err q', (rec - w[f'{v}_t'])[ok].abs().quantile([.5, .9, .99, 1]).round(4).tolist())
# when total SM exists but r/u missing
print('sm_t without r&u', (w.wat_sm_t.notna() & (w.wat_sm_r.isna() | w.wat_sm_u.isna())).sum())
# ladder sums
s = w[['wat_basal_t', 'wat_lim_t', 'wat_unimp_t', 'wat_ns_t']].sum(axis=1, min_count=4)
print('ladder sum q', s.dropna().quantile([0, .5, 1]).tolist())
# SM <= basal? sm is subset of improved; basal is improved within 30 min.
d = w.dropna(subset=['wat_sm_t'])
print('sm>basal frac', (d.wat_sm_t > d.wat_basal_t + 1e-6).mean())
# 2024 missing SM countries, population
y = w[w.year == 2024].copy()
print('2024 rows', len(y), 'pop total (bn)', y.pop_t.sum() / 1e6)
miss = y[y.wat_sm_t.isna()]
print('missing SM 2024:', len(miss), 'pop (bn)', miss.pop_t.sum() / 1e6)
print(miss.sort_values('pop_t', ascending=False)[['name', 'pop_t', 'wat_basal_t', 'wat_imp_prem_t', 'wat_imp_av_t', 'wat_pip_t', 'region_sdg', 'region_income']].head(40).to_string())
print(miss.region_income.value_counts(dropna=False))
print(y[y.wat_sm_t.notna()].region_income.value_counts(dropna=False))
# missing basic too?
print('missing basal 2024', y.wat_basal_t.isna().sum(), y[y.wat_basal_t.isna()].pop_t.sum() / 1e3, 'M')
