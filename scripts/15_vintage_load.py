"""E15a: harmonise the JMP 2021 and 2023 world files to the 2025 column names used by features.py.
Group attributes missing from the old files (income, LDC, LLDC, SIDS, fragility) are taken from the 2025 file
(country classifications, not outcomes). Old files are read only (openpyxl via pandas).
Output: results/vintage/jmp{2021,2023}_{wat,san,hyg}.csv"""
import os, re
import numpy as np, pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join('data', 'raw')
OUT = os.path.join('results', 'vintage'); os.makedirs(OUT, exist_ok=True)
SRC = {2023: os.path.join(DATA, 'JMP_2023_WLD.xlsx'), 2021: os.path.join(DATA, 'JMP_2021_WLD.xlsx')}
REN21 = [(r'^pop_n$', 'pop_t'), (r'_n$', '_t'), (r'^wat_bas_', 'wat_basal_'), (r'^wat_sur_', 'wat_ns_'), (r'^wat_premises_', 'wat_imp_prem_'),
         (r'^wat_available_', 'wat_imp_av_'), (r'^wat_quality_', 'wat_imp_qual_'), (r'^wat_npip_', 'wat_imp_npip_'), (r'^arc_wat_bas_', 'arc_wat_basal_'),
         (r'^san_bas_', 'san_basal_'), (r'^san_od_', 'san_ns_'), (r'^san_lat_', 'san_othimp_'), (r'^arc_san_bas_', 'arc_san_basal_'),
         (r'^arc_san_od_', 'arc_san_ns_'), (r'^hyg_nfac_', 'hyg_ns_')]
GROUPS = ['region_income', 'region_ldc', 'region_lldc', 'region_sids', 'region_fragileoecdany']
ref = pd.read_csv(os.path.join('results', 'jmp_wat.csv'))
grp = ref.sort_values('year').groupby('iso3')[GROUPS].last()


def rename(cols, vintage):
    out = []
    for c in cols:
        if vintage == 2021:
            for pat, rep in REN21:
                c = re.sub(pat, rep, c)
        out.append(c)
    return out


for v, f in SRC.items():
    for s in ['wat', 'san', 'hyg']:
        d = pd.read_excel(f, sheet_name=s)
        d.columns = rename(list(d.columns), v)
        d = d.drop(columns=[c for c in GROUPS if c in d.columns]).merge(grp, left_on='iso3', right_index=True, how='left')
        d.to_csv(os.path.join(OUT, f'jmp{v}_{s}.csv'), index=False)
        print(v, s, d.shape, 'countries', d.iso3.nunique(), 'years', d.year.min(), d.year.max())
    w = pd.read_csv(os.path.join(OUT, f'jmp{v}_wat.csv'))
    # identity checks: area SM = min of the three elements; national = urban/rural weighted
    for a in 'ur':
        ok = w[f'wat_sm_{a}'].notna()
        m = w.loc[ok, [f'wat_imp_prem_{a}', f'wat_imp_av_{a}', f'wat_imp_qual_{a}']].min(axis=1)
        print('  ', v, a, 'SM = min(elements): max abs diff', round((w.loc[ok, f'wat_sm_{a}'] - m).abs().max(), 4), 'n', ok.sum())
    y = w[w.year == w.year.max()]
    print('  ', v, 'latest year', w.year.max(), 'countries with national SM', y.wat_sm_t.notna().sum(), 'pop share', round(y.pop_t[y.wat_sm_t.notna()].sum() / y.pop_t.sum(), 3))
