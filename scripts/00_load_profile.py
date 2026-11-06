"""Load JMP 2025 world file (unformatted sheets) into tidy CSV/parquet and profile structure."""
import pandas as pd, numpy as np, sys, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "JMP_2025_WLD.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
sheets = {}
for s in ['wat', 'san', 'hyg', 'mh']:
    d = pd.read_excel(SRC, sheet_name=s)
    d = d.loc[:, ~d.columns.str.startswith('name_')]
    d = d.drop(columns=[c for c in ['blank', 'name2', 'year2'] if c in d.columns])
    d.to_parquet(os.path.join(OUT, f"jmp_{s}.parquet"))
    d.to_csv(os.path.join(OUT, f"jmp_{s}.csv"), index=False)
    sheets[s] = d
    print('=====', s, d.shape, 'countries', d.iso3.nunique(), 'years', d.year.min(), d.year.max())
w = sheets['wat']
print(w[['sl', 'latestyear']].describe())
print(w.groupby('iso3').latestyear.nunique().value_counts())
print(w.latestyear.value_counts().head(10))
# missingness by year for key vars
key = ['wat_basal_t', 'wat_sm_t', 'wat_sm_r', 'wat_sm_u', 'wat_imp_prem_t', 'wat_imp_av_t', 'wat_imp_qual_t', 'wat_pip_t']
print(w.groupby('year')[key].apply(lambda g: g.notna().sum()).loc[[2000, 2005, 2010, 2015, 2020, 2024]])
print(w[['region_sdg', 'region_income', 'region_fragileoecdany', 'region_ldc']].drop_duplicates().shape)
print(w.region_income.value_counts() / 25)
