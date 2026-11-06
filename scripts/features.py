"""Country-area feature table built only from JMP 2025 world file (water, sanitation, hygiene sheets)."""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from m49 import ISO2M49
R = os.path.join(os.path.dirname(__file__), '..', 'results')
EL = ['prem', 'av', 'qual']
INCOME = {'Low income': 0, 'Lower middle income': 1, 'Upper middle income': 2, 'High income': 3}

def load(vintage=None):
    """vintage=None: JMP 2025 file; 2021 or 2023: earlier editions harmonised by 15_vintage_load.py."""
    src = (lambda s: os.path.join(R, f'jmp_{s}.csv')) if vintage is None else (lambda s: os.path.join(R, 'vintage', f'jmp{vintage}_{s}.csv'))
    w = pd.read_csv(src('wat'))
    s = pd.read_csv(src('san'))
    h = pd.read_csv(src('hyg'))
    keep_s = [c for c in s.columns if c.startswith('san_') and c[-2:] in ('_t', '_u', '_r') and not c.startswith('san_arc')]
    keep_h = [c for c in h.columns if c.startswith('hyg_') and c[-2:] in ('_t', '_u', '_r')]
    d = w.merge(s[['iso3', 'year'] + keep_s], on=['iso3', 'year'], how='left').merge(h[['iso3', 'year'] + keep_h], on=['iso3', 'year'], how='left')
    d['m49'] = d.iso3.map(ISO2M49)
    for a in 'tur':
        d[f'imp_{a}'] = d[f'wat_basal_{a}'] + d[f'wat_lim_{a}']
        for k in EL:
            d[f'r{k}_{a}'] = (d[f'wat_imp_{k}_{a}'] / d[f'imp_{a}']).clip(0, 1)
    return d

def area_long(d, years=None):
    """Stack urban and rural rows. Own-area and other-area JMP indicators as features; qual of own area is the target."""
    if years is not None: d = d[d.year.isin(years)]
    base_vars = ['wat_basal', 'wat_lim', 'wat_unimp', 'wat_ns', 'wat_pip', 'wat_imp_npip', 'imp', 'rprem', 'rav',
                 'san_basal', 'san_lim', 'san_unimp', 'san_ns', 'san_sew', 'san_sep', 'hyg_bas', 'hyg_lim', 'hyg_ns']
    base_vars = [v for v in base_vars if f'{v}_u' in d.columns]
    rows = []
    for a, o in [('u', 'r'), ('r', 'u')]:
        f = pd.DataFrame({'iso3': d.iso3.values, 'year': d.year.values, 'area': a, 'm49': d.m49.values,
                          'sdg': d.region_sdg.values, 'pop': (d.pop_t * (d.prop_u / 100 if a == 'u' else 1 - d.prop_u / 100)).values})
        f['is_urban'] = int(a == 'u')
        f['prop_u'] = d.prop_u.values
        f['log_pop_area'] = np.log10(f['pop'].clip(lower=0.1))
        for v in base_vars:
            f[f'{v}_own'] = d[f'{v}_{a}'].values
            f[f'{v}_oth'] = d[f'{v}_{o}'].values
            f[f'{v}_nat'] = d[f'{v}_t'].values
        # cross-area quality (available for e.g. China urban) and national anchor
        f['rqual_oth'] = d[f'rqual_{o}'].values
        f['rqual_nat'] = d['rqual_t'].values
        f['income'] = d.region_income.map(INCOME).values
        for c in ['region_ldc', 'region_lldc', 'region_sids', 'region_fragileoecdany']:
            f[c] = d[c].notna().astype(int).values if d[c].dtype == object else d[c].fillna(0).values
        f['y_rqual'] = d[f'rqual_{a}'].values
        f['y_sm'] = d[f'wat_sm_{a}'].values
        f['imp_area'] = d[f'imp_{a}'].values
        f['W8'] = d[f'wat_imp_prem_{a}'].values; f['W9'] = d[f'wat_imp_av_{a}'].values; f['W10'] = d[f'wat_imp_qual_{a}'].values
        f['W10_nat'] = d['wat_imp_qual_t'].values
        f['pu'] = d.prop_u.values / 100
        rows.append(f)
    L = pd.concat(rows, ignore_index=True)
    for r in sorted(L.sdg.dropna().unique()):
        L['sdg_' + r[:12].replace(' ', '_')] = (L.sdg == r).astype(int)
    return L
