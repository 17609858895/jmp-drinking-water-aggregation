"""Replicate JMP regional/global SM aggregation (impute missing countries by M49-L2 population-weighted means;
aggregate element ratios among improved users; take min at aggregate level) and contrast with mean-of-country-mins."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts'); from m49 import ISO2M49
pd.set_option('display.width', 250)
w = pd.read_parquet('results/jmp_wat.parquet')
w['m49'] = w.iso3.map(ISO2M49)
assert w.m49.notna().all(), w.loc[w.m49.isna(), 'iso3'].unique()
EL = ['prem', 'av', 'qual']

def area_frame(y, mode):
    """Return long frame of (country, area) with pop, improved %, element ratios. mode: 'national' or 'area'."""
    g = w[w.year == y].copy()
    rows = []
    for _, r in g.iterrows():
        areas = [('t', 1.0)] if mode == 'national' else [('u', r.prop_u / 100), ('r', 1 - r.prop_u / 100)]
        for a, share in areas:
            imp = r[f'wat_basal_{a}'] + r[f'wat_lim_{a}']
            d = dict(iso3=r.iso3, m49=r.m49, sdg=r.region_sdg, area=a, pop=r.pop_t * share, imp=imp, sm=r[f'wat_sm_{a}'])
            src = a
            # national-only element data -> apply national ratio to both areas (assumption, documented)
            if mode == 'area' and pd.isna(r[f'wat_imp_qual_{a}']) and pd.notna(r['wat_imp_qual_t']):
                src = 't'
            imp_src = r[f'wat_basal_{src}'] + r[f'wat_lim_{src}']
            for k in EL:
                d[k] = r[f'wat_imp_{k}_{src}'] / imp_src if imp_src and imp_src > 0 else np.nan
            if src == 't' and a != 't':
                d['sm'] = imp * np.nanmin([d[k] for k in EL]) if pd.notna(d['qual']) else np.nan
            rows.append(d)
    return pd.DataFrame(rows)

def impute_regional(f, col, wcol):
    """Population-weighted mean within M49 subregion (fallback SDG region, then global) for missing col."""
    f = f.copy(); f[col + '_imputed'] = f[col].isna()
    for lvl in ['m49', 'sdg', None]:
        miss = f[col].isna()
        if not miss.any(): break
        if lvl is None:
            ok = f[col].notna(); f.loc[miss, col] = np.average(f.loc[ok, col], weights=f.loc[ok, wcol]); break
        means = f[f[col].notna()].groupby([lvl, 'area']).apply(lambda g: np.average(g[col], weights=g[wcol]) if g[wcol].sum() > 0 else np.nan)
        f.loc[miss, col] = [means.get((a, b), np.nan) for a, b in zip(f.loc[miss, lvl], f.loc[miss, 'area'])]
    return f

def run(y, mode):
    f = area_frame(y, mode)
    f = f[f['pop'].notna()]
    f = impute_regional(f, 'imp', 'pop')
    f['popimp'] = f['pop'] * f.imp / 100
    for k in EL:
        f = impute_regional(f, k, 'popimp')
    out = {}
    for name, g in [('urban', f[f.area == 'u']), ('rural', f[f.area == 'r']), ('total', f)]:
        if len(g) == 0: continue
        W1 = np.average(g.imp, weights=g['pop'])
        ratios = {k: np.average(g[k], weights=g.popimp) for k in EL}
        mom = W1 * min(ratios.values())
        # mean of country/area mins (same imputations)
        sm_c = g.imp * g[EL].min(axis=1)
        moms = np.average(sm_c, weights=g['pop'])
        out[name] = dict(min_of_means=mom, mean_of_mins=moms, binding=min(ratios, key=ratios.get), **{f'r_{k}': v for k, v in ratios.items()})
    if mode == 'area' and 'urban' in out:
        # JMP: total = weighted average of urban & rural aggregate estimates
        pu = f[f.area == 'u']['pop'].sum() / f['pop'].sum()
        out['total_jmp'] = dict(min_of_means=pu * out['urban']['min_of_means'] + (1 - pu) * out['rural']['min_of_means'],
                                mean_of_mins=pu * out['urban']['mean_of_mins'] + (1 - pu) * out['rural']['mean_of_mins'])
    obs_share = f.loc[~f.qual_imputed, 'pop'].sum() / f['pop'].sum()
    return out, f, obs_share

rows = []
for y in [2000, 2010, 2015, 2020, 2024]:
    for mode in ['national', 'area']:
        out, f, sh = run(y, mode)
        for k, v in out.items():
            rows.append(dict(year=y, mode=mode, level=k, obs_pop_share=sh, **v))
R = pd.DataFrame(rows)
R['gap_pp'] = R.min_of_means - R.mean_of_mins
print(R.round(3).to_string())
R.to_csv('results/E03_jmp_replication.csv', index=False)
