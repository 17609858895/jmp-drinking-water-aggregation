"""Aggregation gap (min-of-means vs mean-of-mins) and Frechet bounds for SM drinking water, using countries with JMP SM estimates."""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
w = pd.read_parquet('results/jmp_wat.parquet')
for a in 'tru':
    w[f'imp_{a}'] = w[f'wat_basal_{a}'] + w[f'wat_lim_{a}']
    for k in ['prem', 'av', 'qual']:
        w[f'{k}_{a}'] = w[f'wat_imp_{k}_{a}']
print('W8<=W1 viol', (w.prem_t > w.imp_t + 1e-6).sum(), (w.qual_t > w.imp_t + 1e-6).sum())

def frechet(df, a):
    imp = df[f'imp_{a}'] / 100
    comps = df[[f'prem_{a}', f'av_{a}', f'qual_{a}']] / 100
    ratios = comps.div(imp.replace(0, np.nan), axis=0).clip(upper=1)
    k = ratios.notna().sum(axis=1)
    up = comps.min(axis=1)
    lo = imp * np.maximum(0, ratios.sum(axis=1) - (k - 1))
    indep = imp * ratios.prod(axis=1, min_count=1)
    return up * 100, lo * 100, indep * 100

rows = []
for (y), g in w.groupby('year'):
    g = g[g.wat_sm_t.notna()].copy()
    P = g.pop_t
    has_ru = g.wat_sm_r.notna() & g.wat_sm_u.notna()
    # country-level upper/lower/independence bounds (area-level where possible)
    up_t, lo_t, in_t = frechet(g, 't')
    pu = g.prop_u / 100
    up_r, lo_r, in_r = frechet(g, 'r'); up_u, lo_u, in_u = frechet(g, 'u')
    up = np.where(has_ru, pu * up_u + (1 - pu) * up_r, up_t)
    lo = np.where(has_ru, pu * lo_u + (1 - pu) * lo_r, lo_t)
    ind = np.where(has_ru, pu * in_u + (1 - pu) * in_r, in_t)
    mean_of_mins = np.average(g.wat_sm_t, weights=P)
    # JMP-style regional: min of pop-weighted means of components among improved users (national-level for simplicity)
    imp = g.imp_t
    def mom(gg):
        Pi = gg.pop_t * gg.imp_t / 100
        r = {k: np.nansum(Pi * gg[f'{k}_t'] / gg.imp_t) / Pi[gg[f'{k}_t'].notna()].sum() for k in ['prem', 'av', 'qual']}
        return np.average(gg.imp_t, weights=gg.pop_t) * min(r.values())
    min_of_means = mom(g)
    rows.append(dict(year=y, n=len(g), pop_bn=P.sum() / 1e6, mean_of_mins=mean_of_mins, min_of_means=min_of_means,
                     gap_pp=min_of_means - mean_of_mins, upper=np.average(up, weights=P), indep=np.average(ind, weights=P),
                     lower=np.average(lo, weights=P), check_up_vs_sm=np.abs(up - g.wat_sm_t).max()))
R = pd.DataFrame(rows)
R['gap_people_M'] = R.gap_pp / 100 * R.pop_bn * 1e3
R['lack_upper_bn'] = (1 - R.lower / 100) * R.pop_bn
R['lack_jmpmin_bn'] = (1 - R.mean_of_mins / 100) * R.pop_bn
print(R.round(3).to_string())
R.to_csv('results/E02_aggregation_frechet_by_year.csv', index=False)
# by SDG region 2024
g = w[(w.year == 2024) & w.wat_sm_t.notna()]
out = []
for reg, gg in g.groupby('region_sdg'):
    Pi = gg.pop_t * gg.imp_t / 100
    r = {k: np.nansum(Pi * gg[f'{k}_t'] / gg.imp_t) / Pi[gg[f'{k}_t'].notna()].sum() for k in ['prem', 'av', 'qual']}
    mom = np.average(gg.imp_t, weights=gg.pop_t) * min(r.values())
    mo = np.average(gg.wat_sm_t, weights=gg.pop_t)
    out.append(dict(region=reg, n=len(gg), pop_M=gg.pop_t.sum() / 1e3, mean_of_mins=mo, min_of_means=mom, gap=mom - mo, binding_region=min(r, key=r.get)))
print(pd.DataFrame(out).round(2).to_string())
