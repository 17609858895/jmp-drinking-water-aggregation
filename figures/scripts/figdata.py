"""Figure input tables for the JMP safely managed drinking water study (results only; no model fitting).
Run from the project root: python figures/scripts/figdata.py  ->  results/figdata/*.csv"""
import sys, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'scripts'))
import features as F
from m49 import SUB2L1
FD = ROOT / 'results' / 'figdata'; FD.mkdir(parents=True, exist_ok=True)
d = F.load()
EL = ['rprem', 'rav', 'rqual']
OFFICIAL = {(2015, 'total'): 68, (2015, 'urban'): 83, (2015, 'rural'): 50, (2024, 'total'): 74, (2024, 'urban'): 83, (2024, 'rural'): 60}


# ------------------------------------------------------------------ replication series (JMP 2025 hierarchy), area-level elements
def area_frame(y):
    g = d[d.year == y]; rows = []
    for a in 'ur':
        sh = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'iso3': g.iso3.values, 'm49': g.m49.values, 'l1': g.m49.map(SUB2L1).values, 'area': a,
                          'pop': (g.pop_t * sh).values, 'imp': g[f'imp_{a}'].values})
        for k in EL: f[k] = g[f'{k}_{a}'].values
        rows.append(f)
    f = pd.concat(rows, ignore_index=True)
    return f[f['pop'].notna()].reset_index(drop=True)


def impute(f, col, w):
    f[col + '_obs'] = f[col].notna()
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


def rules(f):
    """(aggregate-then-minimum, minimum-then-aggregate) for urban, rural, total."""
    out = {}
    P = f['pop'].sum(); tot_a, tot_m = 0, 0
    for a, nm in [('u', 'urban'), ('r', 'rural')]:
        g = f[f.area == a]; w = g['pop'] * g.imp / 100
        W1 = np.average(g.imp, weights=g['pop'])
        agg = W1 * min(np.average(g[k], weights=w) for k in EL)
        mm = np.average(g.imp * g[EL].min(axis=1), weights=g['pop'])
        out[nm] = (agg, mm); tot_a += g['pop'].sum() / P * agg; tot_m += g['pop'].sum() / P * mm
    out['total'] = (tot_a, tot_m)
    return out


rows, comp = [], []
for y in range(2000, 2025):
    f = area_frame(y)
    # complete-data universe: countries with observed quality in both areas (no imputation at all)
    both = f.groupby('iso3').rqual.apply(lambda s: s.notna().all())
    fc = f[f.iso3.isin(both[both].index) & f[EL[0]].notna()].copy()
    fc['rav'] = fc.rav.fillna(1.0)   # availability missing -> minimum of the two available elements (JMP rule)
    rc = rules(fc)
    comp.append(dict(year=y, n_countries=fc.iso3.nunique(), pop_bn=fc['pop'].sum() / 1e6, jmp_rule=rc['total'][0], country_rule=rc['total'][1]))
    f = impute(f, 'imp', 'pop'); f['ipop'] = f['pop'] * f.imp / 100
    for k in EL: f = impute(f, k, 'ipop')
    for lvl, (a_, m_) in rules(f).items():
        rows.append(dict(year=y, level=lvl, jmp_rule=a_, country_rule=m_, official=OFFICIAL.get((y, lvl), np.nan)))
R = pd.DataFrame(rows); R.to_csv(FD / 'fig2_replication_series.csv', index=False)
pd.DataFrame(comp).to_csv(FD / 'fig2_complete_universe.csv', index=False)
print(R[R.year.isin([2015, 2024])].round(2).to_string(index=False))
print(pd.DataFrame(comp).iloc[[0, 15, 24]].round(2).to_string(index=False))

# ------------------------------------------------------------------ regional gap (measured countries only, national level, no imputation)
reg = []
for y in [2024]:
    g = d[(d.year == y) & d.wat_sm_t.notna()]
    for rg, gg in g.groupby('region_sdg'):
        Pi = gg.pop_t * gg.imp_t / 100
        r = {k: np.nansum(Pi * gg[f'{k}_t']) / Pi[gg[f'{k}_t'].notna()].sum() for k in EL}
        mom = np.average(gg.imp_t, weights=gg.pop_t) * min(r.values())
        mo = np.average(gg.wat_sm_t, weights=gg.pop_t)
        reg.append(dict(region=rg, n=len(gg), pop_M=gg.pop_t.sum() / 1e3, mean_of_mins=mo, min_of_means=mom, gap=mom - mo))
pd.DataFrame(reg).to_csv(FD / 'fig2_regional_gap_2024.csv', index=False)

# ------------------------------------------------------------------ country table 2024 (maps, binding element, bounds, ML)
y = d[(d.year == 2024) & d.pop_t.notna()].copy()
E11 = pd.read_csv('results/E11_country_estimates_2024.csv').set_index('iso3')
y = y.join(E11[['kind', 'sm_mean', 'sm_lo', 'sm_hi']], on='iso3')


def binding(r):
    """Population-majority binding element; area level where available, national otherwise."""
    pops = {}
    areas = [(a, r.pop_t * (r.prop_u / 100 if a == 'u' else 1 - r.prop_u / 100)) for a in 'ur' if pd.notna(r[f'rqual_{a}'])]
    if len(areas) == 2:
        for a, p in areas:
            v = {k: r[f'r{k}_{a}'] for k in ['prem', 'av', 'qual'] if pd.notna(r[f'r{k}_{a}'])}
            k = min(v, key=v.get); pops[k] = pops.get(k, 0) + p
        return max(pops, key=pops.get)
    if pd.notna(r.rqual_t):
        v = {k: r[f'r{k}_t'] for k in ['prem', 'av', 'qual'] if pd.notna(r[f'r{k}_t'])}
        return min(v, key=v.get)
    return 'none'


y['binding'] = y.apply(binding, axis=1).map({'prem': 'premises', 'av': 'availability', 'qual': 'quality', 'none': 'no estimate'})
y.loc[y.wat_sm_t.isna(), 'binding'] = 'no estimate'
pu = y.prop_u / 100
ubn = y[['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(y.imp_t).fillna(100)
def aub(a): return pd.concat([y[f'wat_imp_prem_{a}'], y[f'wat_imp_av_{a}']], axis=1).min(axis=1).fillna(y[f'imp_{a}'])
uba = pu * np.where(y.wat_sm_u.notna(), y.wat_sm_u, aub('u')) + (1 - pu) * np.where(y.wat_sm_r.notna(), y.wat_sm_r, aub('r'))
y['upper_bound'] = np.where(y.wat_sm_t.notna(), y.wat_sm_t, np.fmax(ubn.values, np.nan_to_num(uba, nan=-1)))
cols = ['iso3', 'name', 'region_sdg', 'region_income', 'pop_t', 'prop_u', 'wat_sm_t', 'binding', 'kind', 'sm_mean', 'sm_lo', 'sm_hi', 'upper_bound',
        'wat_imp_prem_t', 'wat_imp_av_t', 'wat_imp_qual_t', 'imp_t', 'wat_sm_u', 'wat_sm_r']
y[cols].to_csv(FD / 'fig1_country_2024.csv', index=False)
print(y.groupby('binding').pop_t.sum().div(1e6).round(3))

# population by binding element at area level (measured country-areas) + unmeasured
ab = []
for a in 'ur':
    p = y.pop_t * (pu if a == 'u' else 1 - pu)
    has = y[f'rqual_{a}'].notna() & y.wat_sm_t.notna()
    v = y[[f'rprem_{a}', f'rav_{a}', f'rqual_{a}']].copy(); v.columns = ['premises', 'availability', 'quality']
    b = v[has].idxmin(axis=1)
    for k in ['premises', 'availability', 'quality']:
        ab.append(dict(area=a, element=k, pop_bn=p[has][b == k].sum() / 1e6))
    nat = y.wat_sm_t.notna() & ~has
    vn = y.loc[nat, ['rprem_t', 'rav_t', 'rqual_t']]; vn.columns = ['premises', 'availability', 'quality']
    bn = vn.idxmin(axis=1)
    for k in ['premises', 'availability', 'quality']:
        ab.append(dict(area=a, element=k, pop_bn=p[nat][bn == k].sum() / 1e6))
    ab.append(dict(area=a, element='no estimate', pop_bn=p[y.wat_sm_t.isna()].sum() / 1e6))
AB = pd.DataFrame(ab).groupby(['area', 'element']).pop_bn.sum().unstack('element')
AB.to_csv(FD / 'fig1_binding_population.csv'); print(AB.round(3))

# ------------------------------------------------------------------ Fig. 3: series with hard bound and required coverage
S11 = pd.read_csv('results/E11_global_series.csv')
Rt = R[R.level == 'total'].set_index('year')
req = []
for yr in range(2000, 2025):
    g = d[(d.year == yr) & d.pop_t.notna()]
    m = g.wat_sm_t.notna(); P = g.pop_t.sum(); sh = g.pop_t[m].sum() / P
    avg_m = np.average(g.wat_sm_t[m], weights=g.pop_t[m])
    gu = g.prop_u / 100
    ub_n = g[['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(g.imp_t).fillna(100)
    def aub_(a): return pd.concat([g[f'wat_imp_prem_{a}'], g[f'wat_imp_av_{a}']], axis=1).min(axis=1).fillna(g[f'imp_{a}'])
    ub_a = gu * np.where(g.wat_sm_u.notna(), g.wat_sm_u, aub_('u')) + (1 - gu) * np.where(g.wat_sm_r.notna(), g.wat_sm_r, aub_('r'))
    ub = np.fmax(ub_n.values, np.nan_to_num(ub_a, nan=-1))
    ub_unm = np.average(ub[~m.values], weights=g.pop_t[~m].values)
    upper = sh * avg_m + (1 - sh) * ub_unm
    jr = Rt.loc[yr, 'jmp_rule']
    req.append(dict(year=yr, measured_share=sh, mean_measured=avg_m, jmp_rule=jr, required_unmeasured=(jr - sh * avg_m) / (1 - sh),
                    upper_unmeasured=ub_unm, upper_global=upper, country_rule_regional=Rt.loc[yr, 'country_rule']))
Q = pd.DataFrame(req).merge(S11[['year', 'consistent_ml', 'consistent_ml_lo90', 'consistent_ml_hi90']], on='year', how='left')
Q.to_csv(FD / 'fig3_series.csv', index=False); print(Q.iloc[[0, 15, 24]].round(2).to_string(index=False))
ex = y[y.wat_sm_t.isna()].sort_values('pop_t', ascending=False)
ex = ex[ex.imp_t.notna()].head(10)
ex[['iso3', 'name', 'pop_t', 'wat_imp_prem_t', 'wat_imp_av_t', 'upper_bound', 'sm_mean', 'sm_lo', 'sm_hi', 'wat_sm_u', 'prop_u']].to_csv(FD / 'fig3_unmeasured_examples.csv', index=False)

# ------------------------------------------------------------------ Fig. 6: China rural quality scenario
c = d[(d.year == 2024) & (d.iso3 == 'CHN')].iloc[0]
Pt = y.pop_t.sum()
others = (y.pop_t * y.sm_mean)[y.iso3 != 'CHN'].sum()
p_u, p_r = c.pop_t * c.prop_u / 100, c.pop_t * (1 - c.prop_u / 100)
b_r = np.nanmin([c.rprem_r, c.rav_r])
qs = np.linspace(0.4, 1.0, 61)
glob = (others + p_u * c.wat_sm_u + p_r * c.imp_r * np.minimum(b_r, qs)) / Pt
pd.DataFrame({'q_rural': qs, 'china_sm': (p_u * c.wat_sm_u + p_r * c.imp_r * np.minimum(b_r, qs)) / c.pop_t, 'global_sm': glob}).to_csv(FD / 'fig6_china_scenario.csv', index=False)
print('China rural: imp', round(c.imp_r, 2), 'bound ratio', round(b_r, 3), 'ML China', round(E11.loc['CHN', 'sm_mean'], 2))

# ------------------------------------------------------------------ Fig. 6d: regional aggregate-then-minimum (same JMP imputation) vs ML country rule
f = area_frame(2024)
f = f.merge(d[d.year == 2024][['iso3', 'region_sdg']], on='iso3', how='left')
f = impute(f, 'imp', 'pop'); f['ipop'] = f['pop'] * f.imp / 100
for k in EL: f = impute(f, k, 'ipop')
E11r = pd.read_csv('results/E11_regional_2024.csv').set_index('region')
rr = []
for rg, g in f.groupby('region_sdg'):
    a_, m_ = rules(g.reset_index(drop=True))['total'] if set(g.area) == {'u', 'r'} else (np.nan, np.nan)
    e = E11r.loc[rg]
    ub_r = y[y.region_sdg == rg]
    rr.append(dict(region=rg, pop_M=e.pop_M, measured_share=e.measured_share, jmp_rule=a_, country_rule_regional=m_,
                   ml=e.consistent_ml, ml_lo=e.lo90, ml_hi=e.hi90, upper_bound=np.average(ub_r.upper_bound, weights=ub_r.pop_t)))
RR = pd.DataFrame(rr); RR.to_csv(FD / 'fig6_regional.csv', index=False); print(RR.round(2).to_string(index=False))
