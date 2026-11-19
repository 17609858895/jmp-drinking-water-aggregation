"""E17: sensitivity of the 2024 country-preserving total to the completion method and to the regional error correlation.
(1) Country-preserving totals (published national estimates fixed) with the 60 countries without estimates completed by
    regional-mean quality (M49 subregion -> M49 region -> world, weighted by improved-source population), ML plug-in
    imp*min(b, q_hat) or ML expected minimum (the E11 estimator), using the same models and routes as E11.
(2) Decomposition from the all-domain local-minimum value with regional completion (E08, 67.79%) to the completed total.
(3) Global 90% interval with the share rho of the logit-residual variance shared within SDG regions set to 0, 0.25 (E11),
    0.5, 0.75 and 1: Z = logit(q_hat) + sqrt(1 - rho) e_local + sqrt(rho) e_region.
Models, residual pools and routes follow 11_global_country_rule.py for 2024."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
from m49 import SUB2L1
import catboost as cb

YEAR = 2024; NMC = 2000
d = F.load(); d['l1'] = d.m49.map(SUB2L1)


def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))
def model(seed): return cb.CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=3, verbose=0, random_seed=seed, thread_count=8)


def fit_with_oof(train, feats, ycol, unit_cols):
    sw = (1.0 / train.groupby(unit_cols).year.transform('count')).values
    m = model(0).fit(train[feats], train[ycol], sample_weight=sw)
    cs = train.iso3.unique(); fm = {c: i % 5 for i, c in enumerate(np.random.RandomState(1).permutation(cs))}; ff = train.iso3.map(fm).values
    t = (train.year == YEAR).values; oof = np.full(len(train), np.nan)
    for f in range(5):
        mm = model(f).fit(train[feats][ff != f], train[ycol][ff != f], sample_weight=sw[ff != f])
        k = (ff == f) & t
        oof[k] = np.clip(mm.predict(train[feats][k]), 0, 1)
    return m, lg(train[ycol].values[t]) - lg(oof[t]), oof[t]


L = F.area_long(d, list(range(2000, YEAR + 1)))
L = L[L.imp_area.notna() & (L.imp_area > 0)].reset_index(drop=True)
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area', 'rqual_oth', 'rqual_nat']
BASE = [c for c in L.columns if c not in DROP]
obs = L[L.y_rqual.notna()].reset_index(drop=True)
mS1 = fit_with_oof(obs, BASE, 'y_rqual', ['iso3', 'area'])
mS3 = fit_with_oof(obs[obs.rqual_oth.notna()].reset_index(drop=True), BASE + ['rqual_oth'], 'y_rqual', ['iso3', 'area'])
NV = [c for c in d.columns if c.endswith('_t') and (c.startswith('wat_') or c.startswith('san_') or c.startswith('hyg_')) and 'qual' not in c and 'sm' not in c and 'arc' not in c] + ['rprem_t', 'rav_t', 'imp_t']
NV = [c for c in NV if c not in ('wat_imp_prem_t', 'wat_imp_av_t')]
D = d.copy(); D['income_n'] = D.region_income.map(F.INCOME); D['log_pop'] = np.log10(D.pop_t.clip(lower=0.1))
for r in sorted(D.region_sdg.dropna().unique()): D['sdg_' + r[:12]] = (D.region_sdg == r).astype(int)
NF = NV + ['prop_u', 'income_n', 'log_pop'] + [c for c in D.columns if c.startswith('sdg_')]
mN = fit_with_oof(D[D.rqual_t.notna() & (D.year <= YEAR)].reset_index(drop=True), NF, 'rqual_t', ['iso3'])

# regional-mean quality ratios (observed domains of the reference year)
Lt = L[(L.year == YEAR) & L.y_rqual.notna()].copy(); Lt['l1'] = Lt.m49.map(SUB2L1); Lt['w'] = Lt['pop'] * Lt.imp_area
Nt = d[(d.year == YEAR) & d.rqual_t.notna()].copy(); Nt['w'] = Nt.pop_t * Nt.imp_t


def regmean(src, ycol, m49, area=None):
    s = src if area is None else src[src.area == area]
    for lvl, key in [('m49', m49), ('l1', SUB2L1.get(m49)), (None, None)]:
        g = s if lvl is None else s[s[lvl] == key]
        if len(g): return np.average(g[ycol], weights=g.w + 1e-9)


y = d[(d.year == YEAR) & d.pop_t.notna()].reset_index(drop=True)
Lc = L[L.year == YEAR].set_index(['iso3', 'area'])
units = []        # one entry per uncertain unit: dict(i=country index, share, imp, b, qh, model residual pools, region)
fixed = np.zeros(len(y)); reg_val = np.zeros(len(y)); plug_val = np.zeros(len(y))
regional_draw = []
for i, r in y.iterrows():
    if pd.notna(r.wat_sm_t):
        fixed[i] = reg_val[i] = plug_val[i] = r.wat_sm_t; continue
    if all((r.iso3, a) in Lc.index for a in 'ur'):
        pu = r.prop_u / 100
        for a, sh in [('u', pu), ('r', 1 - pu)]:
            row = Lc.loc[(r.iso3, a)]; b = np.nanmin([row.rprem_own, row.rav_own, 1.0])
            if pd.notna(row.y_sm):
                fixed[i] += sh * row.y_sm; reg_val[i] += sh * row.y_sm; plug_val[i] += sh * row.y_sm; continue
            m, res, oofp = mS3 if pd.notna(row.rqual_oth) else mS1
            feats = BASE + ['rqual_oth'] if pd.notna(row.rqual_oth) else BASE
            qh = float(np.clip(m.predict(row[feats].to_frame().T.astype(float)), 0, 1)[0])
            qr = regmean(Lt, 'y_rqual', row.m49, a)
            reg_val[i] += sh * row.imp_area * min(b, qr); plug_val[i] += sh * row.imp_area * min(b, qh)
            units.append(dict(i=i, share=sh, imp=row.imp_area, b=b, qh=qh, res=res, oofp=oofp, region=r.region_sdg))
    elif pd.notna(r.imp_t):
        b = np.nanmin([r.rprem_t, r.rav_t, 1.0])
        qh = float(np.clip(mN[0].predict(D[(D.iso3 == r.iso3) & (D.year == YEAR)][NF].astype(float)), 0, 1)[0])
        qr = regmean(Nt, 'rqual_t', r.m49)
        reg_val[i] = r.imp_t * min(b, qr); plug_val[i] = r.imp_t * min(b, qh)
        units.append(dict(i=i, share=1.0, imp=r.imp_t, b=b, qh=qh, res=mN[1], oofp=mN[2], region=r.region_sdg))
    else:
        g = y[(y.region_sdg == r.region_sdg) & y.wat_sm_t.notna()]
        reg_val[i] = plug_val[i] = np.average(g.wat_sm_t, weights=g.pop_t)
        regional_draw.append((i, g.wat_sm_t.values, (g.pop_t / g.pop_t.sum()).values))
P = y.pop_t.values; Ptot = P.sum()
out = dict(year=YEAR, country_preserving_regional_mean=reg_val @ P / Ptot, country_preserving_ml_plugin=plug_val @ P / Ptot)

# expected-minimum totals and intervals for several regional-correlation shares
regions = sorted(y.region_sdg.dropna().unique())
rows = []
for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rng = np.random.RandomState(17)
    SM = np.tile(fixed, (NMC, 1)).astype(float)
    shared = {rg: rng.choice(mS1[1], NMC) for rg in regions}
    for u in units:
        near = np.argsort(np.abs(u['oofp'] - u['qh']))[:40]
        z = lg(u['qh']) + np.sqrt(1 - rho) * rng.choice(u['res'][near], NMC) + np.sqrt(rho) * shared[u['region']]
        SM[:, u['i']] += u['share'] * u['imp'] * np.minimum(u['b'], ex(z))
    for i, vals, p in regional_draw:
        SM[:, i] = rng.choice(vals, NMC, p=p)
    G = SM @ P / Ptot
    rows.append(dict(rho=rho, mean=G.mean(), lo90=np.percentile(G, 5), hi90=np.percentile(G, 95), width90=np.percentile(G, 95) - np.percentile(G, 5),
                     lack_bn=(1 - G.mean() / 100) * Ptot / 1e6))
R = pd.DataFrame(rows)
out['country_preserving_ml_expected_min'] = R.loc[R.rho == 0.25, 'mean'].iloc[0]
E8 = pd.read_csv('results/E08_series_bounds_m49l1.csv'); e8 = E8[(E8.group == 'World') & (E8.year == YEAR)].iloc[0]
out['all_domain_local_regional_E08'] = e8.mean_of_mins; out['aggregate_first_regional_E08'] = e8.jmp_rule
G11 = pd.read_csv(f'results/E11_global_{YEAR}.csv').iloc[0]; out['E11_consistent_ml'] = G11.consistent_ml
out['measured_pop_share'] = P[~np.isnan(y.wat_sm_t.values)].sum() / Ptot
O = pd.Series(out)
pd.set_option('display.width', 200)
print(O.round(3).to_string()); print(R.round(3).to_string(index=False))
O.to_frame().T.to_csv('results/E17_completion_totals_2024.csv', index=False)
R.to_csv('results/E17_regional_correlation_2024.csv', index=False)
