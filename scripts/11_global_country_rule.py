"""E11: global and regional SM under the JMP country rule applied consistently: population-weighted average of the JMP's
published country estimates (fixed) plus probabilistic imputations for unmeasured countries.
Unmeasured countries:
  area-level elements available -> area models S1 (no quality info) or S3 (other area's quality known), expected minimum per area;
                                   a published area SM (e.g. China urban) is kept as is;
  only national elements         -> national model (N), expected minimum at national level;
  no water data (ARG, NIC, FLK)  -> draws from population-weighted SM of measured countries in the same SDG region.
Also: worst-case (Manski) bounds with quality in [0,1] for unmeasured countries, value of information, and (added 2026-10-26)
the household joint criterion under independence and its Frechet lower bound computed on the SAME draws (published criteria for
measured countries; ML quality draws for unmeasured ones), for the world and for low- and middle-income countries.
Usage: python 11_global_country_rule.py <year>"""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); import features as F
import catboost as cb

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
NMC = 2000
rng = np.random.RandomState(11)
d = F.load()


def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))
def model(seed): return cb.CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=3, verbose=0, random_seed=seed, thread_count=5)


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


SH = {}   # one regional draw vector per SDG region, filled after the models are fitted


def draws(qhat, res, oofp, regs):
    """Local-residual draws in logit space plus the region's shared component SH[region]; returns NMC x n quality draws.
    Fixed 2026-10-26: the regional component is now common to every country-area of an SDG region. Earlier runs called this
    function one unit at a time and drew the component afresh each call, i.e. independently across countries, which made the
    global interval too narrow (old version: scripts/history/11_global_country_rule_before_shared_fix_20261026.py)."""
    Z = np.zeros((NMC, len(qhat)))
    for i in range(len(qhat)):
        near = np.argsort(np.abs(oofp - qhat[i]))[:40]
        reg = SH[regs[i]] if regs[i] in SH else rng.choice(res, NMC) * 0.5
        Z[:, i] = lg(qhat[i]) + rng.choice(res[near], NMC) * np.sqrt(0.75) + reg
    return ex(Z)


def joint(imp, rp, ra, q):
    """Independence value and Frechet lower bound of the joint criterion from available ratios (q may be an array)."""
    el = [v for v in (rp, ra) if pd.notna(v)]
    q = np.asarray(q, float); k = len(el) + 1
    prod = q * np.prod(el) if el else q
    low = np.maximum(0, q + sum(el) - (k - 1))
    return imp * prod, imp * low


# ---------- area-level models ----------
L = F.area_long(d, list(range(2000, YEAR + 1)))
L = L[L.imp_area.notna() & (L.imp_area > 0)].reset_index(drop=True)
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area', 'rqual_oth', 'rqual_nat']
BASE = [c for c in L.columns if c not in DROP]
obs = L[L.y_rqual.notna()].reset_index(drop=True)
mS1 = fit_with_oof(obs, BASE, 'y_rqual', ['iso3', 'area'])
mS3 = fit_with_oof(obs[obs.rqual_oth.notna()].reset_index(drop=True), BASE + ['rqual_oth'], 'y_rqual', ['iso3', 'area'])
# ---------- national-level model ----------
NV = [c for c in d.columns if c.endswith('_t') and (c.startswith('wat_') or c.startswith('san_') or c.startswith('hyg_')) and 'qual' not in c and 'sm' not in c and 'arc' not in c] + ['rprem_t', 'rav_t', 'imp_t']
NV = [c for c in NV if c not in ('wat_imp_prem_t', 'wat_imp_av_t')]
D = d.copy(); D['income_n'] = D.region_income.map(F.INCOME); D['log_pop'] = np.log10(D.pop_t.clip(lower=0.1))
for r in sorted(D.region_sdg.dropna().unique()): D['sdg_' + r[:12]] = (D.region_sdg == r).astype(int)
NF = NV + ['prop_u', 'income_n', 'log_pop'] + [c for c in D.columns if c.startswith('sdg_')]
Dn = D[D.rqual_t.notna() & (D.year <= YEAR)].reset_index(drop=True)
mN = fit_with_oof(Dn, NF, 'rqual_t', ['iso3'])
print('OOF logit-residual SD  S1 %.3f  S3 %.3f  N %.3f' % (np.std(mS1[1]), np.std(mS3[1]), np.std(mN[1])))
# regional component (sd = half the S1 residual sd), shared by all country-areas of a region whichever model imputes them
SH.update({rg: rng.choice(mS1[1], NMC) * 0.5 for rg in sorted(d.region_sdg.dropna().unique())})

# ---------- assemble countries ----------
y = d[(d.year == YEAR) & d.pop_t.notna()].reset_index(drop=True)
Lc = L[L.year == YEAR].set_index(['iso3', 'area'])
SMc = np.zeros((NMC, len(y))); kind = []
INDc = np.zeros((NMC, len(y))); LOWc = np.zeros((NMC, len(y)))
for i, r in y.iterrows():
    if pd.notna(r.wat_sm_t):
        SMc[:, i] = r.wat_sm_t; kind.append('measured')
        both = all((r.iso3, a) in Lc.index and pd.notna(Lc.loc[(r.iso3, a)].y_rqual) for a in 'ur')
        if both:
            pu = r.prop_u / 100; ii, ll = 0, 0
            for a, sh in [('u', pu), ('r', 1 - pu)]:
                row = Lc.loc[(r.iso3, a)]; a_, b_ = joint(row.imp_area, row.rprem_own, row.rav_own, row.y_rqual); ii += sh * a_; ll += sh * b_
        else:
            ii, ll = joint(r.imp_t, r.rprem_t, r.rav_t, r.rqual_t)
        INDc[:, i] = ii; LOWc[:, i] = ll
        continue
    areas_ok = all((r.iso3, a) in Lc.index for a in 'ur')
    if areas_ok:
        parts, jparts = [], []
        for a in 'ur':
            row = Lc.loc[(r.iso3, a)]
            b = np.nanmin([row.rprem_own, row.rav_own, 1.0])
            if pd.notna(row.y_sm):
                parts.append(np.full(NMC, row.y_sm)); jparts.append(joint(row.imp_area, row.rprem_own, row.rav_own, np.full(NMC, row.y_rqual))); continue
            if pd.notna(row.rqual_oth):
                m, res, oofp = mS3; feats = BASE + ['rqual_oth']
            else:
                m, res, oofp = mS1; feats = BASE
            qh = np.clip(m.predict(row[feats].to_frame().T.astype(float)), 0, 1)
            Q = draws(qh, res, oofp, np.array([r.region_sdg]))[:, 0]
            parts.append(row.imp_area * np.minimum(b, Q)); jparts.append(joint(row.imp_area, row.rprem_own, row.rav_own, Q))
        pu = r.prop_u / 100
        SMc[:, i] = pu * parts[0] + (1 - pu) * parts[1]; kind.append('imputed-area')
        INDc[:, i] = pu * jparts[0][0] + (1 - pu) * jparts[1][0]; LOWc[:, i] = pu * jparts[0][1] + (1 - pu) * jparts[1][1]
    elif pd.notna(r.imp_t):
        b = np.nanmin([r.rprem_t, r.rav_t, 1.0])
        xr = D[(D.iso3 == r.iso3) & (D.year == YEAR)][NF].astype(float)
        qh = np.clip(mN[0].predict(xr), 0, 1)
        Q = draws(qh, mN[1], mN[2], np.array([r.region_sdg]))[:, 0]
        SMc[:, i] = r.imp_t * np.minimum(b, Q); kind.append('imputed-national')
        INDc[:, i], LOWc[:, i] = joint(r.imp_t, r.rprem_t, r.rav_t, Q)
    else:
        g = y[(y.region_sdg == r.region_sdg) & y.wat_sm_t.notna()]
        SMc[:, i] = rng.choice(g.wat_sm_t.values, NMC, p=(g.pop_t / g.pop_t.sum()).values); kind.append('regional-draw')
y['kind'] = kind
# countries without any water data: joint values scale the drawn SM by the regional ratios of measured countries
for i in np.where(y.kind.values == 'regional-draw')[0]:
    g = (y.region_sdg.values == y.region_sdg[i]) & (y.kind.values == 'measured')
    w_ = y.pop_t.values[g]; s_ = SMc[0, g]
    INDc[:, i] = SMc[:, i] * np.sum(w_ * INDc[0, g]) / np.sum(w_ * s_); LOWc[:, i] = SMc[:, i] * np.sum(w_ * LOWc[0, g]) / np.sum(w_ * s_)
P = y.pop_t.values; Ptot = P.sum()
G = SMc @ P / Ptot
y['sm_mean'] = SMc.mean(0); y['sm_lo'] = np.percentile(SMc, 5, 0); y['sm_hi'] = np.percentile(SMc, 95, 0)
print(y.groupby('kind').pop_t.agg(['count', 'sum']).assign(bn=lambda x: x['sum'] / 1e6)[['count', 'bn']])

# worst-case bounds for unmeasured countries (quality in [0,1]); SM <= min(premises, availability) <= improved <= 100
meas = y.wat_sm_t.notna().values
ub_c = y[['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(y.imp_t).fillna(100).values
upper = (np.sum(P[meas] * y.wat_sm_t.values[meas]) + np.sum(P[~meas] * ub_c[~meas])) / Ptot
lower = np.sum(P[meas] * y.wat_sm_t.values[meas]) / Ptot
out = dict(year=YEAR, pop_bn=Ptot / 1e6, measured_pop_share=P[meas].sum() / Ptot,
           mean_measured_only=np.average(y.wat_sm_t.values[meas], weights=P[meas]),
           manski_lower=lower, manski_upper=upper,
           consistent_ml=G.mean(), consistent_ml_lo90=np.percentile(G, 5), consistent_ml_hi90=np.percentile(G, 95))
for k in ['manski_upper', 'consistent_ml', 'consistent_ml_lo90', 'consistent_ml_hi90', 'manski_lower']:
    out['lack_bn_' + k] = (1 - out[k] / 100) * Ptot / 1e6
print(pd.Series(out).round(3).to_string())
grp = []
LMIC = y.region_income.isin(['Low income', 'Lower middle income', 'Upper middle income']).values
for gname, k in [('World', np.ones(len(y), bool)), ('LMIC (WB 2025 groups)', LMIC)]:
    Pk = P[k].sum(); row = dict(year=YEAR, group=gname, pop_bn=Pk / 1e6)
    for nm, A in [('country_rule_ml', SMc), ('independence_ml', INDc), ('frechet_lower_ml', LOWc)]:
        v = A[:, k] @ P[k] / Pk
        row[nm] = v.mean(); row[nm + '_lo90'] = np.percentile(v, 5); row[nm + '_hi90'] = np.percentile(v, 95)
        row['lack_bn_' + nm] = (1 - v.mean() / 100) * Pk / 1e6
        row['lack_bn_' + nm + '_lo90'] = (1 - np.percentile(v, 95) / 100) * Pk / 1e6; row['lack_bn_' + nm + '_hi90'] = (1 - np.percentile(v, 5) / 100) * Pk / 1e6
    grp.append(row)
GRP = pd.DataFrame(grp); print(GRP.round(3).T.to_string())
GRP.to_csv(f'results/E11_joint_{YEAR}.csv', index=False)

# regional
reg = []
for rg in sorted(y.region_sdg.dropna().unique()):
    k = (y.region_sdg == rg).values
    v = SMc[:, k] @ P[k] / P[k].sum()
    mk = k & meas
    reg.append(dict(region=rg, pop_M=P[k].sum() / 1e3, measured_share=P[mk].sum() / P[k].sum(),
                    measured_only=np.average(y.wat_sm_t.values[mk], weights=P[mk]) if mk.any() else np.nan,
                    consistent_ml=v.mean(), lo90=np.percentile(v, 5), hi90=np.percentile(v, 95),
                    upper_bound=(np.sum(P[mk] * y.wat_sm_t.values[mk]) + np.sum(P[k & ~meas] * ub_c[k & ~meas])) / P[k].sum()))
R = pd.DataFrame(reg); print(R.round(2).to_string(index=False))

# value of information (country level)
v0 = G.var(); voi = []
for i in np.where(~meas)[0]:
    gi = G - (SMc[:, i] - SMc[:, i].mean()) * P[i] / Ptot
    voi.append(dict(iso3=y.iso3[i], name=y.name[i], kind=y.kind[i], pop_M=P[i] / 1e3, sm_mean=SMc[:, i].mean(), sm_lo=np.percentile(SMc[:, i], 5),
                    sm_hi=np.percentile(SMc[:, i], 95), upper_bound=ub_c[i], var_reduction_share=1 - gi.var() / v0))
V = pd.DataFrame(voi).sort_values('var_reduction_share', ascending=False); V['cum_share'] = V.var_reduction_share.cumsum()
print(V.head(12).round(3).to_string(index=False))
y[['iso3', 'name', 'region_sdg', 'region_income', 'pop_t', 'wat_sm_t', 'kind', 'sm_mean', 'sm_lo', 'sm_hi']].to_csv(f'results/E11_country_estimates_{YEAR}.csv', index=False)
pd.DataFrame([out]).to_csv(f'results/E11_global_{YEAR}.csv', index=False)
R.to_csv(f'results/E11_regional_{YEAR}.csv', index=False)
V.to_csv(f'results/E11_voi_{YEAR}.csv', index=False)
np.save(f'results/E11_global_draws_{YEAR}.npy', G)
# per-country draws (2026-10-26): greedy measurement design and identification-width maps
np.savez_compressed(f'results/E11_country_draws_{YEAR}.npz', iso3=y.iso3.values, pop=P, sm=SMc.astype(np.float32), ind=INDc.astype(np.float32), low=LOWc.astype(np.float32))
