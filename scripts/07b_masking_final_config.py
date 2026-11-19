"""E07b: country-masking experiment using the configuration of the final completion (E11).
Universe and reference: 2024 urban/rural domains with observed quality; the reference total is the population-weighted
mean of their local minima, i.e. the country-consistent target. Hidden countries are drawn as in E07 (missingness-odds
weighting until 20-40% of the universe population is hidden). Differences from E07: quality models are trained on all
2000-2024 records of visible countries with inverse record-count weights (CatBoost 600 iterations, learning rate 0.03,
depth 4, L2 3); residuals come from five-fold country holdouts among visible countries; expected minima use the forty
nearest predicted-quality residuals; 90% intervals combine sqrt(0.75)-scaled local residuals with a 0.5-scaled component
shared by all hidden domains of an SDG region (the E11 scheme). Also: single-country masks for the five most populous
countries in the universe, and paired bootstrap intervals (over masks) for differences in RMSE between estimators.
Usage: python 07b_masking_final_config.py [n_masks]"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import catboost as cb

N_MASK = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NMC = 400
d = F.load()
L = F.area_long(d, list(range(2000, 2025)))
L = L[L.imp_area.notna() & (L.imp_area > 0)].reset_index(drop=True)
A = L[(L.year == 2024) & L['pop'].notna()].reset_index(drop=True)
A['miss'] = A.y_rqual.isna().astype(int)
pc = ['wat_basal_own', 'wat_pip_own', 'rprem_own', 'san_basal_own', 'income', 'log_pop_area', 'is_urban', 'region_fragileoecdany'] + [c for c in A.columns if c.startswith('sdg_')]
pm = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(A[pc], A.miss)
U = A[A.miss == 0].reset_index(drop=True)
U['ps'] = pm.predict_proba(U[pc])[:, 1]
cps = U.groupby('iso3').ps.mean(); cpop = U.groupby('iso3')['pop'].sum()
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area', 'rqual_oth', 'rqual_nat', 'miss', 'ps']
X = [c for c in L.columns if c not in DROP]
U['b'] = np.nanmin(np.column_stack([U.rprem_own.values, U.rav_own.values, np.ones(len(U))]), axis=1)
U['ipop'] = U['pop'] * U.imp_area / 100
truth = np.average(U.y_sm, weights=U['pop'])
T = L[L.y_rqual.notna()].reset_index(drop=True)                       # supervised records, all years
T['w'] = 1.0 / T.groupby(['iso3', 'area']).year.transform('count')
EL = ['rprem_own', 'rav_own']


def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))
def cbm(seed): return cb.CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=3, verbose=0, random_seed=seed, thread_count=1)


def jmp_min_of_means(q, frame):
    out, popsum = 0.0, frame['pop'].sum()
    for a in ['u', 'r']:
        g = frame.area.values == a; w = frame.ipop.values[g]
        W1 = np.average(frame.imp_area.values[g], weights=frame['pop'].values[g])
        rr = [np.average(q[g], weights=w)]
        for e in EL:
            v = frame[e].values[g]; ok = ~np.isnan(v)
            if ok.any(): rr.append(np.average(v[ok], weights=w[ok]))
        out += frame['pop'].values[g].sum() / popsum * W1 * min(rr)
    return out


def regional(train, test):
    res = []
    for _, r in test.iterrows():
        for k in ['m49', 'sdg', None]:
            g = train[train.area == r.area] if k is None else train[(train[k] == r[k]) & (train.area == r.area)]
            if len(g): res.append(np.average(g.y_rqual, weights=g.ipop + 1e-9)); break
    return np.array(res)


def evaluate(hidden, seed):
    rng = np.random.RandomState(seed)
    H = U.iso3.isin(hidden).values; V = ~H
    tr24, te = U[V], U[H]
    q_true = U.y_rqual.values.copy()
    q_reg = q_true.copy(); q_reg[H] = regional(tr24, te)
    TT = T[~T.iso3.isin(hidden)].reset_index(drop=True)
    model = cbm(seed).fit(TT[X], TT.y_rqual, sample_weight=TT.w)
    q_ml = q_true.copy(); q_ml[H] = np.clip(model.predict(te[X]), 0, 1)
    vc = TT.iso3.unique(); fm = {c: i % 5 for i, c in enumerate(rng.permutation(vc))}; ff = TT.iso3.map(fm).values
    t24 = (TT.year == 2024).values; oof = np.full(len(TT), np.nan)
    for f in range(5):
        mm = cbm(seed + f).fit(TT[X][ff != f], TT.y_rqual[ff != f], sample_weight=TT.w[ff != f])
        k = (ff == f) & t24
        oof[k] = np.clip(mm.predict(TT[X][k]), 0, 1)
    oofp = oof[t24]; res = lg(TT.y_rqual.values[t24]) - lg(oofp)
    bH = U.b.values[H]; impH = U.imp_area.values[H]; popH = U['pop'].values[H]; zq = lg(q_ml[H]); regs = U.sdg.values[H]
    near = [np.argsort(np.abs(oofp - q_ml[H][i]))[:40] for i in range(H.sum())]
    e_min = np.array([np.mean(np.minimum(bH[i], ex(zq[i] + res[near[i]]))) for i in range(H.sum())])
    sm_vis = U.y_sm.values[V]; popV = U['pop'].values[V]
    def agg(smH): return (np.sum(sm_vis * popV) + np.sum(smH * popH)) / (popV.sum() + popH.sum())
    est = {'aggregate_first_regional': jmp_min_of_means(q_reg, U), 'local_regional': agg(impH * np.minimum(bH, q_reg[H])),
           'local_ml_plugin': agg(impH * np.minimum(bH, q_ml[H])), 'local_ml_expected_min': agg(impH * e_min),
           'upper_bound': agg(impH * bH)}
    draws = np.zeros(NMC)
    for s in range(NMC):
        shared = {rg: rng.choice(res) * 0.5 for rg in np.unique(regs)}
        eps = np.array([rng.choice(res[near[i]]) * np.sqrt(0.75) + shared[regs[i]] for i in range(H.sum())])
        draws[s] = agg(impH * np.minimum(bH, ex(zq + eps)))
    lo, hi = np.percentile(draws, [5, 95])
    return dict(hidden_pop_share=popH.sum() / U['pop'].sum(), n_hidden=len(hidden), truth=truth, lo90=lo, hi90=hi, **est)


def one(m):
    rng = np.random.RandomState(1000 + m)
    target = rng.uniform(0.2, 0.4) * cpop.sum()
    order = list(cps.index); hidden, hp = [], 0.0
    p = (cps / (1 - cps)).values; p = p / p.sum()
    while hp < target:
        c = rng.choice(order, p=p)
        if c in hidden: continue
        hidden.append(c); hp += cpop[c]
    return dict(mask=m, kind='propensity', hidden='|'.join(sorted(hidden)), **evaluate(hidden, m))


def single(iso, i):
    return dict(mask=N_MASK + i, kind='single country', hidden=iso, **evaluate([iso], 5000 + i))


big = list(cpop.sort_values(ascending=False).index[:5])
jobs = [delayed(one)(m) for m in range(N_MASK)] + [delayed(single)(c, i) for i, c in enumerate(big)]
R = pd.DataFrame(Parallel(n_jobs=24)(jobs))
R.to_csv('results/E07b_masking_final_config.csv', index=False)
EST = ['aggregate_first_regional', 'local_regional', 'local_ml_plugin', 'local_ml_expected_min', 'upper_bound']
P = R[R.kind == 'propensity'].reset_index(drop=True)
err = {c: (P[c] - P.truth).values for c in EST}
rows = []
for c in EST:
    e = err[c]
    rows.append(dict(estimator=c, bias=e.mean(), rmse=np.sqrt((e ** 2).mean()), mae=np.abs(e).mean(), p90_abs=np.quantile(np.abs(e), .9)))
S = pd.DataFrame(rows)
# paired bootstrap over masks for RMSE differences against the local regional estimator
brng = np.random.RandomState(7); B = 2000; idx = brng.randint(0, len(P), (B, len(P)))
for c in EST:
    dif = np.sqrt((err[c][idx] ** 2).mean(1)) - np.sqrt((err['local_regional'][idx] ** 2).mean(1))
    S.loc[S.estimator == c, 'rmse_minus_local_regional'] = np.sqrt((err[c] ** 2).mean()) - np.sqrt((err['local_regional'] ** 2).mean())
    S.loc[S.estimator == c, 'diff_lo95'] = np.percentile(dif, 2.5); S.loc[S.estimator == c, 'diff_hi95'] = np.percentile(dif, 97.5)
cov = ((P.lo90 <= P.truth) & (P.truth <= P.hi90)).mean()
S['coverage90_expected_min'] = np.where(S.estimator == 'local_ml_expected_min', cov, np.nan)
S['median_width90'] = np.where(S.estimator == 'local_ml_expected_min', (P.hi90 - P.lo90).median(), np.nan)
S.to_csv('results/E07b_masking_final_config_summary.csv', index=False)
pd.set_option('display.width', 250)
print('reference total', round(truth, 3), '| hidden share median', round(P.hidden_pop_share.median(), 3), 'IQR',
      round(P.hidden_pop_share.quantile(.25), 3), round(P.hidden_pop_share.quantile(.75), 3), 'range', round(P.hidden_pop_share.min(), 3), round(P.hidden_pop_share.max(), 3))
print(S.round(3).to_string(index=False))
Sg = R[R.kind == 'single country']
print(Sg[['hidden', 'hidden_pop_share', 'truth'] + EST + ['lo90', 'hi90']].round(2).to_string(index=False))
