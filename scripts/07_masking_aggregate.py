"""E07: realistic-missingness masking experiment for the AGGREGATE (population-weighted) SM estimate.
Universe = 2024 country-areas with observed area-level quality. Truth = population-weighted mean of area SM
(the JMP country rule applied consistently). Countries are hidden with probability tied to their estimated
propensity of lacking quality data, until ~20-40% of the universe population is hidden.
Estimators: JMP rule (M49 regional-mean imputation, then min of aggregated elements); regional mean + mean of mins;
ML plug-in + mean of mins; ML expected-minimum (Jensen-corrected) + mean of mins, with Monte-Carlo intervals."""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import catboost as cb

N_MASK = int(sys.argv[1]) if len(sys.argv) > 1 else 200
d = F.load()
A = F.area_long(d, [2024]); A = A[A.imp_area.notna() & (A.imp_area > 0) & A['pop'].notna()].reset_index(drop=True)
A['miss'] = A.y_rqual.isna().astype(int)
pc = ['wat_basal_own', 'wat_pip_own', 'rprem_own', 'san_basal_own', 'income', 'log_pop_area', 'is_urban', 'region_fragileoecdany'] + [c for c in A.columns if c.startswith('sdg_')]
pm = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(A[pc], A.miss)
U = A[A.miss == 0].reset_index(drop=True)
U['ps'] = pm.predict_proba(U[pc])[:, 1]
cps = U.groupby('iso3').ps.mean(); cpop = U.groupby('iso3')['pop'].sum()
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area', 'rqual_oth', 'rqual_nat', 'miss', 'ps']
X_cols = [c for c in U.columns if c not in DROP]
U['b'] = np.nanmin(np.column_stack([U.rprem_own.values, U.rav_own.values, np.ones(len(U))]), axis=1)
U['ipop'] = U['pop'] * U.imp_area / 100
truth = np.average(U.y_sm, weights=U['pop'])
EL = ['rprem_own', 'rav_own']


def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))


def jmp_min_of_means(q, frame):
    """Aggregate element ratios among improved users by area, take the minimum, combine urban and rural."""
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


def cbm(seed):
    return cb.CatBoostRegressor(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=3, verbose=0, random_seed=seed, thread_count=1)


def one(m):
    rng = np.random.RandomState(1000 + m)
    target = rng.uniform(0.2, 0.4) * cpop.sum()
    order = list(cps.index); hidden, hp = [], 0.0
    p = (cps / (1 - cps)).values; p = p / p.sum()
    while hp < target:
        c = rng.choice(order, p=p)
        if c in hidden: continue
        hidden.append(c); hp += cpop[c]
    H = U.iso3.isin(hidden).values; V = ~H
    tr, te = U[V], U[H]
    q_true = U.y_rqual.values.copy()
    # regional mean
    q_reg = q_true.copy(); q_reg[H] = regional(tr, te)
    # ML plug-in
    model = cbm(m).fit(tr[X_cols], tr.y_rqual)
    q_ml = q_true.copy(); q_ml[H] = np.clip(model.predict(te[X_cols]), 0, 1)
    # internal grouped CV on visible rows -> logit residual pool for expected-minimum and MC intervals
    vc = tr.iso3.unique(); fm = {c: i % 5 for i, c in enumerate(rng.permutation(vc))}; ff = tr.iso3.map(fm).values
    oof = np.zeros(len(tr))
    for f in range(5):
        mm = cbm(m + f).fit(tr[X_cols][ff != f], tr.y_rqual[ff != f])
        oof[ff == f] = np.clip(mm.predict(tr[X_cols][ff == f]), 0, 1)
    res = lg(tr.y_rqual.values) - lg(oof)
    bH = U.b.values[H]; impH = U.imp_area.values[H]; popH = U['pop'].values[H]
    zq = lg(q_ml[H])
    # expected minimum per hidden row (local residuals by predicted quality)
    e_min = np.zeros(H.sum())
    for i in range(H.sum()):
        near = np.argsort(np.abs(oof - q_ml[H][i]))[:40]
        e_min[i] = np.mean(np.minimum(bH[i], ex(zq[i] + res[near])))
    sm_vis = U.y_sm.values[V]; popV = U['pop'].values[V]
    def agg(smH): return (np.sum(sm_vis * popV) + np.sum(smH * popH)) / (popV.sum() + popH.sum())
    est = {
        'JMP rule (regional mean, min of means)': jmp_min_of_means(q_reg, U),
        'Regional mean, mean of mins': agg(impH * np.minimum(bH, q_reg[H])),
        'ML plug-in, mean of mins': agg(impH * np.minimum(bH, q_ml[H])),
        'ML expected-min, mean of mins': agg(impH * e_min),
        'Upper bound (q=1 for hidden)': agg(impH * bH),
    }
    # MC interval for the expected-min estimator (residuals resampled jointly by region block to allow correlation)
    draws = []
    regs = U.sdg.values[H]
    for s in range(400):
        eps = np.zeros(H.sum())
        for rg in np.unique(regs):
            k = regs == rg
            shared = rng.choice(res) * 0.5  # partial regional correlation
            eps[k] = shared + rng.choice(res, k.sum()) * np.sqrt(0.75)
        draws.append(agg(impH * np.minimum(bH, ex(zq + eps))))
    lo, hi = np.percentile(draws, [5, 95])
    full_jmp = jmp_min_of_means(q_true, U)
    return dict(mask=m, hidden_pop_share=hp / cpop.sum(), n_hidden=len(hidden), truth=truth, jmp_rule_full_data=full_jmp, lo90=lo, hi90=hi, **est)


R = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(m) for m in range(N_MASK)))
R.to_csv('results/E07_masking_aggregate.csv', index=False)
cols = ['JMP rule (regional mean, min of means)', 'Regional mean, mean of mins', 'ML plug-in, mean of mins', 'ML expected-min, mean of mins', 'Upper bound (q=1 for hidden)']
summ = pd.DataFrame({c: dict(bias=(R[c] - R.truth).mean(), rmse=np.sqrt(((R[c] - R.truth) ** 2).mean()), mae=(R[c] - R.truth).abs().mean(),
                             p90_abs=(R[c] - R.truth).abs().quantile(0.9)) for c in cols}).T
pd.set_option('display.width', 200)
print('truth (mean of area mins over universe):', round(truth, 3), '| JMP rule on FULL data:', round(R.jmp_rule_full_data.iloc[0], 3))
print('hidden pop share: median', round(R.hidden_pop_share.median(), 3), 'range', round(R.hidden_pop_share.min(), 3), round(R.hidden_pop_share.max(), 3))
print(summ.round(3).to_string())
cov = ((R.lo90 <= R.truth) & (R.truth <= R.hi90)).mean()
print('90% MC interval coverage (expected-min):', round(cov, 3), '| median width', round((R.hi90 - R.lo90).median(), 3))
summ.to_csv('results/E07_masking_summary.csv')
