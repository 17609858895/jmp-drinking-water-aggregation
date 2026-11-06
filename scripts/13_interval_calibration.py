"""E13: country-level calibration of the predictive distribution of safely managed coverage (whole country held out).
For each country-area, quality residuals come only from OTHER countries' out-of-fold residuals (logit scale); the SM draw is
imp * min(b, expit(logit q_hat + e)). Residual-pool rules: global; local in predicted quality (k = 40); local in the predicted
distance to the bottleneck g = logit(b) - logit(q_hat) (k = 40); local in both. Calibration is judged on prediction-side groups
(tercile of the predicted probability that quality binds), never on outcome-defined groups.
Usage: python 13_interval_calibration.py [oof tag]"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
TAG = sys.argv[1] if len(sys.argv) > 1 else 'T1_train2000-2024'
O = pd.read_csv(f'results/E04_oof_{TAG}.csv')
A = F.area_long(F.load(), [2024]); A = A[A.y_rqual.notna() & A.imp_area.notna() & (A.imp_area > 0)].reset_index(drop=True)
assert (A.iso3.values == O.iso3.values).all()
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))
b = np.nanmin(np.column_stack([A.rprem_own, A.rav_own, np.ones(len(A))]), axis=1)
imp = A.imp_area.values; y = A.y_sm.values; iso = A.iso3.values; sdg = A.sdg.values
qh = O[[c for c in O.columns if c.startswith('CatBoost_rep')]].mean(1).values
res = lg(O.y_rqual.values) - lg(qh)
g = lg(np.minimum(b, .999)) - lg(qh)
K = 40
rules = {'Global pool': None, 'Local: predicted quality': 'q', 'Local: distance to bottleneck': 'g', 'Local: both': 'qg'}
rows, per = [], []
for name, rule in rules.items():
    mean, lo, hi, pbind = (np.zeros(len(A)) for _ in range(4))
    for i in range(len(A)):
        other = np.where(iso != iso[i])[0]
        if rule is None: pool = other
        else:
            if rule == 'q': dist = np.abs(qh[other] - qh[i])
            elif rule == 'g': dist = np.abs(g[other] - g[i])
            else: dist = np.hypot((qh[other] - qh[i]) / qh.std(), (g[other] - g[i]) / g.std())
            pool = other[np.argsort(dist)[:K]]
        Q = ex(lg(qh[i]) + res[pool]); S = imp[i] * np.minimum(b[i], Q)
        mean[i], lo[i], hi[i] = S.mean(), np.percentile(S, 5), np.percentile(S, 95); pbind[i] = (Q < b[i]).mean()
    cov = (lo <= y + 1e-9) & (y <= hi + 1e-9)
    wink = (hi - lo) + 2 / .1 * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    ter = pd.qcut(pbind + np.random.RandomState(0).uniform(0, 1e-6, len(pbind)), 3, labels=['low', 'mid', 'high'])
    r = dict(rule=name, rmse=np.sqrt(np.mean((mean - y) ** 2)), bias=np.mean(mean - y), coverage90=cov.mean(), width=np.mean(hi - lo),
             winkler=np.mean(wink), iw_coverage90=np.average(cov, weights=O.iw.values))
    for t in ['low', 'mid', 'high']:
        k = (ter == t)
        r[f'bias_pbind_{t}'] = np.mean((mean - y)[k]); r[f'cov_pbind_{t}'] = cov[k].mean()
    for reg in ['Sub-Saharan Africa', 'Latin America and the Caribbean', 'Europe and Northern America']:
        k = sdg == reg; r['cov_' + reg.split()[0]] = cov[k].mean()
    rows.append(r)
    per.append(pd.DataFrame({'rule': name, 'iso3': iso, 'area': A.area.values, 'reported': y, 'mean': mean, 'lo90': lo, 'hi90': hi, 'p_bind': pbind, 'covered': cov}))
R = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(TAG); print(R.round(3).T.to_string())
R.to_csv(f'results/E13_interval_calibration_{TAG}.csv', index=False)
pd.concat(per).to_csv(f'results/E13_interval_units_{TAG}.csv', index=False)

# ---- Mondrian (group-conditional) split-conformal widening on top of the global-pool predictive interval.
# Groups = terciles of the predicted probability that quality binds (prediction-side). Score = distance outside the interval.
# Leave-one-country-out: a unit's widening uses scores from other countries in its group only.
U = per[0].copy()                      # global pool
U['score'] = np.maximum.reduce([U.lo90 - U.reported, U.reported - U.hi90, np.zeros(len(U))])
U['grp'] = pd.qcut(U.p_bind + np.random.RandomState(0).uniform(0, 1e-6, len(U)), 3, labels=False)
for label, by_group in [('Conformal, pooled', False), ('Conformal, Mondrian by binding', True)]:
    lo2, hi2 = U.lo90.values.copy(), U.hi90.values.copy()
    for i in range(len(U)):
        k = (U.iso3.values != U.iso3.values[i]) & ((U.grp.values == U.grp.values[i]) if by_group else True)
        s = U.score.values[k]; n = len(s)
        qv = np.sort(s)[min(n - 1, int(np.ceil((n + 1) * .9)) - 1)]
        lo2[i] -= qv; hi2[i] += qv
    lo2 = np.maximum(lo2, 0); hi2 = np.minimum(hi2, imp)   # SM can never exceed improved coverage
    cov2 = (lo2 <= y + 1e-9) & (y <= hi2 + 1e-9)
    r = dict(rule=label, coverage90=cov2.mean(), width=np.mean(hi2 - lo2), iw_coverage90=np.average(cov2, weights=O.iw.values))
    for t, nm in enumerate(['low', 'mid', 'high']):
        r[f'cov_pbind_{nm}'] = cov2[U.grp.values == t].mean(); r[f'width_pbind_{nm}'] = (hi2 - lo2)[U.grp.values == t].mean()
    for reg in ['Sub-Saharan Africa', 'Latin America and the Caribbean', 'Europe and Northern America']:
        r['cov_' + reg.split()[0]] = cov2[sdg == reg].mean()
    rows.append(r)
    per.append(pd.DataFrame({'rule': label, 'iso3': iso, 'area': A.area.values, 'reported': y, 'mean': U['mean'].values, 'lo90': lo2, 'hi90': hi2,
                             'p_bind': U.p_bind.values, 'covered': cov2}))
R = pd.DataFrame(rows)
print(R[['rule', 'coverage90', 'width', 'iw_coverage90', 'cov_pbind_low', 'cov_pbind_mid', 'cov_pbind_high', 'cov_Sub-Saharan', 'cov_Latin', 'cov_Europe']].round(3).to_string(index=False))
R.to_csv(f'results/E13_interval_calibration_{TAG}.csv', index=False)
pd.concat(per).to_csv(f'results/E13_interval_units_{TAG}.csv', index=False)
