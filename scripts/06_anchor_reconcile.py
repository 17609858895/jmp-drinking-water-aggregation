"""E06: national-identity reconciliation for national-only countries (scenario T2).
Given out-of-fold area predictions q_u, q_r and the national quality element W10_t, find a common logit shift delta with
pu*imp_u*expit(logit q_u + delta) + (1-pu)*imp_r*expit(logit q_r + delta) = W10_t, then recompute area SM."""
import pandas as pd, numpy as np, sys, os
from scipy.optimize import brentq
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); import features as F
tag = sys.argv[1] if len(sys.argv) > 1 else 'T2_train2000-2024'
O = pd.read_csv(f'results/E04_oof_{tag}.csv')
d = F.load(); A = F.area_long(d, [2024])
A = A[A.y_rqual.notna() & A.imp_area.notna() & (A.imp_area > 0) & A.rqual_nat.notna()].reset_index(drop=True)
assert (A.iso3.values == O.iso3.values).all()
b = np.nanmin(np.column_stack([A.rprem_own, A.rav_own, np.ones(len(A))]), axis=1)
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))
rows = []
models = sorted({c.rsplit('_rep', 1)[0] for c in O.columns if '_rep' in c})
for mname in models:
    for rep in range(3):
        col = f'{mname}_rep{rep}'
        if col not in O: continue
        q = O[col].values.copy(); qr = q.copy()
        for c in A.iso3.unique():
            k = np.where((A.iso3 == c).values)[0]
            if len(k) != 2: continue
            pu = A.pu.values[k[0]]; w = np.array([pu if A.area.values[i] == 'u' else 1 - pu for i in k]) * A.imp_area.values[k]
            tgt = A.W10_nat.values[k[0]]
            f = lambda dl: np.sum(w * ex(lg(q[k]) + dl)) - tgt
            try: qr[k] = ex(lg(q[k]) + brentq(f, -20, 20))
            except ValueError: pass
        for nm, qq in [('raw', q), ('reconciled', qr)]:
            sm = A.imp_area.values * np.minimum(b, qq); e = sm - A.y_sm.values
            rows.append(dict(model=mname, version=nm, rep=rep, rmse_sm=np.sqrt(np.mean(e ** 2)), mae_sm=np.abs(e).mean(),
                             r2_sm=1 - np.sum(e ** 2) / np.sum((A.y_sm - A.y_sm.mean()) ** 2), bias=e.mean()))
R = pd.DataFrame(rows).drop(columns='rep').groupby(['model', 'version']).mean().unstack('version')
R = R.sort_values(('rmse_sm', 'reconciled'))
pd.set_option('display.width', 200)
print(tag, '| country-areas', len(A), 'countries with both areas', (A.groupby('iso3').size() == 2).sum())
print(R.round(3).to_string())
R.to_csv(f'results/E06_anchor_reconcile_{tag}.csv')
