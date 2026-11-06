"""E05: plug-in vs distributional ('expected minimum') imputation of SM from out-of-fold quality predictions.
Plug-in: SM_hat = imp * min(b, q_hat). Distributional: SM_hat = imp * E[min(b, Q)], Q = expit(logit(q_hat) + eps),
eps drawn from logit-scale OOF residuals of OTHER countries (optionally from the k nearest predictions: heteroscedastic).
Usage: python 05_jensen_imputation.py <oof csv> <model column stem>"""
import pandas as pd, numpy as np, sys, os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); import features as F
oof = sys.argv[1] if len(sys.argv) > 1 else 'results/E04_oof_T1_train2024-2024.csv'
stem = sys.argv[2] if len(sys.argv) > 2 else 'CatBoost'
O = pd.read_csv(oof)
d = F.load(); A = F.area_long(d, [2024])
A = A[A.y_rqual.notna() & A.imp_area.notna() & (A.imp_area > 0)].reset_index(drop=True)
assert (A.iso3.values == O.iso3.values).all() and (A.area.values == O.area.values).all()
b = np.nanmin(A[['rprem_own', 'rav_own']].values, axis=1)
imp = A.imp_area.values; y_sm = A.y_sm.values; popw = A['pop'].fillna(0).values

def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))

rows = []
for rep in range(3):
    col = f'{stem}_rep{rep}'
    if col not in O: continue
    qh = O[col].values; qt = O.y_rqual.values
    res = lg(qt) - lg(qh)
    plug = imp * np.minimum(b, qh)
    dist_g, dist_k = np.zeros(len(qh)), np.zeros(len(qh))
    for i in range(len(qh)):
        other = O.iso3.values != O.iso3.values[i]
        e = res[other]
        dist_g[i] = imp[i] * np.mean(np.minimum(b[i], ex(lg(qh[i]) + e)))
        # heteroscedastic: residuals of the 40 other-country rows with closest predicted quality
        idx = np.where(other)[0]; near = idx[np.argsort(np.abs(qh[idx] - qh[i]))[:40]]
        dist_k[i] = imp[i] * np.mean(np.minimum(b[i], ex(lg(qh[i]) + res[near])))
    for name, est in [('plug-in min(b,q_hat)', plug), ('E[min] global residuals', dist_g), ('E[min] local residuals', dist_k), ('upper bound q=1', imp * b)]:
        e = est - y_sm
        rows.append(dict(rep=rep, method=name, bias=e.mean(), rmse=np.sqrt(np.mean(e ** 2)), mae=np.abs(e).mean(),
                         popw_bias=np.average(e, weights=popw + 1e-9),
                         binding_bias=e[qt < b].mean(), nonbinding_bias=e[qt >= b].mean(),
                         agg_err_pp=np.average(est, weights=popw + 1e-9) - np.average(y_sm, weights=popw + 1e-9)))
R = pd.DataFrame(rows).drop(columns='rep').groupby('method').mean()
pd.set_option('display.width', 200)
print(stem, oof); print(R.round(3).to_string())
print('share of rows where quality binds:', round(np.mean(O.y_rqual.values < b), 3))
R.to_csv(f'results/E05_jensen_{stem}_{os.path.basename(oof).replace("E04_oof_", "").replace(".csv", "")}.csv')
