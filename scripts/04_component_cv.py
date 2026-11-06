"""E04: leave-country-out prediction of the missing water-quality element (area level) and the implied SM.
Scenario T1 = whole country missing (no national/other-area quality); T2 = national anchor known (national-only countries).
Usage: python 04_component_cv.py <train_years comma list or a-b range> <T1|T2>
"""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__)); import features as F
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import nnls
import lightgbm as lgb, xgboost as xgb, catboost as cb

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
YEAR_TEST = 2024
arg = sys.argv[1] if len(sys.argv) > 1 else '2024'
TRAIN_YEARS = list(range(int(arg.split('-')[0]), int(arg.split('-')[1]) + 1)) if '-' in arg else [int(a) for a in arg.split(',')]
SCEN = sys.argv[2] if len(sys.argv) > 2 else 'T1'
d = F.load()
L = F.area_long(d, sorted(set(TRAIN_YEARS + [YEAR_TEST])))
L = L[L.y_rqual.notna() & L.imp_area.notna() & (L.imp_area > 0)].reset_index(drop=True)
if SCEN == 'T2':  # test only where a national anchor exists (as for national-only countries)
    L = L[L.rqual_nat.notna() | (L.year != YEAR_TEST)].reset_index(drop=True)
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area']
LEAK = {'T1': ['rqual_oth', 'rqual_nat'], 'T2': ['rqual_oth']}[SCEN]
X_cols = [c for c in L.columns if c not in DROP + LEAK]
print('scenario', SCEN, 'train years', TRAIN_YEARS[0], '-', TRAIN_YEARS[-1], 'rows', len(L), 'features', len(X_cols))
test_mask_all = (L.year == YEAR_TEST)
countries = L.loc[test_mask_all, 'iso3'].unique()
print('test country-areas', test_mask_all.sum(), 'countries', len(countries))


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def models():
    imp = lambda: SimpleImputer(strategy='median', add_indicator=True)
    return {
        'Ridge': ('logit', make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))),
        'kNN': ('raw', make_pipeline(imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights='distance'))),
        'RF': ('raw', make_pipeline(imp(), RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.4, n_jobs=16, random_state=0))),
        'ET': ('raw', make_pipeline(imp(), ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5, n_jobs=16, random_state=0))),
        'HGB': ('raw', HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, l2_regularization=1.0, random_state=0)),
        'LGBM': ('raw', lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=8, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, random_state=0, n_jobs=8)),
        'XGB': ('raw', xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                                        min_child_weight=3, n_jobs=8, random_state=0)),
        'CatBoost': ('raw', cb.CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=3, verbose=0, random_seed=0, thread_count=8)),
    }


def fit(name, m, X, y, sw):
    if name == 'kNN':
        return m.fit(X, y)
    if hasattr(m, 'steps'):
        return m.fit(X, y, **{m.steps[-1][0] + '__sample_weight': sw})
    return m.fit(X, y, sample_weight=sw)


def regional_mean(train, test, key):
    out = []
    for _, r in test.iterrows():
        for k in [key, 'sdg', None]:
            g = train[train.area == r.area] if k is None else train[(train[k] == r[k]) & (train.area == r.area)]
            if len(g):
                out.append(np.average(g.y_rqual, weights=(g['pop'] * g.imp_area).fillna(0).clip(lower=1e-6))); break
    return np.array(out)


REPEATS = 3
preds = {}
for rep in range(REPEATS):
    rng = np.random.RandomState(rep)
    perm = {c: i for i, c in enumerate(rng.permutation(countries))}
    folds = L.iso3.map(perm).fillna(-1).astype(int).values
    folds = np.where(folds >= 0, folds % 10, -1)
    for f in range(10):
        te = test_mask_all.values & (folds == f)
        tr = (folds != f) & L.year.isin(TRAIN_YEARS).values
        if te.sum() == 0:
            continue
        Ltr, Lte = L[tr], L[te]
        sw = (1.0 / Ltr.groupby(['iso3', 'area']).year.transform('count')).values
        p = {'RegionalMean_M49': regional_mean(Ltr[Ltr.year == YEAR_TEST], Lte, 'm49'),
             'RegionalMean_SDG': regional_mean(Ltr[Ltr.year == YEAR_TEST], Lte, 'sdg')}
        if SCEN == 'T2':
            p['NationalRatio'] = Lte.rqual_nat.values
        for name, (tt, m) in models().items():
            y = logit(Ltr.y_rqual.values) if tt == 'logit' else Ltr.y_rqual.values
            fit(name, m, Ltr[X_cols], y, sw)
            yp = m.predict(Lte[X_cols])
            p[name] = np.clip(expit(yp) if tt == 'logit' else yp, 0, 1)
        for k, v in p.items():
            preds.setdefault((rep, k), pd.Series(index=L.index, dtype=float)).loc[Lte.index] = v
P = {k: v[test_mask_all] for k, v in preds.items()}
for k, v in P.items():
    if v.isna().any(): print("NaN preds", k, v.isna().sum())
T = L[test_mask_all].copy()

# NNLS super learner, weights estimated on other countries' OOF predictions (outer 5-fold by country)
base = [k for (r, k) in P if r == 0 and k not in ('RegionalMean_SDG', 'NationalRatio')]
for rep in range(REPEATS):
    M = np.column_stack([P[(rep, k)].values for k in base])
    y = T.y_rqual.values
    rng = np.random.RandomState(100 + rep); cs = T.iso3.unique(); fmap = {c: i % 5 for i, c in enumerate(rng.permutation(cs))}
    ff = T.iso3.map(fmap).values
    sp = np.zeros(len(T))
    for f in range(5):
        wgt, _ = nnls(M[ff != f], y[ff != f]); wgt = wgt / wgt.sum()
        sp[ff == f] = M[ff == f] @ wgt
    P[(rep, 'Stack_NNLS')] = pd.Series(np.clip(sp, 0, 1), index=T.index)
    if rep == 0:
        wgt, _ = nnls(M, y); print('stack weights (all):', dict(zip(base, np.round(wgt / wgt.sum(), 3))))

# propensity of being unmeasured among 2024 country-areas -> importance weights for observed test rows
A = F.area_long(d, [YEAR_TEST]); A = A[A.imp_area.notna() & (A.imp_area > 0)].copy()
A['miss'] = A.y_rqual.isna().astype(int)
pc = ['wat_basal_own', 'wat_pip_own', 'rprem_own', 'san_basal_own', 'income', 'log_pop_area', 'is_urban', 'region_fragileoecdany'] + [c for c in A.columns if c.startswith('sdg_')]
pm = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
pm.fit(A[pc], A.miss)
ps = pm.predict_proba(T[pc])[:, 1]
iw = ps / (1 - ps); iw = iw / iw.mean()
print('mean p(miss) among observed test rows', round(ps.mean(), 3), '| ESS of importance weights', round(iw.sum() ** 2 / (iw ** 2).sum(), 1), 'of', len(iw))

rows = []
for (rep, k), pr in P.items():
    q = pr.values
    sm_hat = T.imp_area.values * np.nanmin(np.column_stack([T[['rprem_own', 'rav_own']].values, q]), axis=1)
    e_q = q - T.y_rqual.values; e_sm = sm_hat - T.y_sm.values
    rows.append(dict(rep=rep, model=k, rmse_q=np.sqrt(np.mean(e_q ** 2)), r2_q=1 - np.sum(e_q ** 2) / np.sum((T.y_rqual - T.y_rqual.mean()) ** 2),
                     mae_sm=np.mean(np.abs(e_sm)), rmse_sm=np.sqrt(np.mean(e_sm ** 2)), r2_sm=1 - np.sum(e_sm ** 2) / np.sum((T.y_sm - T.y_sm.mean()) ** 2),
                     bias_sm=np.mean(e_sm), popw_mae_sm=np.average(np.abs(e_sm), weights=T['pop'].fillna(0).values + 1e-9),
                     iw_rmse_sm=np.sqrt(np.average(e_sm ** 2, weights=iw)), iw_bias_sm=np.average(e_sm, weights=iw)))
S = pd.DataFrame(rows).drop(columns='rep').groupby('model').agg(['mean', 'std'])
S.columns = [f'{a}_{b}' for a, b in S.columns]
S = S.sort_values('rmse_sm_mean')
pd.set_option('display.width', 250)
print(S[[c for c in S.columns if c.endswith('_mean')]].round(3).to_string())
tag = f"{SCEN}_train{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"
S.to_csv(f'results/E04_component_cv_{tag}.csv')
pd.DataFrame({f'{k}_rep{r}': v for (r, k), v in P.items()}).assign(iso3=T.iso3, area=T.area, y_rqual=T.y_rqual, y_sm=T.y_sm, iw=iw, pop=T['pop']).to_csv(f'results/E04_oof_{tag}.csv', index=False)
