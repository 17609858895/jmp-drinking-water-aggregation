"""E09: quick test of survival ML for 'time to near-universal service' (country-area level), against JMP-style linear
extrapolation. Landmark design: train on landmark 2006 (events 2007-2014), test on landmark 2014 (events 2015-2024).
Discrete-time hazard models in person-year format (logistic and gradient boosting). Metric: Harrell's C and AUC at horizon."""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

d = F.load()
TARGET = sys.argv[1] if len(sys.argv) > 1 else 'wat_basal'   # or 'wat_sm'
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 99.0


def build(landmark, horizon_end):
    rows = []
    for a in ['u', 'r']:
        g = d[['iso3', 'year', 'region_sdg', 'region_income', f'{TARGET}_{a}', f'wat_pip_{a}', f'san_basal_{a}', f'rprem_{a}']].copy()
        g.columns = ['iso3', 'year', 'sdg', 'income', 'y', 'pip', 'san', 'rprem']
        piv = g.pivot_table(index='iso3', columns='year', values='y')
        if landmark not in piv or landmark - 5 not in piv: continue
        lv, lv5 = piv[landmark], piv[landmark - 5]
        at_risk = lv.notna() & lv5.notna() & (lv < THR)
        for c in piv.index[at_risk]:
            fut = piv.loc[c, [yy for yy in range(landmark + 1, horizon_end + 1) if yy in piv.columns]]
            hit = fut[fut >= THR]
            t = (hit.index[0] - landmark) if len(hit) else (horizon_end - landmark)
            ev = int(len(hit) > 0)
            x = g[(g.iso3 == c) & (g.year == landmark)].iloc[0]
            slope = (lv[c] - lv5[c]) / 5
            rows.append(dict(iso3=c, area=a, T=t, E=ev, level=lv[c], gap=THR - lv[c], slope=slope, pip=x.pip, san=x.san, rprem=x.rprem,
                             income={'Low income': 0, 'Lower middle income': 1, 'Upper middle income': 2, 'High income': 3}.get(x.income, np.nan),
                             urban=int(a == 'u'), lin_time=(THR - lv[c]) / slope if slope > 1e-6 else 1e3))
    return pd.DataFrame(rows)


def person_period(D):
    pp = []
    for _, r in D.iterrows():
        for k in range(1, int(r['T']) + 1):
            z = r.to_dict(); z['k'] = k; z['event_now'] = int(r['E'] == 1 and k == r['T']); pp.append(z)
    return pd.DataFrame(pp)


def c_index(T, E, risk):
    n, conc, tot = len(T), 0.0, 0
    for i in range(n):
        if not E[i]: continue
        for j in range(n):
            if T[j] > T[i]:
                tot += 1; conc += 1.0 if risk[i] > risk[j] else (0.5 if risk[i] == risk[j] else 0)
    return conc / tot


tr, te = build(2006, 2014), build(2014, 2024)
print(TARGET, 'thr', THR, '| train at-risk', len(tr), 'events', tr.E.sum(), '| test at-risk', len(te), 'events', te.E.sum())
feats = ['level', 'gap', 'slope', 'pip', 'san', 'rprem', 'income', 'urban', 'k']
PP = person_period(tr)
models = {
    'Discrete-time logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
    'Discrete-time LightGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0),
}
H = 10
out = {}
for name, m in models.items():
    m.fit(PP[feats], PP.event_now)
    surv = np.ones(len(te))
    for k in range(1, H + 1):
        Z = te.assign(k=k)[feats]
        surv *= 1 - m.predict_proba(Z)[:, 1]
    out[name] = 1 - surv  # cumulative incidence by horizon H
out['Linear extrapolation (JMP-style)'] = -te.lin_time.values
out['Level only'] = te.level.values
y_h = ((te.E == 1) & (te['T'] <= H)).astype(int).values
res = []
for k, v in out.items():
    res.append(dict(model=k, c_index=c_index(te['T'].values, te.E.values, v), auc_10y=roc_auc_score(y_h, v)))
R = pd.DataFrame(res).sort_values('c_index', ascending=False)
print(R.round(3).to_string(index=False))
R.to_csv(f'results/E09_survival_{TARGET}_{int(THR)}.csv', index=False)
