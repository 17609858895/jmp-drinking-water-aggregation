"""Input tables for Supplementary Figures S1-S12 (results only; the only fitting is the logistic propensity model of E07,
refitted here for its diagnostics). Run from the project root: python figures/scripts/si_data.py -> results/figdata/si/*.csv"""
import sys, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from m49 import SUB2L1
from scipy.optimize import nnls, brentq
SI = ROOT / 'results' / 'figdata' / 'si'; SI.mkdir(parents=True, exist_ok=True)
RES = ROOT / 'results'
d = F.load()
EL = ['rprem', 'rav', 'rqual']
OFFICIAL = {(2015, 'total'): 68, (2015, 'urban'): 83, (2015, 'rural'): 50, (2024, 'total'): 74, (2024, 'urban'): 83, (2024, 'rural'): 60}
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))

# ---------------------------------------------------------------- S1: information settings over time, feature table
rows = []
for y in range(2000, 2025):
    A = F.area_long(d, [y]); A = A[A['pop'].notna()]
    st = np.where(A.y_rqual.notna(), 'Area quality', np.where(A.rqual_nat.notna(), 'National only', np.where(A.rqual_oth.notna(), 'Other area', 'No quality')))
    for k in ['Area quality', 'National only', 'Other area', 'No quality']:
        rows.append(dict(year=y, setting=k, pop_bn=A['pop'][st == k].sum() / 1e6, units=int((st == k).sum())))
pd.DataFrame(rows).to_csv(SI / 's1_settings_by_year.csv', index=False)
cnt = d.groupby('year').apply(lambda g: pd.Series(dict(n_sm=g.wat_sm_t.notna().sum(), pop_share=g.pop_t[g.wat_sm_t.notna()].sum() / g.pop_t.sum())))
cnt.to_csv(SI / 's1_countries_with_sm.csv')
A = F.area_long(d, [2024]); A = A[A.imp_area.notna() & (A.imp_area > 0)]
groups = {'Water ladder': ['wat_basal_own', 'wat_lim_own', 'wat_unimp_own', 'wat_ns_own'], 'Piped / non-piped': ['wat_pip_own', 'wat_imp_npip_own'],
          'Premises': ['rprem_own'], 'Availability': ['rav_own'], 'Sanitation': ['san_basal_own', 'san_lim_own', 'san_unimp_own', 'san_ns_own', 'san_sew_own', 'san_sep_own'],
          'Hygiene': ['hyg_bas_own', 'hyg_lim_own', 'hyg_ns_own']}
mr = []
for gname, cols in groups.items():
    for lab, k in [('With quality data', A.y_rqual.notna()), ('Without quality data', A.y_rqual.isna())]:
        mr.append(dict(group=gname, units=lab, available=A.loc[k, cols].notna().mean().mean() * 100))
pd.DataFrame(mr).to_csv(SI / 's1_feature_availability.csv', index=False)
L = F.area_long(d, [2024])
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area']
feat = [c for c in L.columns if c not in DROP]
def fgroup(c):
    for g, cols in groups.items():
        if c.replace('_oth', '_own').replace('_nat', '_own') in cols: return g
    if c.startswith('sdg_'): return 'SDG region'
    if c.startswith('region_') or c == 'income': return 'Development group'
    if c in ('rqual_oth', 'rqual_nat'): return 'Quality anchor (settings 2-3 only)'
    return 'Area and population'
pd.DataFrame({'feature': feat, 'group': [fgroup(c) for c in feat],
              'scope': ['other area' if c.endswith('_oth') else 'national' if c.endswith('_nat') else 'own area' if c.endswith('_own') else '' for c in feat]}).to_csv(SI / 'table_s1_features.csv', index=False)

# ---------------------------------------------------------------- S2/S3: benchmarks and NNLS stacking weights
bench = []
for scen in ['T1', 'T2']:
    for win in ['2024-2024', '2000-2024']:
        S = pd.read_csv(RES / f'E04_component_cv_{scen}_train{win}.csv'); S['scenario'] = scen; S['window'] = win; bench.append(S)
pd.concat(bench).to_csv(SI / 's2_benchmark.csv', index=False)
W = []
for scen in ['T1', 'T2']:
    for win in ['2024-2024', '2000-2024']:
        O = pd.read_csv(RES / f'E04_oof_{scen}_train{win}.csv')
        base = [c[:-5] for c in O.columns if c.endswith('_rep0') and not c.startswith(('RegionalMean_SDG', 'NationalRatio', 'Stack'))]
        M = O[[b + '_rep0' for b in base]].values; w, _ = nnls(M, O.y_rqual.values); w = w / w.sum()
        W += [dict(scenario=scen, window=win, model=b, weight=v) for b, v in zip(base, w)]
pd.DataFrame(W).to_csv(SI / 's3_stack_weights.csv', index=False)

# ---------------------------------------------------------------- S4: national-identity reconciliation (CatBoost, T2, 2000-2024 training)
O = pd.read_csv(RES / 'E04_oof_T2_train2000-2024.csv')
A2 = F.area_long(d, [2024]); A2 = A2[A2.y_rqual.notna() & A2.imp_area.notna() & (A2.imp_area > 0) & A2.rqual_nat.notna()].reset_index(drop=True)
assert (A2.iso3.values == O.iso3.values).all()
b2 = np.nanmin(np.column_stack([A2.rprem_own, A2.rav_own, np.ones(len(A2))]), axis=1)
q = O[[f'CatBoost_rep{r}' for r in range(3)]].mean(1).values; qr = q.copy(); dl = {}
for c in A2.iso3.unique():
    k = np.where((A2.iso3 == c).values)[0]
    if len(k) != 2: continue
    pu = A2.pu.values[k[0]]; w = np.array([pu if A2.area.values[i] == 'u' else 1 - pu for i in k]) * A2.imp_area.values[k]
    f = lambda x: np.sum(w * ex(lg(q[k]) + x)) - A2.W10_nat.values[k[0]]
    try:
        x = brentq(f, -20, 20); qr[k] = ex(lg(q[k]) + x); dl[c] = x
    except ValueError: pass
pd.DataFrame({'iso3': A2.iso3, 'area': A2.area, 'reported': A2.y_sm, 'raw': A2.imp_area * np.minimum(b2, q), 'reconciled': A2.imp_area * np.minimum(b2, qr),
              'delta': A2.iso3.map(dl)}).to_csv(SI / 's4_reconciliation.csv', index=False)

# ---------------------------------------------------------------- S5: expected minimum diagnostics (CatBoost, T1, 2024 training)
O = pd.read_csv(RES / 'E04_oof_T1_train2024-2024.csv')
A1 = F.area_long(d, [2024]); A1 = A1[A1.y_rqual.notna() & A1.imp_area.notna() & (A1.imp_area > 0)].reset_index(drop=True)
b1 = np.nanmin(np.column_stack([A1.rprem_own, A1.rav_own, np.ones(len(A1))]), axis=1)
qh = O[[f'CatBoost_rep{r}' for r in range(3)]].mean(1).values; qt = O.y_rqual.values
res = lg(qt) - lg(qh); emin = np.zeros(len(qh))
for i in range(len(qh)):
    idx = np.where(O.iso3.values != O.iso3.values[i])[0]; near = idx[np.argsort(np.abs(qh[idx] - qh[i]))[:40]]
    emin[i] = A1.imp_area.values[i] * np.mean(np.minimum(b1[i], ex(lg(qh[i]) + res[near])))
pd.DataFrame({'iso3': A1.iso3, 'area': A1.area, 'q_hat': qh, 'q_true': qt, 'logit_resid': res, 'b': b1, 'binding': qt < b1, 'reported': A1.y_sm,
              'plugin': A1.imp_area * np.minimum(b1, qh), 'expected_min': emin}).to_csv(SI / 's5_expected_min.csv', index=False)

# ---------------------------------------------------------------- S6: propensity model and masks
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
A = F.area_long(d, [2024]); A = A[A.imp_area.notna() & (A.imp_area > 0) & A['pop'].notna()].reset_index(drop=True)
A['miss'] = A.y_rqual.isna().astype(int)
pc = ['wat_basal_own', 'wat_pip_own', 'rprem_own', 'san_basal_own', 'income', 'log_pop_area', 'is_urban', 'region_fragileoecdany'] + [c for c in A.columns if c.startswith('sdg_')]
pm = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
cvp = cross_val_predict(pm, A[pc], A.miss, groups=A.iso3, cv=GroupKFold(5), method='predict_proba')[:, 1]
pm.fit(A[pc], A.miss)
pd.DataFrame({'iso3': A.iso3, 'area': A.area, 'miss': A.miss, 'pop': A['pop'], 'ps': pm.predict_proba(A[pc])[:, 1], 'ps_cv': cvp}).to_csv(SI / 's6_propensity.csv', index=False)
print('propensity AUC in-sample %.3f, grouped CV %.3f' % (roc_auc_score(A.miss, pm.predict_proba(A[pc])[:, 1]), roc_auc_score(A.miss, cvp)))

# ---------------------------------------------------------------- S7/S8: imputation hierarchy and national-only handling
def frame(y, nat_fallback=False):
    g = d[d.year == y]; out = []
    for a in 'ur':
        sh = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'iso3': g.iso3.values, 'm49': g.m49.values, 'l1': g.m49.map(SUB2L1).values, 'sdg': g.region_sdg.values, 'area': a,
                          'pop': (g.pop_t * sh).values, 'imp': g[f'imp_{a}'].values})
        for k in EL:
            v = g[f'{k}_{a}'].values
            if nat_fallback:
                v = np.where(np.isnan(g[f'rqual_{a}'].values) & ~np.isnan(g['rqual_t'].values), g[f'{k}_t'].values, v)
            f[k] = v
        out.append(f)
    f = pd.concat(out, ignore_index=True)
    return f[f['pop'].notna()].reset_index(drop=True)
def impute(f, col, w, levels):
    for lvl in levels + [None]:
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
    out, P, ta, tm = {}, f['pop'].sum(), 0, 0
    for a, nm in [('u', 'urban'), ('r', 'rural')]:
        g = f[f.area == a]; w = g['pop'] * g.imp / 100
        agg = np.average(g.imp, weights=g['pop']) * min(np.average(g[k], weights=w) for k in EL)
        mm = np.average(g.imp * g[EL].min(axis=1), weights=g['pop'])
        out[nm] = (agg, mm); ta += g['pop'].sum() / P * agg; tm += g['pop'].sum() / P * mm
    out['total'] = (ta, tm); return out
HIER = {'M49 sub-region, region (JMP 2025)': ['m49', 'l1'], 'M49 sub-region, SDG region': ['m49', 'sdg'], 'SDG region only': ['sdg'], 'Global mean only': []}
rows = []
for y in range(2000, 2025):
    for hname, lv in HIER.items():
        f = frame(y); f = impute(f, 'imp', 'pop', lv); f['ipop'] = f['pop'] * f.imp / 100
        for k in EL: f = impute(f, k, 'ipop', lv)
        for lvl, (a_, m_) in rules(f).items():
            rows.append(dict(year=y, hierarchy=hname, handling='strict', level=lvl, jmp_rule=a_, country_rule=m_, official=OFFICIAL.get((y, lvl), np.nan)))
    f = frame(y, nat_fallback=True); lv = HIER['M49 sub-region, region (JMP 2025)']
    f = impute(f, 'imp', 'pop', lv); f['ipop'] = f['pop'] * f.imp / 100
    for k in EL: f = impute(f, k, 'ipop', lv)
    for lvl, (a_, m_) in rules(f).items():
        rows.append(dict(year=y, hierarchy='M49 sub-region, region (JMP 2025)', handling='national ratio', level=lvl, jmp_rule=a_, country_rule=m_, official=OFFICIAL.get((y, lvl), np.nan)))
H = pd.DataFrame(rows); H.to_csv(SI / 's7_s8_hierarchy_handling.csv', index=False)
print(H[(H.year == 2024) & (H.level == 'total')].round(2).to_string(index=False))

# ---------------------------------------------------------------- S9: country identification ranges (measured countries, 2024)
y = d[(d.year == 2024) & d.wat_sm_t.notna()].copy()
def joint(imp, rp, ra, q):
    el = [v for v in (rp, ra) if pd.notna(v)]; k = len(el) + 1
    return imp * q * (np.prod(el) if el else 1), imp * max(0, q + sum(el) - (k - 1))
rr = []
for _, r in y.iterrows():
    if pd.notna(r.rqual_u) and pd.notna(r.rqual_r) and pd.notna(r.imp_u) and pd.notna(r.imp_r):
        pu = r.prop_u / 100; ii = ll = 0
        for a, sh in [('u', pu), ('r', 1 - pu)]:
            i_, l_ = joint(r[f'imp_{a}'], r[f'rprem_{a}'], r[f'rav_{a}'], r[f'rqual_{a}']); ii += sh * i_; ll += sh * l_
    else:
        ii, ll = joint(r.imp_t, r.rprem_t, r.rav_t, r.rqual_t)
    rr.append(dict(iso3=r.iso3, name=r['name'], pop=r.pop_t, upper=r.wat_sm_t, independence=ii, lower=ll))
pd.DataFrame(rr).to_csv(SI / 's9_country_ranges_2024.csv', index=False)

# ---------------------------------------------------------------- S11: 2030 extrapolation
E8 = pd.read_csv(RES / 'E08_series_bounds_m49l1.csv'); E8 = E8[E8.group == 'World']
E11 = pd.read_csv(RES / 'E11_global_series.csv')
proj = []
for nm, yr, v in [('Aggregate, then minimum', E8.year, E8.jmp_rule), ('Minimum, then aggregate (regional)', E8.year, E8.mean_of_mins),
                  ('Minimum, then aggregate (ML)', E11.year, E11.consistent_ml)]:
    yr, v = np.asarray(yr), np.asarray(v); k = yr >= 2015
    s, c = np.polyfit(yr[k], v[k], 1)
    proj.append(dict(series=nm, slope=s, v2024=v[yr == 2024][0], v2030=v[yr == 2024][0] + 6 * s, year_99=2024 + (99 - v[yr == 2024][0]) / s))
pd.DataFrame(proj).to_csv(SI / 's11_projection.csv', index=False); print(pd.DataFrame(proj).round(2))

# ---------------------------------------------------------------- S12: countries without 2024 rows
w = d
miss = sorted(set(w.iso3) - set(w[w.year == 2024].iso3))
g = w[w.iso3.isin(miss)].sort_values('year').groupby('iso3').agg(name=('name', 'last'), last_year=('year', 'max'), pop_last=('pop_t', 'last'),
                                                                  sm_last=('wat_sm_t', 'last'), basic_last=('wat_basal_t', 'last'))
g.to_csv(SI / 's12_missing_rows.csv'); print(g.round(1).to_string())
