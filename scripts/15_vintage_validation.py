"""E15: retrospective validation across JMP editions.
Train on the 2021 edition (reference year 2020) or 2023 edition (reference year 2022),
then compare with subsequently reported country estimates in the 2025 edition at matched years.
The earlier-edition setting uses its non-quality inputs; a second setting uses updated non-quality inputs.
Global completion retains the same country rule and regional-residual mechanism as E11.
Usage: python scripts/15_vintage_validation.py <2021|2023>"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
from m49 import SUB2L1
import catboost as cb

V = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
T = {2023: 2022, 2021: 2020}[V]
NMC = 2000
rng = np.random.RandomState(15)
OUT = os.path.join('results', 'vintage')
old, new = F.load(vintage=V), F.load()
for d_ in (old, new): d_['l1'] = d_.m49.map(SUB2L1)


def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def ex(z): return 1 / (1 + np.exp(-z))
def model(seed): return cb.CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=3, verbose=0, random_seed=seed, thread_count=5)


def fit_with_oof(train, feats, ycol, unit_cols):
    """Final model + grouped 5-fold out-of-fold predictions for year T rows (the calibration set)."""
    sw = (1.0 / train.groupby(unit_cols).year.transform('count')).values
    m = model(0).fit(train[feats], train[ycol], sample_weight=sw)
    cs = train.iso3.unique(); fm = {c: i % 5 for i, c in enumerate(np.random.RandomState(1).permutation(cs))}; ff = train.iso3.map(fm).values
    t = (train.year == T).values; oof = np.full(len(train), np.nan)
    for f in range(5):
        mm = model(f).fit(train[feats][ff != f], train[ycol][ff != f], sample_weight=sw[ff != f])
        k = (ff == f) & t
        oof[k] = np.clip(mm.predict(train[feats][k]), 0, 1)
    cal = train[t].copy(); cal['qh'] = oof[t]; cal['res'] = lg(cal[ycol].values) - lg(cal.qh.values)
    return dict(m=m, feats=feats, cal=cal.reset_index(drop=True))


def qdraws(M, qh, excl=None, shared=None):
    """Quality draws: local OOF logit residuals (k = 40 nearest in predicted quality) plus a regional component (E11 scheme).
    shared=None draws the regional component independently; otherwise it is the region's common draw vector."""
    c = M['cal']; ok = (c.iso3.values != excl) if excl is not None else np.ones(len(c), bool)
    res, op = c.res.values[ok], c.qh.values[ok]
    near = np.argsort(np.abs(op - qh))[:40]
    reg = rng.choice(res, NMC) * 0.5 if shared is None else shared
    return ex(lg(qh) + rng.choice(res[near], NMC) * np.sqrt(0.75) + reg)


def mondrian(M, imp, b, y):
    """Leave-own-country-out predictive intervals on the calibration set, then split-conformal quantiles by tercile of P(quality binds)."""
    c = M['cal']; lo, hi, pb = (np.zeros(len(c)) for _ in range(3))
    for i in range(len(c)):
        Q = qdraws(M, c.qh.values[i], excl=c.iso3.values[i]); S = imp[i] * np.minimum(b[i], Q)
        lo[i], hi[i], pb[i] = np.percentile(S, 5), np.percentile(S, 95), (Q < b[i]).mean()
    score = np.maximum.reduce([lo - y, y - hi, np.zeros(len(c))]); ok = ~np.isnan(score); score, pb = score[ok], pb[ok]
    cut = np.quantile(pb, [1 / 3, 2 / 3]); grp = np.digitize(pb, cut)
    qv = []
    for g in range(3):
        s = np.sort(score[grp == g]); n = len(s); qv.append(s[min(n - 1, int(np.ceil((n + 1) * .9)) - 1)])
    cov_raw = ((lo <= y + 1e-9) & (y <= hi + 1e-9))[ok].mean()
    return dict(cut=cut, qv=np.array(qv), cov_raw_cal=cov_raw, n=len(c))


# ---------------- models trained on the earlier edition only ----------------
L = F.area_long(old, list(range(2000, T + 1)))
L = L[L.imp_area.notna() & (L.imp_area > 0)].reset_index(drop=True)
DROP = ['iso3', 'year', 'area', 'm49', 'sdg', 'pop', 'y_rqual', 'y_sm', 'W8', 'W9', 'W10', 'W10_nat', 'pu', 'imp_area', 'rqual_oth', 'rqual_nat']
BASE = [c for c in L.columns if c not in DROP]
obs = L[L.y_rqual.notna()].reset_index(drop=True)
MOD = {'S1': fit_with_oof(obs, BASE, 'y_rqual', ['iso3', 'area']),
       'S3': fit_with_oof(obs[obs.rqual_oth.notna()].reset_index(drop=True), BASE + ['rqual_oth'], 'y_rqual', ['iso3', 'area'])}
NV = [c for c in old.columns if c.endswith('_t') and c[:4] in ('wat_', 'san_', 'hyg_') and 'qual' not in c and 'sm' not in c and 'arc' not in c] + ['rprem_t', 'rav_t', 'imp_t']
NV = [c for c in NV if c not in ('wat_imp_prem_t', 'wat_imp_av_t')]


def natframe(d):
    D = d.copy(); D['income_n'] = D.region_income.map(F.INCOME); D['log_pop'] = np.log10(D.pop_t.clip(lower=0.1))
    for r in sorted(old.region_sdg.dropna().unique()): D['sdg_' + r[:12]] = (D.region_sdg == r).astype(int)
    return D


Dold, Dnew = natframe(old), natframe(new)
NF = NV + ['prop_u', 'income_n', 'log_pop'] + [c for c in Dold.columns if c.startswith('sdg_')]
MOD['N'] = fit_with_oof(Dold[Dold.rqual_t.notna() & (Dold.year <= T)].reset_index(drop=True), NF, 'rqual_t', ['iso3'])
for k, M in MOD.items():
    c = M['cal']
    if k == 'N': imp, b, y = c.imp_t.values, np.nanmin(np.column_stack([c.rprem_t, c.rav_t, np.ones(len(c))]), 1), c.wat_sm_t.values
    else: imp, b, y = c.imp_area.values, np.nanmin(np.column_stack([c.rprem_own, c.rav_own, np.ones(len(c))]), 1), c.y_sm.values
    M['conf'] = mondrian(M, imp, b, y)
    print(f'{k}: n_cal {len(c)}  OOF logit-residual SD {c.res.std():.3f}  raw 90% coverage on calibration {M["conf"]["cov_raw_cal"]:.3f}  '
          f'Mondrian widening by tercile {np.round(M["conf"]["qv"], 2)}')

# regional-mean (JMP-style) quality ratios from measured units of the earlier edition, year T
Lt_obs = L[(L.year == T) & L.y_rqual.notna()].copy(); Lt_obs['l1'] = Lt_obs.m49.map(SUB2L1); Lt_obs['w'] = Lt_obs['pop'] * Lt_obs.imp_area
Nt_obs = old[(old.year == T) & old.rqual_t.notna()].copy(); Nt_obs['w'] = Nt_obs.pop_t * Nt_obs.imp_t


def regmean(src, ycol, m49, area=None):
    s = src if area is None else src[src.area == area]
    for lvl, key in [('m49', m49), ('l1', SUB2L1.get(m49)), (None, None)]:
        g = s if lvl is None else s[s[lvl] == key]
        if len(g): return np.average(g[ycol], weights=g.w + 1e-9)


# ---------------- per-country prediction ----------------
def b_of(*v): return np.nanmin(list(v) + [1.0])


def area_pred(row, iso, shared=None):
    """Returns dict for one area row of area_long: draws of SM, plug-in, regional mean, upper, interval with Mondrian widening."""
    imp, b = row.imp_area, b_of(row.rprem_own, row.rav_own)
    if pd.notna(row.y_sm) and pd.notna(row.y_rqual):         # published in the earlier edition: fixed
        v = row.y_sm; return dict(route='published', S=np.full(NMC, v), plug=v, reg=v, up=v, lo=v, hi=v, q=row.y_rqual, qreg=row.y_rqual)
    key = 'S3' if pd.notna(row.rqual_oth) else 'S1'; M = MOD[key]
    qh = float(np.clip(M['m'].predict(row[M['feats']].to_frame().T.astype(float)), 0, 1)[0])
    Q = qdraws(M, qh, shared=shared); S = imp * np.minimum(b, Q)
    pb = (Q < b).mean(); w = M['conf']['qv'][np.digitize([pb], M['conf']['cut'])[0]]
    qr = regmean(Lt_obs, 'y_rqual', row.m49, row.area)
    return dict(route=key, S=S, plug=imp * min(b, qh), reg=imp * min(b, qr), up=imp * b, q=qh, qreg=qr,
                lo=max(0, np.percentile(S, 5) - w), hi=min(imp, np.percentile(S, 95) + w), lo_raw=np.percentile(S, 5), hi_raw=np.percentile(S, 95))


def country_pred(iso, src, Lsrc, Dsrc, shared=None, region_pool=None):
    r = src[(src.iso3 == iso) & (src.year == T)].iloc[0]
    rows = {a: Lsrc[(Lsrc.iso3 == iso) & (Lsrc.year == T) & (Lsrc.area == a)] for a in 'ur'}
    if all(len(x) and pd.notna(x.imp_area.iloc[0]) and x.imp_area.iloc[0] > 0 for x in rows.values()):
        pu = r.prop_u / 100; P = {a: area_pred(rows[a].iloc[0], iso, shared) for a in 'ur'}
        out = dict(route='area:' + '/'.join(P[a]['route'] for a in 'ur'), areas=P)
        for k in ['S', 'plug', 'reg', 'up', 'lo', 'hi']: out[k] = pu * P['u'][k] + (1 - pu) * P['r'][k]
        return out
    if pd.notna(r.imp_t) and r.imp_t > 0:
        M = MOD['N']; b = b_of(r.rprem_t, r.rav_t)
        xr = Dsrc[(Dsrc.iso3 == iso) & (Dsrc.year == T)].reindex(columns=NF).astype(float)
        qh = float(np.clip(M['m'].predict(xr), 0, 1)[0]); Q = qdraws(M, qh, shared=shared); S = r.imp_t * np.minimum(b, Q)
        pb = (Q < b).mean(); w = M['conf']['qv'][np.digitize([pb], M['conf']['cut'])[0]]
        qr = regmean(Nt_obs, 'rqual_t', r.m49)
        return dict(route='national', S=S, plug=r.imp_t * min(b, qh), reg=r.imp_t * min(b, qr), up=r.imp_t * b,
                    lo=max(0, np.percentile(S, 5) - w), hi=min(r.imp_t, np.percentile(S, 95) + w))
    # no water data in the source: population-weighted distribution of measured SM in the same SDG region
    g = region_pool[region_pool.region_sdg == r.region_sdg]
    S = rng.choice(g.wat_sm_t.values, NMC, p=(g.pop_t / g.pop_t.sum()).values); m = np.average(g.wat_sm_t, weights=g.pop_t)
    return dict(route='no water data', S=S, plug=m, reg=m, up=100.0, lo=np.percentile(S, 5), hi=np.percentile(S, 95))


ot, nt = old[old.year == T].set_index('iso3'), new[new.year == T].set_index('iso3')
TARGET = [i for i in ot.index if pd.isna(ot.loc[i, 'wat_sm_t']) and i in nt.index and pd.notna(nt.loc[i, 'wat_sm_t'])]
pool_old = old[(old.year == T) & old.wat_sm_t.notna()]
Lnew = F.area_long(new, [T]); Lnew = Lnew.reindex(columns=sorted(set(Lnew.columns) | set(BASE))).fillna({c: 0 for c in BASE if c.startswith('sdg_')})
Lnew.loc[Lnew.iso3.isin(TARGET), ['y_sm', 'y_rqual', 'rqual_oth', 'rqual_nat']] = np.nan   # 2025 quality of target countries never enters
rows, arows = [], []
for setting, src, Ls, Ds in [('as published then', old, L, Dold), ('2025 non-quality values', new, Lnew, Dnew)]:
    for iso in TARGET:
        p = country_pred(iso, src, Ls, Ds, region_pool=pool_old)
        a = nt.loc[iso]
        rows.append(dict(setting=setting, iso3=iso, name=a['name'], region_sdg=a.region_sdg, pop_M=a.pop_t / 1e3, route=p['route'], actual_2025=a.wat_sm_t,
                         upper=p['up'], regional_mean=p['reg'], ml_plugin=p['plug'], ml_expected_min=p['S'].mean(), ml_lo90=p['lo'], ml_hi90=p['hi']))
        for ar, P in p.get('areas', {}).items():
            if P['route'] == 'published': continue
            act = a[f'wat_sm_{ar}']; qa = a[f'rqual_{ar}']
            arows.append(dict(setting=setting, iso3=iso, area=ar, route=P['route'], actual_2025=act, actual_q=qa, q_ml=P['q'], q_reg=P['qreg'],
                              upper=P['up'], regional_mean=P['reg'], ml_plugin=P['plug'], ml_expected_min=P['S'].mean(),
                              ml_lo90=P['lo'], ml_hi90=P['hi'], ml_lo90_raw=P['lo_raw'], ml_hi90_raw=P['hi_raw']))
U = pd.DataFrame(rows); A = pd.DataFrame(arows)
U.to_csv(os.path.join(OUT, f'E15_countries_{V}.csv'), index=False); A.to_csv(os.path.join(OUT, f'E15_areas_{V}.csv'), index=False)

METH = ['upper', 'regional_mean', 'ml_plugin', 'ml_expected_min']


def metrics(D, wcol=None):
    out = []
    for m in METH:
        e = D[m] - D.actual_2025; w = D[wcol] if wcol else np.ones(len(D))
        r = dict(method=m, n=len(D), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(), pop_weighted_bias=np.average(e, weights=w),
                 r2=1 - (e ** 2).sum() / ((D.actual_2025 - D.actual_2025.mean()) ** 2).sum())
        if wcol: r['lacking_pred_M'] = (D[wcol] * (1 - D[m] / 100)).sum(); r['lacking_actual_M'] = (D[wcol] * (1 - D.actual_2025 / 100)).sum()
        if m == 'ml_expected_min':
            r['coverage90'] = ((D.ml_lo90 <= D.actual_2025 + 1e-9) & (D.actual_2025 <= D.ml_hi90 + 1e-9)).mean(); r['width90'] = (D.ml_hi90 - D.ml_lo90).mean()
            # published values carry one decimal and the national value is not always the exact urban/rural blend: 0.5-point tolerance
            r['coverage90_tol05'] = ((D.ml_lo90 <= D.actual_2025 + .5) & (D.actual_2025 <= D.ml_hi90 + .5)).mean()
        out.append(r)
    return pd.DataFrame(out)


summ = []
for setting in U.setting.unique():
    for lvl, D, w in [('country', U[U.setting == setting], 'pop_M'), ('country, excl. no-water-data', U[(U.setting == setting) & (U.route != 'no water data')], 'pop_M'),
                      ('area', A[(A.setting == setting) & A.actual_2025.notna()], None)]:
        s = metrics(D, w); s.insert(0, 'level', lvl); s.insert(0, 'setting', setting); summ.append(s)
    D = A[(A.setting == setting) & A.actual_q.notna()]
    for m in ['q_reg', 'q_ml']:
        e = (D[m] - D.actual_q) * 100
        summ.append(pd.DataFrame([dict(setting=setting, level='area quality ratio', method=m, n=len(D), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                                       r2=1 - (e ** 2).sum() / (((D.actual_q - D.actual_q.mean()) * 100) ** 2).sum())]))
    if setting == 'as published then':
        D = A[(A.setting == setting) & A.actual_2025.notna()]
        summ.append(pd.DataFrame([dict(setting=setting, level='area', method='ml_expected_min, raw interval', n=len(D),
                                       coverage90_tol05=((D.ml_lo90_raw <= D.actual_2025 + .5) & (D.actual_2025 <= D.ml_hi90_raw + .5)).mean(),
                                       coverage90=((D.ml_lo90_raw <= D.actual_2025 + 1e-9) & (D.actual_2025 <= D.ml_hi90_raw + 1e-9)).mean(), width90=(D.ml_hi90_raw - D.ml_lo90_raw).mean())]))
S = pd.concat(summ, ignore_index=True)
S.to_csv(os.path.join(OUT, f'E15_summary_{V}.csv'), index=False)
pd.set_option('display.width', 250)
print(f'\n=== JMP {V} file -> year {T}: {len(TARGET)} countries newly estimated in 2025, {nt.loc[TARGET].pop_t.sum() / 1e6:.2f} bn people')
print(U[U.setting == 'as published then'].sort_values('pop_M', ascending=False).round(1).drop(columns=['setting']).to_string(index=False))
print(S.round(3).to_string(index=False))


# ---------------- global level ----------------
def jmp_rule(d, y, imputed=True):
    """JMP aggregation: impute missing area ratios (M49 sub-region -> M49 region -> global), aggregate ratios, take the minimum."""
    g = d[d.year == y]; fr = []
    for a in 'ur':
        sh = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'iso3': g.iso3.values, 'm49': g.m49.values, 'l1': g.l1.values, 'area': a, 'pop': (g.pop_t * sh).values, 'imp': g[f'imp_{a}'].values})
        for k in ['rprem', 'rav', 'rqual']: f[k] = g[f'{k}_{a}'].values
        fr.append(f)
    f = pd.concat(fr, ignore_index=True); f = f[f['pop'].notna()].reset_index(drop=True)

    def imp_(col, w):
        for lvl in ['m49', 'l1', None]:
            miss = f[col].isna()
            if not miss.any(): break
            src = f[f[col].notna()]
            if lvl is None:
                for a in 'ur':
                    s = src[src.area == a]; f.loc[miss & (f.area == a), col] = np.average(s[col], weights=s[w] + 1e-9)
                break
            mu = src.groupby([lvl, 'area']).apply(lambda q: np.average(q[col], weights=q[w] + 1e-9))
            f.loc[miss, col] = [mu.get((x, a), np.nan) for x, a in zip(f.loc[miss, lvl], f.loc[miss, 'area'])]
    imp_('imp', 'pop'); f['ipop'] = f['pop'] * f.imp / 100
    if imputed:
        for k in ['rprem', 'rav', 'rqual']: imp_(k, 'ipop')
    P = f['pop'].sum(); v = 0
    for a in 'ur':
        q = f[f.area == a]
        v += q['pop'].sum() / P * np.average(q.imp, weights=q['pop']) * min(np.average(q[k][q[k].notna()], weights=q.ipop[q[k].notna()]) for k in ['rprem', 'rav', 'rqual'])
    return v


def hard_upper(d, y):
    g = d[(d.year == y) & d.pop_t.notna()]; m = g.wat_sm_t.notna()
    ub = g[['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(g.imp_t).fillna(100)
    return (np.sum(g.pop_t[m] * g.wat_sm_t[m]) + np.sum(g.pop_t[~m] * ub[~m])) / g.pop_t.sum(), g.pop_t[m].sum() / g.pop_t.sum()


# country-rule ML on the earlier edition for all its unmeasured countries; regional component shared within SDG regions or independent
yo = old[(old.year == T) & old.pop_t.notna()].reset_index(drop=True)
unm = yo.wat_sm_t.isna().values
glob = {}
for mode in ['independent', 'shared']:
    SH = {}
    if mode == 'shared':
        for rg in yo.region_sdg.dropna().unique(): SH[rg] = rng.choice(MOD['S1']['cal'].res.values, NMC) * 0.5
    SM = np.tile(yo.wat_sm_t.fillna(0).values, (NMC, 1)).astype(float)
    for i in np.where(unm)[0]:
        SM[:, i] = country_pred(yo.iso3[i], old, L, Dold, shared=SH.get(yo.region_sdg[i]), region_pool=pool_old)['S']
    glob[mode] = SM
P = yo.pop_t.values; Gi = glob['independent'] @ P / P.sum(); Gs = glob['shared'] @ P / P.sum()
up_old, ms_old = hard_upper(old, T); up_new, ms_new = hard_upper(new, T)
E11p = f'results/E11_global_{T}.csv'
e11 = pd.read_csv(E11p).iloc[0] if os.path.exists(E11p) else None
G = dict(vintage=V, year=T, measured_pop_share_then=ms_old, measured_pop_share_2025=ms_new,
         aggregate_rule_then=jmp_rule(old, T), aggregate_rule_2025=jmp_rule(new, T),
         aggregate_rule_available_then=jmp_rule(old, T, imputed=False), published_then={2023: 73.0, 2021: 74.0}[V],
         hard_upper_then=up_old, hard_upper_2025=up_new,
         country_rule_ml_then=Gi.mean(), country_rule_ml_then_lo90=np.percentile(Gi, 5), country_rule_ml_then_hi90=np.percentile(Gi, 95),
         country_rule_ml_then_shared_lo90=np.percentile(Gs, 5), country_rule_ml_then_shared_hi90=np.percentile(Gs, 95),
         country_rule_ml_2025=e11.consistent_ml if e11 is not None else np.nan)
# what each rule implied for the countries unmeasured then, against the 2025 file (published value where now measured, 2025 ML otherwise)
Pu = P[unm].sum(); meas_sum = np.sum(P[~unm] * yo.wat_sm_t.values[~unm])
G['implied_unmeasured_by_aggregate_rule'] = (G['aggregate_rule_then'] * P.sum() - meas_sum) / Pu
G['implied_unmeasured_by_published'] = (G['published_then'] * P.sum() - meas_sum) / Pu
G['implied_unmeasured_by_country_rule_ml'] = (glob['independent'][:, unm] @ P[unm]).mean() / Pu
G['implied_unmeasured_upper_bound'] = (up_old * P.sum() - meas_sum) / Pu
ce = pd.read_csv(f'results/E11_country_estimates_{T}.csv').set_index('iso3') if os.path.exists(f'results/E11_country_estimates_{T}.csv') else None
if ce is not None:
    iu = yo.iso3[unm].values; v25 = ce.reindex(iu)
    val = v25.wat_sm_t.fillna(v25.sm_mean)
    ok = val.notna().values
    G['unmeasured_then_value_2025'] = np.average(val.values[ok], weights=P[unm][ok])
    G['unmeasured_then_share_now_measured'] = P[unm][v25.wat_sm_t.notna().values].sum() / Pu
G = pd.Series(G); print(G.round(3).to_string())
G.to_frame().T.to_csv(os.path.join(OUT, f'E15_global_{V}.csv'), index=False)
