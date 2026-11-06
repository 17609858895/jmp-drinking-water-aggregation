"""E16: replicate the published global safely managed (SM) values of the 2021, 2023 and 2025 JMP editions from each edition's own
world file, under two versions of the aggregate-then-minimum rule:
  imputed     - missing area ratios imputed by M49 sub-region -> M49 region -> global means (JMP 2025 Annex 1);
  available   - each criterion averaged over the countries that report it (no imputation).
Published values (secondary summaries of the reports, checked 2026-10-26; full reports not re-read):
  2021 report: 2016 70%, 2020 74% (urban 86%, rural 60%); 2023 report: 2015 69%, 2022 73%; 2025 report: 2015 68%, 2024 74% (urban 83%, rural 60%).
Also the imputation-free upper bound of each edition (quality = 100% where SM is missing)."""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts'); import features as F
from m49 import SUB2L1
EL = ['rprem', 'rav', 'rqual']
PUB = {(2021, 2016): (70, np.nan, np.nan), (2021, 2020): (74, 86, 60), (2023, 2015): (69, np.nan, np.nan), (2023, 2022): (73, np.nan, np.nan),
       (2025, 2015): (68, np.nan, np.nan), (2025, 2024): (74, 83, 60)}


def frame(d, y):
    g = d[d.year == y]; fr = []
    for a in 'ur':
        sh = g.prop_u / 100 if a == 'u' else 1 - g.prop_u / 100
        f = pd.DataFrame({'m49': g.m49.values, 'l1': g.m49.map(SUB2L1).values, 'area': a, 'pop': (g.pop_t * sh).values, 'imp': g[f'imp_{a}'].values})
        for k in EL: f[k] = g[f'{k}_{a}'].values
        fr.append(f)
    f = pd.concat(fr, ignore_index=True); return f[f['pop'].notna()].reset_index(drop=True)


def impute(f, col, w):
    for lvl in ['m49', 'l1', None]:
        miss = f[col].isna()
        if not miss.any(): break
        src = f[f[col].notna()]
        if lvl is None:
            for a in 'ur':
                s = src[src.area == a]; f.loc[miss & (f.area == a), col] = np.average(s[col], weights=s[w] + 1e-9)
            break
        mu = {k: np.average(q[col], weights=q[w] + 1e-9) for k, q in src.groupby([lvl, 'area'])}
        f.loc[miss, col] = [mu.get((x, a), np.nan) for x, a in zip(f.loc[miss, lvl], f.loc[miss, 'area'])]


def rule(d, y, imputed):
    f = frame(d, y); impute(f, 'imp', 'pop'); f['ipop'] = f['pop'] * f.imp / 100
    if imputed:
        for k in EL: impute(f, k, 'ipop')
    out, P = {}, f['pop'].sum()
    for a in 'ur':
        s = f[f.area == a]
        m = [np.average(s[k][s[k].notna()], weights=s.ipop[s[k].notna()]) for k in EL]
        out[a] = np.average(s.imp, weights=s['pop']) * min(m); out['binding_' + a] = EL[int(np.argmin(m))]
    out['t'] = sum(f[f.area == a]['pop'].sum() / P * out[a] for a in 'ur')
    return out


def upper(d, y):
    g = d[(d.year == y) & d.pop_t.notna()]; m = g.wat_sm_t.notna()
    ub = g[['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(g.imp_t).fillna(100)
    return (np.sum(g.pop_t[m] * g.wat_sm_t[m]) + np.sum(g.pop_t[~m] * ub[~m])) / g.pop_t.sum(), g.pop_t[m].sum() / g.pop_t.sum()


rows = []
for ed in [2021, 2023, 2025]:
    d = F.load(vintage=None if ed == 2025 else ed)
    for y in range(2015, d.year.max() + 1):
        a, b = rule(d, y, False), rule(d, y, True); ub, ms = upper(d, y); p = PUB.get((ed, y), (np.nan,) * 3)
        rows.append(dict(edition=ed, year=y, measured_pop_share=ms, published=p[0], published_u=p[1], published_r=p[2],
                         available_t=a['t'], available_u=a['u'], available_r=a['r'], imputed_t=b['t'], imputed_u=b['u'], imputed_r=b['r'],
                         binding_u=a['binding_u'], binding_r=a['binding_r'], hard_upper=ub))
R = pd.DataFrame(rows)
R.to_csv(os.path.join('results', 'vintage', 'E16_edition_replication.csv'), index=False)
pd.set_option('display.width', 250)
print(R.round(2).to_string(index=False))

# ---- revisions between editions for countries measured in both (same year): how much do published country values move?
E = {ed: F.load(vintage=None if ed == 2025 else ed) for ed in [2021, 2023, 2025]}
rev = []
for a, b, y in [(2021, 2023, 2020), (2023, 2025, 2022), (2021, 2025, 2020)]:
    A_ = E[a][E[a].year == y].set_index('iso3'); B_ = E[b][E[b].year == y].set_index('iso3')
    both = A_.index[A_.wat_sm_t.notna()].intersection(B_.index[B_.wat_sm_t.notna()])
    for c in ['wat_sm_t', 'wat_imp_prem_t', 'wat_imp_av_t', 'wat_imp_qual_t', 'imp_t']:
        x = (B_.loc[both, c] - A_.loc[both, c]).dropna()
        rev.append(dict(old=a, new=b, year=y, var=c, n=len(x), mean=x.mean(), sd=x.std(), mad=x.abs().median(), q90_abs=x.abs().quantile(.9),
                        share_abs_gt5=(x.abs() > 5).mean()))
    # non-quality bound b*imp for countries newly measured later: how far did the published upper bound move?
    new_ = A_.index[A_.wat_sm_t.isna()].intersection(B_.index[B_.wat_sm_t.notna()])
    ub = lambda D, i: D.loc[i, ['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(D.loc[i, 'imp_t'])
    x = (ub(B_, new_) - ub(A_, new_)).dropna()
    rev.append(dict(old=a, new=b, year=y, var='upper bound, newly measured', n=len(x), mean=x.mean(), sd=x.std(), mad=x.abs().median(), q90_abs=x.abs().quantile(.9),
                    share_abs_gt5=(x.abs() > 5).mean()))
RV = pd.DataFrame(rev); RV.to_csv(os.path.join('results', 'vintage', 'E16_revisions.csv'), index=False)
print(RV.round(2).to_string(index=False))
