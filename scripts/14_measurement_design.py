"""E14: which countries' water-quality data would most reduce the uncertainty of the 2024 global estimate.
Measuring a country is approximated by fixing its SM draws at their mean (the measured value is assumed exact; the
uncertainty that remains is the joint uncertainty of the others, including the shared regional component).
Greedy selection versus largest-population-first and random orders (200). Also per-country identification widths."""
import os
import numpy as np, pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
Z = np.load('results/E11_country_draws_2024.npz', allow_pickle=True)
iso, P, SM, IND, LOW = Z['iso3'], Z['pop'], Z['sm'].astype(float), Z['ind'].astype(float), Z['low'].astype(float)
w = P / P.sum()
C = pd.read_csv('results/E11_country_estimates_2024.csv').set_index('iso3').loc[iso]
unm = np.where(C.kind.values != 'measured')[0]
D = (SM[:, unm] - SM[:, unm].mean(0)) * w[unm]          # centred, weighted contributions


def width(cols):
    """90% interval width (points) of the global estimate when only `cols` (indices into unm) stay uncertain."""
    if len(cols) == 0: return 0.0
    g = D[:, cols].sum(1); return np.percentile(g, 95) - np.percentile(g, 5)


full = width(list(range(len(unm))))
order_g, left = [], list(range(len(unm)))
curve_g = [full]
while left:
    best = min(left, key=lambda j: width([k for k in left if k != j]))
    order_g.append(best); left.remove(best); curve_g.append(width(left))
order_p = list(np.argsort(-P[unm]))
curve_p = [width(order_p[k:]) for k in range(len(unm) + 1)]
rng = np.random.RandomState(3); R = []
for _ in range(200):
    o = list(rng.permutation(len(unm))); R.append([width(o[k:]) for k in range(len(unm) + 1)])
R = np.array(R)
n = np.arange(len(unm) + 1)
out = pd.DataFrame({'n_measured': n, 'greedy': curve_g, 'largest_population_first': curve_p, 'random_median': np.median(R, 0),
                    'random_lo': np.percentile(R, 2.5, 0), 'random_hi': np.percentile(R, 97.5, 0)})
out.to_csv('results/E14_measurement_curves.csv', index=False)
pri = pd.DataFrame({'rank': np.arange(1, len(unm) + 1), 'iso3': iso[unm][order_g], 'name': C.name.values[unm][order_g], 'pop_M': P[unm][order_g] / 1e3,
                    'width_after': curve_g[1:]})
pri.to_csv('results/E14_greedy_order.csv', index=False)
print('global 90%% width with all unmeasured uncertain: %.2f points' % full)
print(out.iloc[[0, 1, 2, 3, 5, 10, 20]].round(3).to_string(index=False))
print(pri.head(10).round(2).to_string(index=False))
# identification width per country (Frechet range under the country rule)
W = pd.DataFrame({'iso3': iso, 'sm_mean': SM.mean(0), 'independence_mean': IND.mean(0), 'lower_mean': LOW.mean(0), 'kind': C.kind.values, 'pop': P})
W['width'] = W.sm_mean - W.lower_mean
W.to_csv('results/E14_identification_width_2024.csv', index=False)
print('population-weighted identification width: %.1f points' % np.average(W.width, weights=W['pop']))
