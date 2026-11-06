"""E11b: collect the per-year E11 outputs into the series tables used by the figures.
results/E11_global_series.csv : E11_global_<year>.csv rows plus the aggregate-then-minimum value, independence value and Frechet
                                lower bound of E08 (JMP 2025 imputation hierarchy, results/E08_series_bounds_m49l1.csv) for the world.
results/E11_joint_series.csv  : E11_joint_<year>.csv rows (world and low/middle-income countries).
Years: every year for which 11_global_country_rule.py has been run."""
import os, glob, re
import pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
years = sorted(int(re.findall(r'(\d{4})', f)[-1]) for f in glob.glob('results/E11_global_[0-9][0-9][0-9][0-9].csv'))
G = pd.concat([pd.read_csv(f'results/E11_global_{y}.csv') for y in years], ignore_index=True)
E8 = pd.read_csv('results/E08_series_bounds_m49l1.csv'); E8 = E8[E8.group == 'World'][['year', 'jmp_rule', 'independence', 'frechet_lower']]
G = G.merge(E8, on='year', how='left')
G.to_csv('results/E11_global_series.csv', index=False)
J = pd.concat([pd.read_csv(f'results/E11_joint_{y}.csv') for y in years], ignore_index=True)
J.to_csv('results/E11_joint_series.csv', index=False)
print('years', years)
print(G[['year', 'consistent_ml', 'consistent_ml_lo90', 'consistent_ml_hi90', 'manski_upper', 'jmp_rule']].round(2).to_string(index=False))
