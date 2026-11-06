"""Compact quantitative views of the released tables, without model fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, ROSE, PURPLE, GOLD = '#99BADF', '#99CDCE', '#DF9E9B', '#999ACD', '#F6D58B'


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(output / (name + '.svg'), facecolor='white')
    plt.close(fig)
    print(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    source = ROOT / ('reference_results' if args.reference else 'results')
    output = ROOT / 'outputs' / ('reference_plots' if args.reference else 'plots')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    series = pd.read_csv(source / 'E08_series_bounds_m49l1.csv')
    world = series[series.group == 'World']
    completion = pd.read_csv(source / 'E11_global_series.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(world.year, world.jmp_rule, color=ROSE, label='Aggregate, then minimum')
    ax.plot(world.year, world.mean_of_mins, color=BLUE, label='Minimum, then aggregate')
    ax.plot(completion.year, completion.manski_upper, color=PURPLE, linestyle='--', label='Country-preserving ceiling')
    ax.plot(completion.year, completion.consistent_ml, color=TEAL, label='Country rule + ML')
    ax.fill_between(completion.year, completion.consistent_ml_lo90, completion.consistent_ml_hi90, color=TEAL, alpha=.25)
    ax.set(xlabel='Year', ylabel='Global coverage (%)')
    ax.legend(frameon=False, fontsize=8)
    save(fig, output, 'aggregation_and_completion')

    country = pd.read_csv(source / 'E11_country_estimates_2024.csv')
    country = country[country.wat_sm_t.isna()].nlargest(15, 'pop_t').sort_values('sm_mean')
    fig, ax = plt.subplots(figsize=(7, 5))
    # Means can lie outside central quantile intervals for skewed mixtures.
    # Draw endpoints directly instead of converting them into distances from the mean.
    positions = np.arange(len(country))
    ax.hlines(positions, country.sm_lo, country.sm_hi, color=BLUE, linewidth=1.5)
    ax.scatter(country.sm_mean, positions, color=TEAL, s=20, zorder=3)
    ax.set_yticks(positions, country.name)
    ax.set(xlabel='Country coverage (%)', xlim=(0, 100))
    save(fig, output, 'completion_of_unreported_countries')

    bench = pd.read_csv(source / 'E04_component_cv_T1_train2000-2024.csv').sort_values('rmse_sm_mean')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(bench.model, bench.rmse_sm_mean, xerr=bench.rmse_sm_std, color=BLUE, error_kw={'elinewidth': .8, 'capsize': 2})
    ax.invert_yaxis()
    ax.set(xlabel='Country-domain RMSE (percentage points)')
    save(fig, output, 'country_holdout_benchmarks')

    masks = pd.read_csv(source / 'E07_masking_summary.csv', index_col=0)
    labels = ['Aggregate, then minimum', 'Regional mean, country rule', 'ML plug-in, country rule',
              'ML expected minimum, country rule', 'Upper bound']
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels, masks.rmse, color=[ROSE, BLUE, TEAL, TEAL, PURPLE])
    ax.invert_yaxis(); ax.set(xlabel='Global masking RMSE (percentage points)')
    save(fig, output, 'population_masking_benchmarks')

    fig, ax = plt.subplots(figsize=(7, 4))
    for edition, color in [(2021, BLUE), (2023, TEAL)]:
        summary = pd.read_csv(source / 'vintage' / f'E15_summary_{edition}.csv')
        summary = summary[(summary.setting == 'as published then') & (summary.level == 'country')]
        ax.plot(summary.method, summary.rmse, marker='o', color=color, label=f'{edition} edition')
    ax.set(xlabel='Completion method', ylabel='Country RMSE (percentage points)')
    ax.tick_params(axis='x', labelrotation=20); ax.legend(frameon=False)
    save(fig, output, 'retrospective_edition_benchmarks')

    curves = pd.read_csv(source / 'E14_measurement_curves.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curves.n_measured, curves.greedy, color=TEAL, label='Greedy')
    ax.plot(curves.n_measured, curves.largest_population_first, color=BLUE, label='Population first')
    ax.plot(curves.n_measured, curves.random_median, color=PURPLE, label='Random median')
    ax.fill_between(curves.n_measured, curves.random_lo, curves.random_hi, color=PURPLE, alpha=.15)
    ax.set(xlabel='Additional countries measured', ylabel='Global 90% model interval width (points)')
    ax.legend(frameon=False)
    save(fig, output, 'measurement_priorities')


if __name__ == '__main__':
    main()
