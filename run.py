"""Explicit entry points for JMP analyses; importing this file performs no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'raw'
DATASETS = {
    '2025': ('JMP_2025_WLD.xlsx', '5b26a1fb97c5a8f91916ddb1a59664072455cb44223e88eec536a841c5bbeed7'),
    '2023': ('JMP_2023_WLD.xlsx', 'b19fe9fa3fbc51f513835f15ec2009eb129d0d4b8c4ed2081c344054adb79943'),
    '2021': ('JMP_2021_WLD.xlsx', 'ffca1c3b08b6f30fd774793759f0c4f553f8b489c81f5f855f8b5d370f6d8d05'),
}


def execute(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print('Running', Path(script).name, *args, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def check_data(editions):
    failed = []
    for edition in editions:
        name, expected = DATASETS[str(edition)]
        path = DATA / name
        if not path.exists():
            failed.append(f'{name}: missing; see DATA_SOURCES.md')
            continue
        actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if actual != expected:
            failed.append(f'{name}: SHA256 differs from the study input ({actual})')
        else:
            print(name, 'SHA256 verified')
    if failed:
        raise SystemExit('\n'.join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('check-data', help='Verify locally downloaded edition workbooks')
    p.add_argument('--editions', nargs='+', choices=DATASETS, default=['2025'])
    sub.add_parser('prepare', help='Read the 2025 workbook and write CSV/parquet caches')
    sub.add_parser('aggregation', help='Run algebraic checks, aggregation bounds and decomposition (no ML)')
    p = sub.add_parser('completion', help='Fit country-preserving completion for specified years')
    p.add_argument('--years', nargs='+', type=int, default=[2024])
    sub.add_parser('benchmarks', help='Run all four country-holdout benchmark settings (expensive)')
    sub.add_parser('postprocess', help='Expected-minimum/anchor/interval diagnostics after benchmarks')
    p = sub.add_parser('masking', help='Fit and evaluate population-weighted country masks (expensive)')
    p.add_argument('--replicates', type=int, default=200)
    sub.add_parser('measurement', help='Measurement priorities from the 2024 country draws')
    sub.add_parser('thresholds', help='Three service-threshold backtests')
    sub.add_parser('editions', help='Harmonise older workbooks and run retrospective edition tests')
    sub.add_parser('figure-tables', help='Generate original numerical figure input tables after analyses')
    p = sub.add_parser('plots', help='Create compact quantitative plots from generated or reference results')
    p.add_argument('--reference', action='store_true', help='Use the supplied published-analysis summary tables')
    p = sub.add_parser('smoke', help='Check release summaries, bounds, dependencies and source syntax; no fitting')
    p.add_argument('--prepared', action='store_true', help='Also check prepared real-data feature tables')
    args = parser.parse_args()
    (ROOT / 'results').mkdir(exist_ok=True)
    if args.command == 'check-data':
        check_data(args.editions)
    elif args.command == 'prepare':
        check_data(['2025']); execute('scripts/00_load_profile.py')
    elif args.command == 'aggregation':
        for script in ['01_identity_checks.py', '02_aggregation_bounds.py', '03_jmp_replication.py']:
            execute('scripts/' + script)
        for hierarchy in ['m49l1', 'sdg']:
            execute('scripts/08_series_bounds.py', hierarchy)
        execute('scripts/12_gap_decomposition.py')
    elif args.command == 'completion':
        for year in args.years:
            if not 2000 <= year <= 2024:
                raise SystemExit('Years must be between 2000 and 2024.')
            execute('scripts/11_global_country_rule.py', year)
        execute('scripts/11b_series.py')
    elif args.command == 'benchmarks':
        for years in ['2024', '2000-2024']:
            for setting in ['T1', 'T2']:
                execute('scripts/04_component_cv.py', years, setting)
    elif args.command == 'postprocess':
        for method in ['CatBoost', 'RegionalMean_M49', 'Stack_NNLS']:
            execute('scripts/05_jensen_imputation.py', 'results/E04_oof_T1_train2024-2024.csv', method)
        for window in ['2024-2024', '2000-2024']:
            execute('scripts/06_anchor_reconcile.py', 'T2_train' + window)
            execute('scripts/13_interval_calibration.py', 'T1_train' + window)
    elif args.command == 'masking':
        if args.replicates < 1:
            raise SystemExit('--replicates must be positive.')
        execute('scripts/07_masking_aggregate.py', args.replicates)
    elif args.command == 'measurement':
        execute('scripts/14_measurement_design.py')
    elif args.command == 'thresholds':
        for target, threshold in [('wat_basal', 90), ('wat_basal', 99), ('wat_sm', 95)]:
            execute('scripts/09_survival_backtest.py', target, threshold)
    elif args.command == 'editions':
        check_data(['2021', '2023'])
        execute('scripts/15_vintage_load.py')
        for edition in [2021, 2023]:
            execute('scripts/15_vintage_validation.py', edition)
        execute('scripts/16_edition_replication.py')
    elif args.command == 'figure-tables':
        execute('figures/scripts/figdata.py'); execute('figures/scripts/si_data.py')
    elif args.command == 'plots':
        execute('figures/plot_results.py', *(['--reference'] if args.reference else []))
    elif args.command == 'smoke':
        execute('checks/smoke_check.py', *(['--prepared'] if args.prepared else []))


if __name__ == '__main__':
    main()
