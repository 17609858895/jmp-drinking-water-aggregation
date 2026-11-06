"""Release-integrity and scientific-invariant checks, without model training."""
from pathlib import Path
import argparse
import ast
import importlib
import importlib.metadata
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    for script in ROOT.rglob('*.py'):
        if '.venv' not in script.parts:
            ast.parse(script.read_text(encoding='utf-8'), filename=str(script))
    ref = ROOT / 'reference_results'
    series = pd.read_csv(ref / 'E08_series_bounds_m49l1.csv')
    assert np.all(series.jmp_rule >= series.mean_of_mins - 1e-8), 'Aggregation-order inequality violated'
    assert np.all(series.frechet_lower <= series.independence + 1e-8)
    assert np.all(series.independence <= series.mean_of_mins + 1e-8)
    np.testing.assert_allclose(series.gap_people_M, series.gap_pp / 100 * series.pop_bn * 1000, atol=1e-8)
    global_ = pd.read_csv(ref / 'E11_global_2024.csv').iloc[0]
    countries = pd.read_csv(ref / 'E11_country_estimates_2024.csv')
    published = countries.wat_sm_t.notna()
    np.testing.assert_allclose(countries.loc[published, 'sm_mean'], countries.loc[published, 'wat_sm_t'], atol=1e-8)
    estimate = np.average(countries.sm_mean, weights=countries.pop_t)
    np.testing.assert_allclose(estimate, global_.consistent_ml, atol=1e-8)
    assert global_.manski_lower <= global_.consistent_ml_lo90 <= global_.consistent_ml <= global_.consistent_ml_hi90 <= global_.manski_upper
    masking = pd.read_csv(ref / 'E07_masking_summary.csv', index_col=0)
    assert masking.loc['Regional mean, mean of mins', 'rmse'] < masking.loc['ML expected-min, mean of mins', 'rmse']
    checks = ['Python source syntax', 'aggregation and dependence bounds', 'population-equivalent units',
              'published-country preservation', 'country-to-world weighted total', 'retained negative masking result']
    if args.prepared:
        sys.path.insert(0, str(ROOT / 'scripts'))
        import features as F
        data = F.load()
        assert not data.duplicated(['iso3', 'year']).any()
        areas = F.area_long(data, [2024])
        assert not areas.duplicated(['iso3', 'area', 'year']).any()
        assert set(areas.area) == {'u', 'r'}
        observed = areas[areas.y_rqual.notna() & areas.imp_area.notna() & (areas.imp_area > 0)]
        assert len(observed) == 187 and observed.iso3.nunique() == 104
        checks.append('prepared real-data joins and 187 observed quality country-domains in 104 countries')
        year = data[(data.year == 2024) & data.pop_t.notna()]
        measured = year.wat_sm_t.notna()
        upper = year[['wat_imp_prem_t', 'wat_imp_av_t']].min(axis=1).fillna(year.imp_t).fillna(100)
        numerator = (year.loc[measured, 'pop_t'] * year.loc[measured, 'wat_sm_t']).sum()
        ceiling = (numerator + (year.loc[~measured, 'pop_t'] * upper[~measured]).sum()) / year.pop_t.sum()
        lower = numerator / year.pop_t.sum()
        np.testing.assert_allclose([ceiling, lower], [global_.manski_upper, global_.manski_lower], atol=1e-10)
        checks.append('imputation-free country-preserving ceiling/lower bound recomputed from raw-prepared data')
        generated = ROOT / 'results' / 'E08_series_bounds_m49l1.csv'
        if generated.exists():
            recomputed = pd.read_csv(generated)
            pd.testing.assert_frame_equal(recomputed, series, check_exact=False, rtol=1e-12, atol=1e-10)
            checks.append('all recomputed aggregation/dependence series cells match reference outputs')
    for module in ['scipy.optimize', 'sklearn', 'catboost', 'lightgbm', 'xgboost', 'pyarrow', 'openpyxl', 'joblib']:
        importlib.import_module(module)
    checks.append('analysis dependencies import successfully')
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost', 'matplotlib', 'pyarrow', 'openpyxl', 'joblib']}
    report = dict(status='PASS', checks=checks, python=sys.version.split()[0], packages=versions,
                  model_training_performed=False)
    output = ROOT / 'outputs'; output.mkdir(exist_ok=True)
    (output / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
