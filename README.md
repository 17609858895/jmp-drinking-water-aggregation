# Drinking-water aggregation and country-preserving completion

Core Python analysis code for a study of global safely managed drinking-water coverage using the WHO/UNICEF JMP 2025 household world file, with retrospective checks using the 2021 and 2023 editions.

The study compares **aggregate-then-minimum** with **minimum-then-aggregate**, derives a feasible ceiling while preserving published country estimates, and evaluates missing-quality completion and measurement priorities. The code retains the benchmark results in which better country-domain prediction did not produce the lowest global masking error: regional-mean completion outperformed the ML expected-minimum estimator in that test.

This is an analysis and numerical-figure release. It includes the analysis scripts, original numerical figure-input generators, compact quantitative plotting code, and small reference-result tables. It excludes raw third-party workbooks, trained models, manuscript files, decorative illustrations, geographic artwork and submission-layout assembly. The compact plots are newly arranged views of the numerical results; they do not reproduce the submission's figure layout pixel for pixel.

## Environment

Python **3.12.3** was used for the release checks. A dedicated environment is recommended:

```bash
python -m venv .venv
# Activate .venv using your operating system's standard command.
python -m pip install -r requirements.txt
python run.py --help
```

`requirements.txt` records the installed package versions used for the release smoke checks, including scikit-learn 1.5.1, CatBoost 1.2.10, LightGBM 4.6.0 and XGBoost 3.2.0. A complete environment lock from the original model runs was not retained in the available project files. Full training was **not** repeated during packaging, and exact agreement of retrained ML results with the supplied reference tables has not been established. The fixed random seeds and scientific algorithms are retained in the analysis scripts.

## Quick checks and quantitative plots

No raw data or model fitting is needed for these commands:

```bash
python run.py smoke
python run.py plots --reference
```

The smoke checks verify syntax and installed dependencies, the aggregation/dependence inequalities, population units, preservation of published country values, the population-weighted global completion, and the retained negative masking result. Six PNG/SVG plots are written to `outputs/reference_plots/`. Their error bars and shaded regions retain the quantities in the reference tables: the country-domain benchmark uses across-repetition standard deviations; the completion intervals use fifth/ninety-fifth model-draw percentiles. The random-priority band uses the precomputed random-priority quantiles. These quantities are not interchangeable.

The release also checked real-workbook preparation, feature-table joins and the inexpensive aggregation/decomposition calculations against existing results. It did not retrain the full ML benchmarks, 200-mask experiment or edition models. `reference_results/` contains previously computed outputs, not newly rerun ML results.

## Recompute from the official workbooks

Download and place the files exactly as described in [DATA_SOURCES.md](DATA_SOURCES.md). Commands run from this repository's root. Generated caches, model draws and plots are ignored by Git.

```bash
python run.py check-data --editions 2025 2023 2021
python run.py prepare
python run.py smoke --prepared
python run.py aggregation
python run.py completion --years 2000 2005 2010 2015 2020 2022 2024
python run.py benchmarks
python run.py postprocess
python run.py masking --replicates 200
python run.py measurement
python run.py thresholds
python run.py editions
python run.py figure-tables
python run.py plots
```

`completion`, `benchmarks`, `masking` and `editions` fit models and can be computationally substantial. They run only when explicitly selected; there is no implicit full-training command. Start with `completion --years 2024` if only the latest year is required. To build the seven-year series and retrospective comparisons, run all years listed above. `figure-tables` requires the preceding analyses, including all four benchmark settings and threshold backtests. It refits the small missingness-propensity model used for diagnostics. `editions` requires preparation of the 2025 file and the matched-year completion outputs (2020 and 2022) for all comparison columns.

## Code map

| Scripts | Role |
|---|---|
| `00_load_profile.py`, `features.py`, `m49.py` | Workbook preparation; country-area predictors; regional lookup |
| `01_identity_checks.py`–`03_jmp_replication.py` | Reported-value identities and alternative aggregation calculations |
| `04_component_cv.py` | Country-grouped holdouts for T1 (whole-country quality missing) and T2 (national quality anchor available), using one-year and pooled training |
| `05_jensen_imputation.py`, `06_anchor_reconcile.py` | Expected-minimum and national-anchor diagnostics |
| `07_masking_aggregate.py` | Population-weighted country-mask evaluation of global estimators |
| `08_series_bounds.py`, `12_gap_decomposition.py` | Aggregation/dependence series and country accounting contributions |
| `09_survival_backtest.py` | Service-threshold models and linear-extrapolation comparison, including weak/negative results |
| `11_global_country_rule.py`, `11b_series.py` | Published-country-preserving completion, available-criterion dependence bounds and time series |
| `13_interval_calibration.py`, `14_measurement_design.py` | Empirical interval diagnostics and model-based measurement priorities |
| `15_vintage_load.py`, `15_vintage_validation.py`, `16_edition_replication.py` | Older-edition harmonisation, retrospective matched-year tests and edition replication |
| `figures/scripts/figdata.py`, `si_data.py` | Original numerical figure-input tables |
| `figures/plot_results.py` | Portable quantitative plots from generated or reference summaries |

Numbering follows the study analysis sequence. The exploratory E10 completion route is omitted because the final country-preserving estimator uses E11. Some original CSV column names retain development labels (for example `manski_upper` for the country-preserving ceiling, `consistent_ml` for country-rule completion, and `MINIMA` in diagnostic labels). They do not identify a separately validated new ML algorithm.

## Interpretation and reference outputs

For the matched 2024 calculation, the original reference outputs give 73.59% under aggregate-then-minimum and 67.79% under minimum-then-aggregate. Their population-equivalent accounting difference is approximately 470.5 million. The country-preserving ceiling is 70.79%; the model-based country-rule completion is 69.63%, with a 90% model interval of 68.44–70.28%.

The aggregation difference, available-criterion household-dependence range and completion-model interval answer different questions. They are not additive components of one uncertainty distribution. The completion interval conditions on the reported non-quality inputs and the implemented residual model; it does not include every source of survey or JMP revision uncertainty. Measurement priorities describe variance reduction within that model. Edition tests use later-published estimates at matched years and do not establish prospective forecast performance. The accounting differences do not identify particular households or imply a measured health burden.

`reference_results/README.md` documents the supplied summary tables. Raw input checksums are provided separately; no raw workbook is redistributed. No code license is added in this release.
