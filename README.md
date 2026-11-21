# Drinking-water aggregation and country-preserving completion

Core Python analysis code for a study of global safely managed drinking-water coverage using the WHO/UNICEF JMP 2025 household world file, with retrospective checks using the 2021 and 2023 editions.

The study compares **aggregate-then-minimum** with **minimum-then-aggregate**, derives a feasible ceiling while preserving published country estimates, and evaluates missing-quality completion and measurement priorities. It applies the same ceiling to regions and country groups and, as a control, to safely managed sanitation, an indicator that adds its components. The code retains the benchmark results in which better country-domain prediction did not produce the lowest global masking error: regional-mean completion outperformed the ML expected-minimum estimator in that test.

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

The smoke checks verify syntax and installed dependencies, the aggregation/dependence inequalities, population units, preservation of published country values, the population-weighted global completion, the retained negative masking result, and the group ceilings with the sanitation control. Seven PNG/SVG plots are written to `outputs/reference_plots/`. Their error bars and shaded regions retain the quantities in the reference tables: the country-domain benchmark uses across-repetition standard deviations; the completion intervals use fifth/ninety-fifth model-draw percentiles. The random-priority band uses the precomputed random-priority quantiles. These quantities are not interchangeable.

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
| `20_group_ceilings.py`, `21a_parse_annex9.py`, `21_sanitation_control.py` | Country-preserving ceilings of regions and country groups; the same test for safely managed sanitation |
| `figures/scripts/figdata.py`, `si_data.py` | Original numerical figure-input tables |
| `figures/plot_results.py` | Portable quantitative plots from generated or reference summaries |

Numbering follows the study analysis sequence. The exploratory E10 completion route is omitted because the final country-preserving estimator uses E11. Some original CSV column names retain development labels (for example `manski_upper` for the country-preserving ceiling, `consistent_ml` for country-rule completion, and `MINIMA` in diagnostic labels). They do not identify a separately validated new ML algorithm.

## Checks used in the revised manuscript

The scripts below implement the updated masking, edition, reconstruction and sensitivity analyses. The original commands above remain available for the earlier diagnostics and portable plots. Run these commands from the repository root after preparing the JMP files. The vintage analyses require data harmonisation and the 2025-edition completion outputs for 2020 and 2022. The completion sensitivity requires the aggregation series and 2024 completion output.

```bash
python scripts/07b_masking_final_config.py 200
python scripts/15_vintage_prospective.py 2021
python scripts/15_vintage_prospective.py 2023
python scripts/17_completion_sensitivity.py
```

For regional reconstruction, copy `reference_results/jmp2025_annex8_sm.csv` to `results/`. Alternatively, generate the same comparison inputs by running `python scripts/18a_parse_annex8.py /path/to/JMP_2025_report.pdf` with PyMuPDF installed and the [2025 JMP report](https://www.who.int/publications/i/item/9789240115149). Then run:

```bash
python scripts/18_regional_reconstruction.py
python scripts/19_reporting_checks.py
```

The reporting check uses the prepared 2025, 2021 and 2023 water tables. It calculates national-component and relaxed ceilings, the expanded 2024 denominator, and the three household constructions on a matched subset with all components. Its results and the updated numerical summaries are included in `reference_results/`. The global dependence scenarios use approximations for places with incomplete components. The matched-domain bounds use common marginal inputs and weights. The release verifies the new arithmetic checks and regional comparison directly, and supplies the previously computed model outputs for comparison after retraining.

### Group ceilings and the sanitation control

`20_group_ceilings.py` repeats the ceiling calculation for each SDG region, development group and income group: national estimates of safely managed drinking water stay fixed, every other country takes its highest feasible coverage, and the result is compared with the Annex 8 value. It reads the prepared 2025 water table and the output of `18_regional_reconstruction.py`. `21_sanitation_control.py` applies the same test to safely managed sanitation, which adds its components; a country without an estimate is bounded between zero and its at-least-basic coverage. Copy `reference_results/jmp2025_annex9_sm.csv` to `results/`, or regenerate it with `python scripts/21a_parse_annex9.py /path/to/JMP_2025_report.pdf`, then run:

```bash
python scripts/20_group_ceilings.py
python scripts/21_sanitation_control.py
```

Both scripts are arithmetic and fit no model. Group membership follows the classification columns of the 2025 world file. The release recomputed both tables and the regional reconstruction from the official 2025 workbook and re-read the Annex 9 values from the report; all match the supplied reference tables. `python run.py plots` adds a comparison of the 2024 totals with their ceilings once both tables exist.

## Interpretation and reference outputs

For the matched 2024 calculation, the original reference outputs give 73.59% under aggregate-then-minimum and 67.79% under minimum-then-aggregate. Their population-equivalent accounting difference is approximately 470.5 million. The country-preserving ceiling is 70.79%; the model-based country-rule completion is 69.63%, with a 90% model interval of 68.44–70.28%. In Central and Southern Asia, national estimates cover 99.98% of the population and average 70.30%, and the regional value is 77%. Of the twelve group totals that the aggregate-first rule reproduces within one point, nine lie above their permissive ceilings in 2024. No regional or global total of safely managed sanitation lies above its ceiling in 2015 or 2024; the totals for Europe and Northern America and for high-income countries lie below their floors.

The aggregation difference, household dependence scenario spread and completion-model interval answer different questions. They are not additive components of one uncertainty distribution. The completion interval conditions on the reported non-quality inputs and the implemented residual model; it does not include every source of survey or JMP revision uncertainty. Measurement priorities describe variance reduction within that model. Edition tests use later-published estimates at matched years and do not establish prospective forecast performance. The accounting differences do not identify particular households or imply a measured health burden.

`reference_results/README.md` documents the supplied summary tables. Raw input checksums are provided separately; no raw workbook is redistributed. No code license is added in this release.
