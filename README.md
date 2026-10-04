# Does women's schooling today improve a ten-year fertility forecast?

NYCU Data Mining HW1 (linear regression, ACL-style short paper).

We forecast a country's **total fertility rate (TFR) ten years ahead** (children per woman) for 145 countries and measure how much **women's schooling** adds beyond fertility's own level and recent trend. The final predictor is a 5-feature ridge model, evaluated under a rolling-origin protocol with a 10-year embargo. The test set (2005 and 2010 forecast origins, 2015 and 2020 targets) was scored once.

| | CV MAE (4 folds) | Test MAE | Test Hit@0.2 |
|---|---|---|---|
| Persistence | 0.661 | 0.378 | 37% |
| Damped trend | 0.425 | 0.303 | 40% |
| Gradient boosting (raw inputs) | 0.347 | 0.273 | 49% |
| **Ours: Ridge + engineered features** | **0.328** | **0.224** | **56%** |

On the test set the education features improve MAE by only 0.004 to 0.013 children per woman. See `paper/main.pdf`.

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh          # downloads data, runs every step, builds paper/main.pdf (~3 min)
```

The pipeline is deterministic. `results/TEST_LOCK.json` stores a hash of the frozen design (features, knot, hyperparameters). `src/06_test_and_analysis.py` refuses to score the test set if that design changes, so the test set cannot be used for tuning.

## Layout

```
src/01_download.py          raw data + SHA-256 manifest (data/raw/MANIFEST.tsv)
src/02_build_panel.py       country x start-year panel; leakage asserts
src/03_eda.py               Section 4 analysis, training rows only (t <= 1995)
src/04_folds.py             rolling-origin folds with 10-year embargo -> results/folds.json
src/05_experiments.py       CV only: knot, group elimination, alpha tuning, baselines, ablation
src/06_test_and_analysis.py one-time test scoring, significance, coefficients, error analysis
src/07_paper_assets.py      LaTeX tables + number macros (paper/generated/)
src/common.py               constants, features, folds, metrics shared by all steps
results/                    tables (csv/txt), figures (pdf/png), final_config.json, TEST_LOCK.json
paper/                      ACL template, main.tex, refs.bib
docs/                       feasibility assessments (before and after the build)
```

Every number in the paper is produced by `07_paper_assets.py` from `results/tables/`, so none of them is typed by hand.

## Data sources (all public)

| Data | Source | Used for |
|---|---|---|
| TFR, under-5 mortality | UN World Population Prospects 2024, Demographic Indicators (medium; estimates to 2023) | target, features |
| Years of schooling by sex and age | Barro-Lee v3 (2021 update; 10-year bands) and v2.2 (5-year bands, EDA only), github.com/barrolee/BarroLeeDataSet | features |
| GDP per capita | Penn World Table 10.01 (dataverse.nl, doi:10.34894/QT5BCC) | candidate feature |
| Urban share | UN World Urbanization Prospects 2025 (Degree of Urbanisation) | candidate feature |

All of these include Taiwan. Exact URLs and checksums are in `data/raw/MANIFEST.tsv`.

## Team

Team ID: **13**.

| Student ID | Name |
|---|---|
| 314580057 | 李尚哲 |
| 314580055 | 呂丞頤 |
| 314580075 | 黃駿甯 |
| 414551035 | 吳美真 |
| 112550176 | 孫傅康 |
| 112550149 | 傅維雨 |
| 315554024 | 張凱華 |
| 315554034 | 李玠廷 |

Contributions and the AI usage statement are in the paper.
