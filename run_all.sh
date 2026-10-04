#!/usr/bin/env bash
# Reproduce every number, table and figure from the raw public data.
# Order matters: 05 (CV only) freezes the design in results/final_config.json;
# 06 scores the test set once and refuses to run if that config ever changes.
set -euo pipefail
cd "$(dirname "$0")/src"
python3 01_download.py           # raw data + SHA-256 manifest
python3 02_build_panel.py        # country x start-year panel, leakage asserts
python3 03_eda.py                # Section 4 (training rows only)
python3 04_folds.py              # rolling-origin folds with 10-year embargo
python3 05_experiments.py        # tuning, baselines, ablation (CV only)
python3 06_test_and_analysis.py  # one-time test evaluation + Section 8
python3 07_paper_assets.py       # LaTeX tables and number macros
cd ../paper && latexmk -pdf -interaction=nonstopmode main.tex > /dev/null && echo "paper/main.pdf built"
