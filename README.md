# Premier League AI Valuation

A machine learning project that predicts Premier League players'
market values using performance statistics and player attributes.

The project aims to identify undervalued and overvalued players, build an
underrated XI, and highlight potential bargains for each club. It may later be
extended to player similarity, recruitment recommendations, transfer analysis,
and squad-building tools.

The main purpose of the project is to develop a practical understanding of the
complete machine learning workflow, including data collection, data cleaning,
feature engineering, model evaluation, residual analysis, interpretation, and
reproducibility.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## Build the interim datasets

After placing the source tables in `data/raw/`, run:

```bash
python scripts/build_datasets.py
```

This creates a labelled historical player-season dataset and an unlabelled
2025/26 scoring dataset in `data/interim/`.

## Workflow so far

1. Validated players, appearances, games, and valuation schemas and join keys.
2. Aggregated match appearances into one row per player-season.
3. Matched each completed season to the first valuation within 120 days.
4. Added stable player attributes and age at the historical season cutoff.
5. Recorded sourced data corrections separately without changing raw files.
6. Split chronologically: train through 2022/23, validate on 2023/24, reserve
   2024/25 for final testing, and retain 2025/26 for later scoring.
7. Used five expanding-window validation folds for feature decisions.

## Experiments

| Experiment | Decision | Finding |
|---|---|---|
| Median dummy | Benchmark | Established the minimum baseline. |
| Log-target Ridge | Rejected | Worse errors and implausible elite predictions. |
| Raw-target Ridge | Retained | Best target representation for the linear baseline. |
| Numeric season trend | Rejected | Improved RMSE but worsened primary MAE. |
| Nonlinear age curve | Retained | Values peak in the mid-20s rather than changing linearly. |
| Sub-position | Retained | Granular roles improved temporal validation. |
| Stabilised per-90 rates | Rejected | Gains were tiny and inconsistent across seasons. |

The selected Ridge features currently average approximately **€9.31m MAE** and
**0.502 R²** across five walk-forward validation seasons. Next: tune Ridge
regularisation before evaluating once on the untouched 2024/25 test season.
