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
8. Evaluated the frozen Ridge baseline on 2024/25 and analysed errors by
   position, value, minutes, and age.
9. Compared Random Forest models against Ridge and added previous-season
   history features.
10. Scored the 2025/26 Premier League population and produced first valuation
    gap ranking views.

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
| Ridge alpha tuning | Retained | Alpha 100 produced the lowest mean walk-forward MAE, though the gain was marginal. |
| Frozen Ridge test | Baseline complete | Achieved €10.39m MAE and 0.426 R² on the 2024/25 test season. |
| Random Forest | Retained | Beat Ridge on every walk-forward fold and improved final test MAE to €9.46m. |
| History Random Forest | Selected | Previous-season stats and previous market value reduced final test MAE to €5.80m and lifted R² to 0.824. |
| 2025/26 scoring | First output | Generated candidate undervalued, overvalued, high-confidence, and recruitment-style bargain lists. |

The selected model is now a history-aware Random Forest. It averaged
approximately **€5.94m MAE** across five walk-forward validation seasons and
achieved **€5.80m MAE** on the held-out 2024/25 test season.

Current ranking outputs should be treated as candidate shortlists rather than
final truths. The model is strongest for players with previous Premier League
market values, and weaker for new arrivals or breakout players with limited
historical context.
