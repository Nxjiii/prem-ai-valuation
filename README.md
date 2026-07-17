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

The nonlinear modelling notebook can then save ranking outputs to
`data/processed/`, including:

- `scoring_results_2025_26.csv`
- `undervalued_players_2025_26.csv`
- `overvalued_players_2025_26.csv`
- `high_confidence_undervalued_2025_26.csv`
- `recruitment_bargains_2025_26.csv`

`notebooks/04_big_six_plots.ipynb` uses these outputs to generate Big Six
Transfermarkt-vs-model valuation plots in `reports/figures/`.

Reusable project logic lives in `src/prem_valuation/`:

- `data.py`: raw/interim data loading and dataset construction
- `features.py`: feature lists, engineered features, PL history features, and
  weighted all-competition history features
- `modeling.py`: model builders and temporal evaluation helpers
- `rankings.py`: scoring, season-club metadata, ranking, and CSV output helpers

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
| History Random Forest | Retained | Previous-season PL stats and previous market value reduced final test MAE to €5.80m and lifted R² to 0.824. |
| Weighted history Random Forest | Selected | Adds previous all-competition production with league/competition weights and latest preseason market value; improves walk-forward MAE and reduces the new-signing blind spot. |
| Valuation update model | Experimental | Predicts percentage change from preseason value; helps some elite compression cases but has slightly worse validation MAE. Saved as comparison columns. |
| 2025/26 scoring | First output | Generated candidate undervalued, overvalued, high-confidence, and recruitment-style bargain lists. |

The selected model is now a weighted-history Random Forest. It averaged
approximately **€5.61m MAE** across five walk-forward validation seasons. Its
held-out 2024/25 test performance is close to the PL-history model, while its
2025/26 scoring is better suited to players arriving from outside the Premier
League because previous non-PL production and latest known preseason market
value are no longer ignored.

Current ranking outputs should be treated as candidate shortlists rather than
final truths. The model is strongest for players with previous Premier League
market values, and weaker for new arrivals or breakout players with limited
historical context.
