# Premier League AI Valuation

A machine learning project that estimates Premier League players' market values
using performance statistics, player attributes, historical value context, team
context, and official Premier League player stats.

The current version analyses the **2025/26 Premier League season**. In the data,
`season = 2025` means the 2025/26 season that ran from August 2025 to May 2026.
This is **not** a 2026/27 preview model.

The project aims to identify undervalued and overvalued players, explain the
model's valuation gaps, and create club-level views that can support football
analysis, recruitment-style shortlists, and content ideas.

The main purpose of the project is to develop a practical understanding of the
complete machine learning workflow, including data collection, data cleaning,
feature engineering, temporal validation, model evaluation, residual analysis,
interpretation, and reproducibility.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## Build the datasets

After placing the Transfermarkt source tables in `data/raw/`, run:

```bash
python scripts/build_datasets.py
```

This creates a labelled historical player-season dataset and an unlabelled
2025/26 scoring dataset in `data/interim/`.

Premier League official player stats can then be collected and merged into the
interim datasets:

```bash
python scripts/scrape_premierleague_player_stats.py
python scripts/build_premierleague_stats_datasets.py
```

## Run the valuation model

The combined valuation pipeline saves ranking outputs to `data/processed/`:

```bash
python scripts/build_combined_valuation_outputs.py
```

Main processed outputs:

- `scoring_results_2025_26.csv`
- `undervalued_players_2025_26.csv`
- `overvalued_players_2025_26.csv`
- `high_confidence_undervalued_2025_26.csv`
- `recruitment_bargains_2025_26.csv`

`notebooks/04_big_six_plots.ipynb` uses the processed scoring output to
generate Big Six Transfermarkt-vs-model valuation plots and player tables in
`reports/figures/`.

## Explanation tools

The project includes a first practical explainability layer. It does not use an
LLM or SHAP yet; it creates rule-based reason codes from model inputs,
position-relative percentiles, and any valuation guardrails that affected the
final estimate.

Explanation percentiles use detailed role groups where possible, such as
Centre-Back, Left-Back, Right-Back, Defensive Midfield, Attacking Midfield,
Left Winger, Right Winger, and Centre-Forward. This keeps broad model training
stable while making the reporting layer fairer and more dashboard-friendly.

Explain one player:

```bash
python scripts/explain_player_valuations.py --player saka
```

Explain a full squad:

```bash
python scripts/explain_player_valuations.py --team arsenal
```

These commands save Markdown and CSV reports in `reports/explanations/`.

If the local shell helpers are installed, the shorter versions are:

```bash
predict saka
predict_team arsenal
```

## Project structure

Reusable project logic lives in `src/prem_valuation/`:

- `data.py`: raw/interim data loading and dataset construction
- `features.py`: feature lists, engineered features, history features, team
  context, and Premier League advanced-stat features
- `modeling.py`: model builders and temporal evaluation helpers
- `rankings.py`: scoring, season-club metadata, ranking, and CSV output helpers
- `premierleague_stats.py`: Premier League stat cleaning, joining, and rate
  features
- `roles.py`: detailed role-group mapping for reporting and explanations

Key scripts live in `scripts/`:

- `build_datasets.py`: creates the base historical and scoring datasets
- `scrape_premierleague_player_stats.py`: collects official Premier League
  player statistics
- `build_premierleague_stats_datasets.py`: joins Premier League stats onto the
  model datasets
- `build_combined_valuation_outputs.py`: builds final 2025/26 valuation outputs
- `explain_player_valuations.py`: creates player/team explanation reports

## Workflow so far

1. Validated players, appearances, games, and valuation schemas and join keys.
2. Aggregated match appearances into one row per player-season.
3. Matched each completed season to the first valuation within 120 days.
4. Added stable player attributes and age at the historical season cutoff.
5. Recorded sourced data corrections separately without changing raw files.
6. Split chronologically: train through 2022/23, validate on 2023/24, reserve
   2024/25 for final testing, and retain 2025/26 for scoring.
7. Used expanding-window validation folds for feature decisions.
8. Evaluated Ridge and Random Forest baselines.
9. Added previous-season history, weighted all-competition history, team
   context, and Premier League official advanced stats.
10. Built a combined scoring model using position-specific football-ability
    models, elite/prospect trajectory signals, and valuation guardrails.
11. Scored the 2025/26 Premier League population and produced ranking, plot,
    and explanation outputs.

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
| Weighted history Random Forest | Retained | Added previous all-competition production with league/competition weights and latest known market value. |
| Team-context Random Forest | Retained | Added team points, league position, title/top-four context, and player minutes share; improved walk-forward MAE to about €4.76m. |
| Valuation update model | Retained as prior version | Predicted percentage change from preseason value and reached about €4.65m MAE on the held-out 2024/25 test season. |
| Premier League advanced stats | Retained | Added position-relevant official PL stats such as chances, box touches, passing, tackling, clearances, blocks, saves, and clean sheets. |
| Position-specific ability models | Selected | Trains separate Random Forest estimators for attackers, midfielders, defenders, and goalkeepers so roles are judged with more relevant features. |
| Elite/prospect guardrails | Selected | Prevents obvious elite players, new high-value signings, and strong young profiles from being unrealistically collapsed by one noisy season. |
| Value-band calibration | Selected | Applies a conservative correction learned from 2024/25 holdout residuals. |
| Explanation cards | Retained | Generates player/team Markdown reports explaining positive signals, negative signals, and guardrails behind each valuation. |
| 2025/26 scoring | Current output | Generates candidate undervalued, overvalued, recruitment-style bargain, Big Six plot, and explanation views for the 2025/26 Premier League season. |

## Current model status

The selected scoring approach is now a combined model. It starts with
position-specific Random Forest football-ability estimates, then applies
elite/prospect floors, high-value-signing protection, non-attacking elite
guardrails, and value-band calibration.

The combined model is designed less as a pure leaderboard metric and more as a
usable valuation system with sensible football-domain guardrails. Current
ranking outputs should be treated as candidate shortlists rather than final
truths.

The model is strongest when players have reliable recent history, current-season
Premier League minutes, and valid market-value context. It is still weaker for
unusual cases such as new arrivals, injuries, very young breakout players, and
players whose Transfermarkt value is heavily shaped by reputation, contract
status, or transfer-fee context.
