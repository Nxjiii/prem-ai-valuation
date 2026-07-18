"""Evaluate whether official Premier League stats improve the selected model."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.data import INTERIM_DIR, load_appearances, load_competitions, load_games, load_valuations
from prem_valuation.features import (
    PREMIERLEAGUE_STATS_FEATURES,
    PREMIERLEAGUE_STATS_FEATURES_ALL,
    PREMIERLEAGUE_STATS_NUMERIC_FEATURES,
    TARGET,
    TEAM_CONTEXT_FEATURES_ALL,
    TEAM_CONTEXT_NUMERIC_FEATURES,
    add_engineered_features,
    add_preseason_market_values,
    add_previous_season_features,
    add_team_context_features,
    add_weighted_previous_history,
    build_player_season_clubs,
    build_team_season_context,
    build_weighted_appearance_history,
    SELECTED_CATEGORICAL_FEATURES,
)
from prem_valuation.modeling import (
    add_log_value_change_target,
    evaluate_predictions,
    make_random_forest_model,
    predict_value_from_log_change,
    walk_forward_evaluate_log_change,
)


def build_model_frame() -> pd.DataFrame:
    model_data = pd.read_csv(
        INTERIM_DIR / "labelled_player_seasons_with_pl_stats.csv.gz",
        parse_dates=["season_end_date", "valuation_date", "date_of_birth"],
    )
    model_data = model_data.loc[
        model_data["season"].ge(2016)
        & model_data["has_premierleague_stats"]
    ].copy()
    model_data = add_engineered_features(model_data)

    valuations = load_valuations()
    model_data = add_previous_season_features(model_data)
    model_data = add_preseason_market_values(model_data, valuations)

    appearances = load_appearances()
    games = load_games()
    competitions = load_competitions()
    appearance_history = build_weighted_appearance_history(
        appearances,
        games,
        competitions,
    )
    model_data = add_weighted_previous_history(model_data, appearance_history)

    team_context = build_team_season_context(games)
    player_season_clubs = build_player_season_clubs(appearances, games)
    model_data = add_team_context_features(
        model_data,
        player_season_clubs,
        team_context,
    )
    return add_log_value_change_target(model_data, target=TARGET)


def evaluate_feature_set(
    data: pd.DataFrame,
    *,
    name: str,
    numeric_features: list[str],
    feature_columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    builder = lambda: make_random_forest_model(
        numeric_features,
        SELECTED_CATEGORICAL_FEATURES,
    )
    cv = walk_forward_evaluate_log_change(
        data,
        feature_columns,
        TARGET,
        "log_value_change",
        builder,
    )
    cv_summary = cv[["mae", "rmse", "r2"]].mean()

    development = data.loc[data["season"].between(2016, 2023)].copy()
    test = data.loc[data["season"].eq(2024)].copy()
    train = development.loc[
        development["log_value_change"].notna()
        & development["previous_known_market_value_in_eur"].gt(0)
    ].copy()
    test = test.loc[test["previous_known_market_value_in_eur"].gt(0)].copy()

    model = builder()
    model.fit(train[feature_columns], train["log_value_change"])
    predictions = predict_value_from_log_change(model, test[feature_columns])
    test_summary = pd.Series(evaluate_predictions(test[TARGET], predictions))

    cv_summary.name = f"{name}_cv"
    test_summary.name = f"{name}_test"
    return cv_summary, test_summary


def main() -> None:
    data = build_model_frame()

    feature_groups = {
        "baseline_team_context": [],
        "official_pl_stats_all": PREMIERLEAGUE_STATS_FEATURES,
        "official_pl_attack_raw": [
            "pl_total_scoring_att",
            "pl_ontarget_scoring_att",
            "pl_big_chance_created",
            "pl_big_chance_missed",
        ],
        "official_pl_passing_raw": [
            "pl_total_pass",
            "pl_accurate_pass",
            "pl_total_cross",
            "pl_accurate_cross",
            "pl_touches",
            "pl_poss_lost_all",
        ],
        "official_pl_defensive_raw": [
            "pl_total_tackle",
            "pl_won_tackle",
            "pl_interception",
            "pl_total_clearance",
            "pl_outfielder_block",
            "pl_duel_won",
            "pl_duel_lost",
            "pl_aerial_won",
            "pl_aerial_lost",
        ],
        "official_pl_defensive_per90": [
            "pl_total_tackle_per_90",
            "pl_won_tackle_per_90",
            "pl_interception_per_90",
            "pl_total_clearance_per_90",
            "pl_outfielder_block_per_90",
            "pl_duel_won_per_90",
            "pl_duel_lost_per_90",
            "pl_aerial_won_per_90",
            "pl_aerial_lost_per_90",
        ],
        "official_pl_rates_only": [
            "pl_pass_completion_rate",
            "pl_cross_completion_rate",
            "pl_tackle_success_rate",
            "pl_duel_win_rate",
            "pl_aerial_win_rate",
        ],
    }

    rows = []
    for name, extra_features in feature_groups.items():
        numeric_features = TEAM_CONTEXT_NUMERIC_FEATURES + extra_features
        feature_columns = numeric_features + SELECTED_CATEGORICAL_FEATURES
        cv, test = evaluate_feature_set(
            data,
            name=name,
            numeric_features=numeric_features,
            feature_columns=feature_columns,
        )
        rows.append({"model": name, "split": "walk_forward_cv", **cv.to_dict()})
        rows.append({"model": name, "split": "2024_test", **test.to_dict()})

    summary = pd.DataFrame(rows)
    print(
        summary.pivot(index="model", columns="split", values="mae")
        .sort_values("walk_forward_cv")
    )
    print()
    print(summary)


if __name__ == "__main__":
    main()
