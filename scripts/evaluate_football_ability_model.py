"""Train/evaluate a football-ability value model.

Unlike the V1 market-following model, this model deliberately excludes exact
previous Transfermarkt values. It tries to estimate value from football evidence:
age, role, minutes, team context, history, and official Premier League stats.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.features import (  # noqa: E402
    PREMIERLEAGUE_STATS_FEATURES,
    SELECTED_CATEGORICAL_FEATURES,
    TARGET,
    TEAM_CONTEXT_NUMERIC_FEATURES,
)
from prem_valuation.modeling import (  # noqa: E402
    evaluate_predictions,
    make_random_forest_model,
    predict_non_negative,
    walk_forward_evaluate,
)
from scripts.evaluate_premierleague_stats_model import build_model_frame  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports" / "football_ability_model"
PREVIOUS_VALUE_COLUMNS = [
    "previous_market_value_in_eur",
    "previous_known_market_value_in_eur",
]


ABILITY_BASE_FEATURES = [
    feature
    for feature in TEAM_CONTEXT_NUMERIC_FEATURES
    if feature not in PREVIOUS_VALUE_COLUMNS
]
ABILITY_NUMERIC_FEATURES = ABILITY_BASE_FEATURES + PREMIERLEAGUE_STATS_FEATURES
ABILITY_FEATURES = ABILITY_NUMERIC_FEATURES + SELECTED_CATEGORICAL_FEATURES

ROLLING_MEMORY_FEATURES = [
    "recent_2yr_minutes",
    "recent_2yr_goals",
    "recent_2yr_assists",
    "recent_2yr_goal_involvements",
    "recent_3yr_minutes",
    "recent_3yr_goals",
    "recent_3yr_assists",
    "recent_3yr_goal_involvements",
    "recent_3yr_avg_minutes",
    "recent_3yr_avg_goal_involvements",
    "best_recent_minutes",
    "best_recent_goal_involvements",
    "current_minutes_vs_recent_3yr_avg",
    "current_gi_vs_recent_3yr_avg",
]

POTENTIAL_AND_RISK_FEATURES = [
    "is_u21",
    "is_u23",
    "is_30_plus",
    "is_32_plus",
    "u21_minutes",
    "u23_minutes",
    "is_teen_regular",
    "is_u21_regular",
    "is_u23_high_minutes",
    "age_adjusted_minutes",
    "age_adjusted_goal_involvements",
    "minutes_drop_from_recent_avg",
    "gi_drop_from_recent_avg",
    "low_current_minutes_after_high_recent_minutes",
    "older_player_output_spike",
]

MARKET_CONTEXT_NUMERIC_FEATURES = [
    "recent_2yr_avg_value_m",
    "recent_3yr_avg_value_m",
    "recent_3yr_max_value_m",
    "recent_3yr_min_value_m",
    "recent_value_range_m",
    "value_change_last_year_m",
    "is_recent_30m_player",
    "is_recent_50m_player",
    "is_recent_80m_player",
    "is_u21_recent_20m_player",
    "is_u23_recent_40m_player",
    "has_recent_value_drop",
    "has_recent_value_rise",
]

MARKET_CONTEXT_CATEGORICAL_FEATURES = [
    "previous_value_band",
    "recent_peak_value_band",
]

YOUTH_POTENTIAL_FEATURES = [
    "young_age_factor",
    "u21_top4_minutes",
    "u23_top4_minutes",
    "u21_title_team_minutes",
    "u23_title_team_minutes",
    "u23_minutes_share_top4",
    "u23_minutes_share_title_team",
    "young_minutes_score",
    "young_goal_involvement_score",
    "young_shot_score",
    "young_chance_creation_score",
    "young_touch_score",
    "position_minutes_percentile",
    "position_age_adjusted_minutes_percentile",
    "u23_position_minutes_percentile",
    "u23_attack",
    "u23_midfield",
    "u23_defender",
    "u23_centre_forward",
    "u23_winger",
    "u23_attacking_midfield",
    "u23_defensive_midfield",
    "u23_centre_back",
    "u23_fullback",
]

ABILITY_MEMORY_NUMERIC_FEATURES = (
    ABILITY_NUMERIC_FEATURES
    + ROLLING_MEMORY_FEATURES
    + POTENTIAL_AND_RISK_FEATURES
)
ABILITY_MEMORY_FEATURES = (
    ABILITY_MEMORY_NUMERIC_FEATURES + SELECTED_CATEGORICAL_FEATURES
)

ABILITY_MARKET_CONTEXT_NUMERIC_FEATURES = (
    ABILITY_MEMORY_NUMERIC_FEATURES + MARKET_CONTEXT_NUMERIC_FEATURES
)
ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES = (
    SELECTED_CATEGORICAL_FEATURES + MARKET_CONTEXT_CATEGORICAL_FEATURES
)
ABILITY_MARKET_CONTEXT_FEATURES = (
    ABILITY_MARKET_CONTEXT_NUMERIC_FEATURES
    + ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES
)

ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES = (
    ABILITY_MARKET_CONTEXT_NUMERIC_FEATURES + YOUTH_POTENTIAL_FEATURES
)
ABILITY_YOUTH_CONTEXT_FEATURES = (
    ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES
    + ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES
)


def build_ability_model_data() -> pd.DataFrame:
    data = build_model_frame()
    data = data.loc[
        data["season"].ge(2016)
        & data["has_premierleague_stats"]
    ].copy()
    return add_memory_and_potential_features(data)


def add_memory_and_potential_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add rolling football memory and simple potential/risk flags."""
    result = data.sort_values(["player_id", "season"]).copy()
    result["goal_involvements"] = result["goals"] + result["assists"]

    rolling_source_columns = {
        "minutes_played": "minutes",
        "goals": "goals",
        "assists": "assists",
        "goal_involvements": "goal_involvements",
    }

    for source_column, output_name in rolling_source_columns.items():
        shifted = result.groupby("player_id")[source_column].shift(1)
        result[f"recent_2yr_{output_name}"] = (
            shifted.groupby(result["player_id"])
            .rolling(2, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        result[f"recent_3yr_{output_name}"] = (
            shifted.groupby(result["player_id"])
            .rolling(3, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        result[f"best_recent_{output_name}"] = (
            shifted.groupby(result["player_id"])
            .rolling(3, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

    result["recent_3yr_avg_minutes"] = result["recent_3yr_minutes"] / 3
    result["recent_3yr_avg_goal_involvements"] = (
        result["recent_3yr_goal_involvements"] / 3
    )
    result["current_minutes_vs_recent_3yr_avg"] = (
        result["minutes_played"] / result["recent_3yr_avg_minutes"].replace(0, np.nan)
    ).astype("float64")
    result["current_gi_vs_recent_3yr_avg"] = (
        result["goal_involvements"]
        / result["recent_3yr_avg_goal_involvements"].replace(0, np.nan)
    ).astype("float64")

    result["is_u21"] = result["age_at_season_end"].lt(21).astype(int)
    result["is_u23"] = result["age_at_season_end"].lt(23).astype(int)
    result["is_30_plus"] = result["age_at_season_end"].ge(30).astype(int)
    result["is_32_plus"] = result["age_at_season_end"].ge(32).astype(int)
    result["u21_minutes"] = result["minutes_played"] * result["is_u21"]
    result["u23_minutes"] = result["minutes_played"] * result["is_u23"]
    result["is_teen_regular"] = (
        result["age_at_season_end"].lt(20)
        & result["minutes_played"].ge(900)
    ).astype(int)
    result["is_u21_regular"] = (
        result["age_at_season_end"].lt(21)
        & result["minutes_played"].ge(900)
    ).astype(int)
    result["is_u23_high_minutes"] = (
        result["age_at_season_end"].lt(23)
        & result["minutes_played"].ge(1800)
    ).astype(int)
    result["age_adjusted_minutes"] = (
        result["minutes_played"] / result["age_at_season_end"].clip(lower=16)
    )
    result["age_adjusted_goal_involvements"] = (
        result["goal_involvements"]
        / result["age_at_season_end"].clip(lower=16)
    )
    result["minutes_drop_from_recent_avg"] = (
        result["recent_3yr_avg_minutes"] - result["minutes_played"]
    )
    result["gi_drop_from_recent_avg"] = (
        result["recent_3yr_avg_goal_involvements"] - result["goal_involvements"]
    )
    result["low_current_minutes_after_high_recent_minutes"] = (
        result["minutes_played"].lt(1800)
        & result["recent_3yr_avg_minutes"].ge(2200)
    ).astype(int)
    result["older_player_output_spike"] = (
        result["age_at_season_end"].ge(30)
        & result["current_gi_vs_recent_3yr_avg"].gt(1.25)
    ).astype(int)
    memory_feature_columns = ROLLING_MEMORY_FEATURES + POTENTIAL_AND_RISK_FEATURES
    result[memory_feature_columns] = (
        result[memory_feature_columns]
        .replace({pd.NA: np.nan})
        .astype("float64")
    )
    result = add_market_context_features(result)
    return add_youth_potential_features(result)


def value_band(values: pd.Series) -> pd.Series:
    """Convert euro values into broad market-context bands."""
    return pd.cut(
        values,
        bins=[
            0,
            2_000_000,
            5_000_000,
            10_000_000,
            20_000_000,
            40_000_000,
            70_000_000,
            100_000_000,
            float("inf"),
        ],
        labels=[
            "€0–2m",
            "€2–5m",
            "€5–10m",
            "€10–20m",
            "€20–40m",
            "€40–70m",
            "€70–100m",
            "€100m+",
        ],
    )


def add_market_context_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add soft market-history context without raw exact previous value."""
    result = data.sort_values(["player_id", "season"]).copy()
    previous_values = result.groupby("player_id")["market_value_in_eur"].shift(1)
    result["previous_value_band"] = value_band(previous_values)

    result["recent_2yr_avg_value_m"] = (
        previous_values.groupby(result["player_id"])
        .rolling(2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        / 1_000_000
    )
    result["recent_3yr_avg_value_m"] = (
        previous_values.groupby(result["player_id"])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        / 1_000_000
    )
    result["recent_3yr_max_value_m"] = (
        previous_values.groupby(result["player_id"])
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        / 1_000_000
    )
    result["recent_3yr_min_value_m"] = (
        previous_values.groupby(result["player_id"])
        .rolling(3, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
        / 1_000_000
    )
    result["recent_value_range_m"] = (
        result["recent_3yr_max_value_m"] - result["recent_3yr_min_value_m"]
    )
    previous_previous_values = result.groupby("player_id")["market_value_in_eur"].shift(2)
    result["value_change_last_year_m"] = (
        previous_values - previous_previous_values
    ) / 1_000_000
    result["recent_peak_value_band"] = value_band(
        result["recent_3yr_max_value_m"] * 1_000_000
    )
    result["is_recent_30m_player"] = result["recent_3yr_max_value_m"].ge(30).astype(int)
    result["is_recent_50m_player"] = result["recent_3yr_max_value_m"].ge(50).astype(int)
    result["is_recent_80m_player"] = result["recent_3yr_max_value_m"].ge(80).astype(int)
    result["is_u21_recent_20m_player"] = (
        result["age_at_season_end"].lt(21)
        & result["recent_3yr_max_value_m"].ge(20)
    ).astype(int)
    result["is_u23_recent_40m_player"] = (
        result["age_at_season_end"].lt(23)
        & result["recent_3yr_max_value_m"].ge(40)
    ).astype(int)
    result["has_recent_value_drop"] = result["value_change_last_year_m"].le(-10).astype(int)
    result["has_recent_value_rise"] = result["value_change_last_year_m"].ge(10).astype(int)
    result[MARKET_CONTEXT_NUMERIC_FEATURES] = (
        result[MARKET_CONTEXT_NUMERIC_FEATURES]
        .replace({pd.NA: np.nan})
        .astype("float64")
    )
    return result


def add_youth_potential_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add non-market potential proxies based on young senior involvement."""
    result = data.copy()
    young_age_factor = (24 - result["age_at_season_end"]).clip(lower=0)
    result["young_age_factor"] = young_age_factor

    result["u21_top4_minutes"] = (
        result["minutes_played"] * result["is_u21"] * result["team_top_4"]
    )
    result["u23_top4_minutes"] = (
        result["minutes_played"] * result["is_u23"] * result["team_top_4"]
    )
    result["u21_title_team_minutes"] = (
        result["minutes_played"] * result["is_u21"] * result["team_won_league"]
    )
    result["u23_title_team_minutes"] = (
        result["minutes_played"] * result["is_u23"] * result["team_won_league"]
    )
    result["u23_minutes_share_top4"] = (
        result["player_minutes_share"] * result["is_u23"] * result["team_top_4"]
    )
    result["u23_minutes_share_title_team"] = (
        result["player_minutes_share"] * result["is_u23"] * result["team_won_league"]
    )
    result["young_minutes_score"] = result["minutes_played"] * young_age_factor
    result["young_goal_involvement_score"] = (
        result["goal_involvements"] * young_age_factor
    )
    result["young_shot_score"] = (
        result["pl_total_scoring_att"].fillna(0) * young_age_factor
    )
    result["young_chance_creation_score"] = (
        result["pl_big_chance_created"].fillna(0) * young_age_factor
    )
    result["young_touch_score"] = (
        result["pl_touches"].fillna(0) * young_age_factor
    )

    result["position_minutes_percentile"] = (
        result.groupby(["season", "position"])["minutes_played"]
        .rank(pct=True)
    )
    result["position_age_adjusted_minutes_percentile"] = (
        result.groupby(["season", "position"])["age_adjusted_minutes"]
        .rank(pct=True)
    )
    result["u23_position_minutes_percentile"] = (
        result["position_minutes_percentile"] * result["is_u23"]
    )

    sub_position = result["sub_position"].fillna("")
    result["u23_attack"] = (
        result["is_u23"].eq(1) & result["position"].eq("Attack")
    ).astype(int)
    result["u23_midfield"] = (
        result["is_u23"].eq(1) & result["position"].eq("Midfield")
    ).astype(int)
    result["u23_defender"] = (
        result["is_u23"].eq(1) & result["position"].eq("Defender")
    ).astype(int)
    result["u23_centre_forward"] = (
        result["is_u23"].eq(1) & sub_position.eq("Centre-Forward")
    ).astype(int)
    result["u23_winger"] = (
        result["is_u23"].eq(1)
        & sub_position.isin(["Left Winger", "Right Winger"])
    ).astype(int)
    result["u23_attacking_midfield"] = (
        result["is_u23"].eq(1) & sub_position.eq("Attacking Midfield")
    ).astype(int)
    result["u23_defensive_midfield"] = (
        result["is_u23"].eq(1) & sub_position.eq("Defensive Midfield")
    ).astype(int)
    result["u23_centre_back"] = (
        result["is_u23"].eq(1) & sub_position.eq("Centre-Back")
    ).astype(int)
    result["u23_fullback"] = (
        result["is_u23"].eq(1)
        & sub_position.isin(["Left-Back", "Right-Back"])
    ).astype(int)

    result[YOUTH_POTENTIAL_FEATURES] = (
        result[YOUTH_POTENTIAL_FEATURES]
        .replace({pd.NA: np.nan})
        .astype("float64")
    )
    return result


def evaluate_ability_model(
    data: pd.DataFrame,
    *,
    numeric_features: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    builder = lambda: make_random_forest_model(
        numeric_features,
        SELECTED_CATEGORICAL_FEATURES,
    )
    return walk_forward_evaluate(
        data,
        feature_columns,
        TARGET,
        builder,
    )


def build_test_predictions(
    data: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    feature_columns: list[str],
    prediction_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data["season"].between(2016, 2023)].copy()
    test = data.loc[data["season"].eq(2024)].copy()

    model = make_random_forest_model(
        numeric_features,
        categorical_features,
    )
    model.fit(train[feature_columns], train[TARGET])
    predictions = predict_non_negative(model, test[feature_columns])

    results = test[
        [
            "season",
            "player_id",
            "player_name",
            "position",
            "sub_position",
            "age_at_season_end",
            "minutes_played",
            "previous_known_market_value_in_eur",
            "team_points",
            "team_final_position",
            "team_won_league",
            "player_minutes_share",
            TARGET,
        ]
    ].copy()
    results[prediction_column] = predictions
    results["ability_gap_vs_transfermarkt"] = (
        results[prediction_column] - results[TARGET]
    )
    results["absolute_error"] = (
        results[prediction_column] - results[TARGET]
    ).abs()
    results["transfermarkt_value_m"] = results[TARGET] / 1_000_000
    results[f"{prediction_column}_m"] = results[prediction_column] / 1_000_000
    results["ability_gap_m"] = results["ability_gap_vs_transfermarkt"] / 1_000_000
    results["absolute_error_m"] = results["absolute_error"] / 1_000_000

    feature_importances = (
        pd.DataFrame({
            "feature": model.named_steps["preprocessor"]
            .get_feature_names_out(feature_columns),
            "importance": model.named_steps["regressor"].feature_importances_,
        })
        .assign(
            feature=lambda table: table["feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return results, feature_importances


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = build_ability_model_data()

    feature_sets = {
        "output_profile_v1": {
            "numeric": ABILITY_NUMERIC_FEATURES,
            "categorical": SELECTED_CATEGORICAL_FEATURES,
            "features": ABILITY_FEATURES,
            "prediction_column": "football_ability_value",
        },
        "memory_potential_v2": {
            "numeric": ABILITY_MEMORY_NUMERIC_FEATURES,
            "categorical": SELECTED_CATEGORICAL_FEATURES,
            "features": ABILITY_MEMORY_FEATURES,
            "prediction_column": "football_ability_value_v2",
        },
        "market_context_v3": {
            "numeric": ABILITY_MARKET_CONTEXT_NUMERIC_FEATURES,
            "categorical": ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES,
            "features": ABILITY_MARKET_CONTEXT_FEATURES,
            "prediction_column": "football_ability_value_v3",
        },
        "youth_context_v4": {
            "numeric": ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES,
            "categorical": ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES,
            "features": ABILITY_YOUTH_CONTEXT_FEATURES,
            "prediction_column": "football_ability_value_v4",
        },
    }

    metric_rows = []
    prediction_tables = {}
    importance_tables = {}

    for model_name, config in feature_sets.items():
        cv = evaluate_ability_model(
            data,
            numeric_features=config["numeric"],
            feature_columns=config["features"],
        )
        test_results, feature_importances = build_test_predictions(
            data,
            numeric_features=config["numeric"],
            categorical_features=config["categorical"],
            feature_columns=config["features"],
            prediction_column=config["prediction_column"],
        )

        cv_summary = cv[["mae", "rmse", "r2"]].mean()
        test_metrics = pd.Series(
            evaluate_predictions(
                test_results[TARGET],
                test_results[config["prediction_column"]],
            )
        )
        metric_rows.append({"model": model_name, "split": "walk_forward_cv", **cv_summary})
        metric_rows.append({"model": model_name, "split": "2024_test", **test_metrics})
        prediction_tables[model_name] = test_results
        importance_tables[model_name] = feature_importances

        cv.to_csv(REPORTS_DIR / f"{model_name}_walk_forward_cv.csv", index=False)
        test_results.to_csv(
            REPORTS_DIR / f"{model_name}_test_2024_predictions.csv",
            index=False,
        )
        feature_importances.to_csv(
            REPORTS_DIR / f"{model_name}_feature_importance.csv",
            index=False,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)

    test_results = prediction_tables["youth_context_v4"]
    feature_importances = importance_tables["youth_context_v4"]
    prediction_column = "football_ability_value_v4"
    prediction_column_m = "football_ability_value_v4_m"

    print("Football ability model metrics")
    print(metrics.pivot(index="model", columns="split", values="mae").to_string())
    print()
    print("Top feature importances")
    print(feature_importances.head(20).to_string(index=False))
    print()
    print("2024 players ability model rates above Transfermarkt")
    print(
        test_results.sort_values("ability_gap_vs_transfermarkt", ascending=False)
        [
            [
                "player_name",
                "position",
                "age_at_season_end",
                "minutes_played",
                "transfermarkt_value_m",
                prediction_column_m,
                "ability_gap_m",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print()
    print("2024 players ability model rates below Transfermarkt")
    print(
        test_results.sort_values("ability_gap_vs_transfermarkt", ascending=True)
        [
            [
                "player_name",
                "position",
                "age_at_season_end",
                "minutes_played",
                "transfermarkt_value_m",
                prediction_column_m,
                "ability_gap_m",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print()
    print(f"Saved outputs to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
