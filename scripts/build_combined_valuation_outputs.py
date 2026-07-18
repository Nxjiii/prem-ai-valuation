"""Build the combined 2025/26 valuation outputs.

The combined model keeps the football-ability regressor as the base estimate,
then uses the elite-trajectory classifier as a protection/floor signal for
young or prime-age elite players. This prevents one-season output from dragging
players like Saka/Foden/Haaland unrealistically far below their established
trajectory, while still allowing the model to disagree with Transfermarkt.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.data import PROCESSED_DIR  # noqa: E402
from prem_valuation.modeling import make_random_forest_model, predict_non_negative  # noqa: E402
from prem_valuation.rankings import (  # noqa: E402
    build_scoring_results,
    make_ranking_tables,
    save_ranking_outputs,
)
from scripts.evaluate_football_ability_model import (  # noqa: E402
    ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES,
    ABILITY_YOUTH_CONTEXT_FEATURES,
    ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES,
    add_memory_and_potential_features,
    build_ability_model_data,
)
from prem_valuation.features import PREMIERLEAGUE_STATS_FEATURES  # noqa: E402
from scripts.evaluate_prospect_model import (  # noqa: E402
    PROSPECT_AGE_LIMIT,
    PROSPECT_FEATURES,
    PROSPECT_NUMERIC_FEATURES,
    add_future_elite_labels,
    build_scoring_model_frame,
    make_prospect_classifier,
    prospect_training_pool,
)
from prem_valuation.data import load_valuations  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports" / "combined_model"
CALIBRATION_SHRINKAGE = 0.50
MAX_CALIBRATION_ADJUSTMENT = 12_500_000
VALUE_BAND_BINS = [
    0,
    5_000_000,
    15_000_000,
    30_000_000,
    60_000_000,
    100_000_000,
    float("inf"),
]
VALUE_BAND_LABELS = [
    "£0–5m",
    "£5–15m",
    "£15–30m",
    "£30–60m",
    "£60–100m",
    "£100m+",
]

BASE_ABILITY_NUMERIC_FEATURES = [
    feature
    for feature in ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES
    if feature not in PREMIERLEAGUE_STATS_FEATURES
]

POSITION_RELEVANT_PL_FEATURES = {
    "Attack": [
        "pl_total_scoring_att",
        "pl_ontarget_scoring_att",
        "pl_big_chance_created",
        "pl_big_chance_missed",
        "pl_big_chance_scored",
        "pl_total_scoring_att_per_90",
        "pl_ontarget_scoring_att_per_90",
        "pl_big_chance_created_per_90",
        "pl_big_chance_missed_per_90",
        "pl_big_chance_scored_per_90",
        "pl_touches_in_opp_box",
        "pl_touches_in_opp_box_per_90",
        "pl_hit_woodwork",
        "pl_penalty_won",
        "pl_total_fastbreak",
        "pl_att_hd_goal",
        "pl_att_rf_goal",
        "pl_att_lf_goal",
        "pl_touches",
        "pl_touches_per_90",
        "pl_total_pass",
        "pl_accurate_pass",
        "pl_pass_completion_rate",
        "pl_poss_lost_all",
        "pl_poss_lost_all_per_90",
        "pl_duel_won",
        "pl_duel_win_rate",
    ],
    "Midfield": [
        "pl_total_pass",
        "pl_accurate_pass",
        "pl_pass_completion_rate",
        "pl_total_long_balls",
        "pl_accurate_long_balls",
        "pl_long_ball_completion_rate",
        "pl_total_through_ball",
        "pl_accurate_through_ball",
        "pl_through_ball_completion_rate",
        "pl_touches",
        "pl_touches_per_90",
        "pl_poss_lost_all",
        "pl_poss_lost_all_per_90",
        "pl_big_chance_created",
        "pl_big_chance_created_per_90",
        "pl_total_corners_intobox",
        "pl_total_cross",
        "pl_accurate_cross",
        "pl_cross_completion_rate",
        "pl_total_tackle",
        "pl_won_tackle",
        "pl_tackle_success_rate",
        "pl_interception",
        "pl_interception_per_90",
        "pl_duel_won",
        "pl_duel_lost",
        "pl_duel_win_rate",
    ],
    "Defender": [
        "pl_total_tackle",
        "pl_won_tackle",
        "pl_tackle_success_rate",
        "pl_interception",
        "pl_interception_per_90",
        "pl_total_clearance",
        "pl_total_clearance_per_90",
        "pl_effective_clearance",
        "pl_effective_clearance_per_90",
        "pl_head_clearance",
        "pl_head_clearance_per_90",
        "pl_clearance_off_line",
        "pl_clearance_off_line_per_90",
        "pl_outfielder_block",
        "pl_outfielder_block_per_90",
        "pl_blocked_scoring_att",
        "pl_blocked_scoring_att_per_90",
        "pl_duel_won",
        "pl_duel_lost",
        "pl_duel_win_rate",
        "pl_aerial_won",
        "pl_aerial_lost",
        "pl_aerial_win_rate",
        "pl_last_man_tackle",
        "pl_penalty_conceded",
        "pl_own_goals",
        "pl_error_lead_to_goal",
        "pl_total_pass",
        "pl_accurate_pass",
        "pl_pass_completion_rate",
        "pl_touches",
    ],
    "Goalkeeper": [
        "pl_total_pass",
        "pl_accurate_pass",
        "pl_pass_completion_rate",
        "pl_touches",
        "pl_error_lead_to_goal",
        "pl_saves",
        "pl_penalty_save",
        "pl_goals_conceded",
        "pl_clean_sheet",
        "pl_save_rate",
        "pl_keeper_throws",
        "pl_goal_kicks",
    ],
}


def numeric_features_for_position(position: str) -> list[str]:
    """Use position-relevant PL stat groups inside each position model."""
    return BASE_ABILITY_NUMERIC_FEATURES + POSITION_RELEVANT_PL_FEATURES[position]


def feature_columns_for_position(position: str) -> list[str]:
    return numeric_features_for_position(position) + ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES


def value_band(values: pd.Series) -> pd.Series:
    """Assign the value bands used for calibration."""
    return pd.cut(
        values,
        bins=VALUE_BAND_BINS,
        labels=VALUE_BAND_LABELS,
        include_lowest=True,
    )


def score_current_football_ability(
    historical_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train position-specific football-ability models and score 2025/26 players."""
    global_model = make_random_forest_model(
        ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES,
        ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES,
    )
    global_model.fit(
        historical_data[ABILITY_YOUTH_CONTEXT_FEATURES],
        historical_data["market_value_in_eur"],
    )

    scored = current_data.copy()
    scored["football_ability_value"] = predict_non_negative(
        global_model,
        scored[ABILITY_YOUTH_CONTEXT_FEATURES],
    )
    scored["football_ability_model_type"] = "global_fallback"

    for position in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
        train = historical_data.loc[historical_data["position"].eq(position)].copy()
        current_position = scored["position"].eq(position)
        if len(train) < 150 or not current_position.any():
            continue

        numeric_features = numeric_features_for_position(position)
        feature_columns = feature_columns_for_position(position)
        model = make_random_forest_model(
            numeric_features,
            ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES,
        )
        model.fit(train[feature_columns], train["market_value_in_eur"])
        scored.loc[current_position, "football_ability_value"] = predict_non_negative(
            model,
            scored.loc[current_position, feature_columns],
        )
        scored.loc[current_position, "football_ability_model_type"] = (
            f"position_specific_{position.lower()}"
        )
    return scored


def score_current_elite_trajectory(
    historical_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> pd.DataFrame:
    """Train the elite-trajectory classifier and score 2025/26 players."""
    labelled = add_future_elite_labels(historical_data, load_valuations())
    prospect_data = prospect_training_pool(labelled)

    model = make_prospect_classifier()
    model.fit(prospect_data[PROSPECT_FEATURES], prospect_data["future_elite_player"])

    scored = current_data.copy()
    probabilities = np.zeros(len(scored))
    eligible = (
        scored["age_at_season_end"].lt(PROSPECT_AGE_LIMIT)
        & scored["minutes_played"].gt(0)
        & scored["current_market_value_in_eur"].notna()
    )
    probabilities[eligible.to_numpy()] = model.predict_proba(
        scored.loc[eligible, PROSPECT_FEATURES]
    )[:, 1]
    scored["elite_trajectory_probability"] = probabilities
    return scored


def add_combined_value(scored: pd.DataFrame) -> pd.DataFrame:
    """Combine football value with an elite-trajectory floor."""
    result = scored.copy()
    previous_reference = result[
        [
            "previous_known_market_value_in_eur",
            "previous_market_value_in_eur",
        ]
    ].max(axis=1)
    recent_peak_reference = result["recent_3yr_max_value_m"] * 1_000_000
    result["elite_trajectory_reference_value"] = pd.concat(
        [previous_reference, recent_peak_reference],
        axis=1,
    ).max(axis=1)

    probability = result["elite_trajectory_probability"].fillna(0)
    eligible_for_floor = (
        result["age_at_season_end"].lt(PROSPECT_AGE_LIMIT)
        & probability.ge(0.50)
        & result["elite_trajectory_reference_value"].gt(0)
    )
    floor_multiplier = 0.35 + (0.50 * probability)
    result["elite_trajectory_floor_value"] = 0.0
    result.loc[eligible_for_floor, "elite_trajectory_floor_value"] = (
        result.loc[eligible_for_floor, "elite_trajectory_reference_value"]
        * floor_multiplier.loc[eligible_for_floor]
    )
    high_confidence_elite = (
        result["age_at_season_end"].lt(PROSPECT_AGE_LIMIT)
        & probability.ge(0.75)
        & result["current_market_value_in_eur"].gt(0)
    )
    result["elite_market_sanity_floor_value"] = 0.0
    result.loc[high_confidence_elite, "elite_market_sanity_floor_value"] = (
        result.loc[high_confidence_elite, "current_market_value_in_eur"] * 0.85
    )
    established_elite = (
        result["age_at_season_end"].lt(30)
        & result["previous_known_market_value_in_eur"].ge(100_000_000)
        & result["minutes_played"].ge(1_800)
        & result["player_minutes_share"].ge(0.50)
        & (
            result["team_top_4"].eq(1)
            | result["team_points"].ge(65)
        )
    )
    result["established_elite_status_floor_value"] = 0.0
    result.loc[established_elite, "established_elite_status_floor_value"] = (
        result.loc[established_elite, "previous_known_market_value_in_eur"]
    )
    ultra_elite_still_performing = (
        result["age_at_season_end"].lt(30)
        & result["current_market_value_in_eur"].ge(150_000_000)
        & probability.ge(0.85)
        & result["minutes_played"].ge(1_800)
        & result["player_minutes_share"].ge(0.50)
    )
    result["ultra_elite_current_value_floor"] = 0.0
    result.loc[ultra_elite_still_performing, "ultra_elite_current_value_floor"] = (
        result.loc[ultra_elite_still_performing, "current_market_value_in_eur"]
    )
    elite_non_attacking_role = (
        result["position"].isin(["Midfield", "Defender"])
        & result["age_at_season_end"].lt(29)
        & result["previous_known_market_value_in_eur"].ge(70_000_000)
        & result["minutes_played"].ge(1_200)
        & result["player_minutes_share"].ge(0.35)
        & (
            result["team_top_4"].eq(1)
            | result["team_points"].ge(65)
        )
    )
    result["elite_non_attacking_role_floor_value"] = 0.0
    result.loc[elite_non_attacking_role, "elite_non_attacking_role_floor_value"] = (
        result.loc[elite_non_attacking_role, "previous_known_market_value_in_eur"]
    )
    new_high_value_signing = (
        result["age_at_season_end"].lt(27)
        & result["previous_market_value_in_eur"].isna()
        & result["previous_known_market_value_in_eur"].ge(60_000_000)
        & result["current_market_value_in_eur"].ge(60_000_000)
        & result["minutes_played"].ge(1_200)
        & result["player_minutes_share"].ge(0.35)
    )
    result["new_high_value_signing_floor_value"] = 0.0
    result.loc[new_high_value_signing, "new_high_value_signing_floor_value"] = (
        np.minimum(
            result.loc[
                new_high_value_signing,
                [
                    "previous_known_market_value_in_eur",
                    "elite_trajectory_floor_value",
                ],
            ].max(axis=1),
            result.loc[new_high_value_signing, "current_market_value_in_eur"],
        )
    )

    result["combined_model_value"] = np.maximum(
        result["football_ability_value"],
        result[
            [
                "elite_trajectory_floor_value",
                "elite_market_sanity_floor_value",
                "established_elite_status_floor_value",
                "ultra_elite_current_value_floor",
                "elite_non_attacking_role_floor_value",
                "new_high_value_signing_floor_value",
            ]
        ].max(axis=1),
    )
    result["elite_trajectory_adjustment"] = (
        result["combined_model_value"] - result["football_ability_value"]
    )
    return result


def build_combined_scoring_frame() -> pd.DataFrame:
    """Build, score, and combine the current-season model frame."""
    historical_data = build_ability_model_data()
    current_raw = build_scoring_model_frame(historical_data)
    current_raw["season_club_name"] = current_raw["team_name"]

    combined = pd.concat([historical_data, current_raw], ignore_index=True, sort=False)
    combined = add_memory_and_potential_features(combined)
    current_data = combined.loc[combined["season"].eq(2025)].copy()

    scored = score_current_football_ability(historical_data, current_data)
    scored = score_current_elite_trajectory(historical_data, scored)
    scored = add_combined_value(scored)
    calibration = build_value_band_calibration(historical_data)
    calibration.to_csv(REPORTS_DIR / "value_band_calibration.csv", index=False)
    return apply_value_band_calibration(scored, calibration)


def build_2024_holdout_predictions(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Predict 2024 from earlier seasons for out-of-sample calibration."""
    train = historical_data.loc[historical_data["season"].between(2016, 2023)].copy()
    holdout = historical_data.loc[historical_data["season"].eq(2024)].copy()
    holdout["current_market_value_in_eur"] = holdout["market_value_in_eur"]

    scored = score_current_football_ability(train, holdout)
    scored = score_current_elite_trajectory(train, scored)
    scored = add_combined_value(scored)
    scored["actual_value"] = scored["market_value_in_eur"]
    scored["holdout_residual"] = scored["actual_value"] - scored["combined_model_value"]
    scored["calibration_band"] = value_band(scored["actual_value"])
    return scored


def build_value_band_calibration(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Learn a conservative value-band residual correction from 2024 holdout."""
    holdout = build_2024_holdout_predictions(historical_data)
    calibration = (
        holdout.dropna(subset=["calibration_band"])
        .groupby("calibration_band", observed=False)
        .agg(
            rows=("player_id", "count"),
            median_residual=("holdout_residual", "median"),
            mean_residual=("holdout_residual", "mean"),
            median_prediction=("combined_model_value", "median"),
            median_actual=("actual_value", "median"),
        )
        .reset_index()
    )
    calibration["calibration_adjustment"] = (
        calibration["median_residual"] * CALIBRATION_SHRINKAGE
    ).clip(
        lower=-MAX_CALIBRATION_ADJUSTMENT,
        upper=MAX_CALIBRATION_ADJUSTMENT,
    )
    return calibration


def apply_value_band_calibration(
    scored: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the learned band correction to current scored players."""
    result = scored.copy()
    result["uncalibrated_combined_model_value"] = result["combined_model_value"]
    result["calibration_band"] = value_band(result["current_market_value_in_eur"])
    adjustment_lookup = calibration.set_index("calibration_band")[
        "calibration_adjustment"
    ]
    result["value_band_calibration_adjustment"] = (
        result["calibration_band"].map(adjustment_lookup).astype("float64").fillna(0)
    )
    current_gap = result["combined_model_value"] - result["current_market_value_in_eur"]
    positive_adjustment = result["value_band_calibration_adjustment"].gt(0)
    negative_adjustment = result["value_band_calibration_adjustment"].lt(0)
    result.loc[positive_adjustment, "value_band_calibration_adjustment"] = np.minimum(
        result.loc[positive_adjustment, "value_band_calibration_adjustment"],
        (-current_gap.loc[positive_adjustment]).clip(lower=0),
    )
    result.loc[negative_adjustment, "value_band_calibration_adjustment"] = np.maximum(
        result.loc[negative_adjustment, "value_band_calibration_adjustment"],
        (-current_gap.loc[negative_adjustment]).clip(upper=0),
    )
    result["combined_model_value"] = (
        result["combined_model_value"]
        + result["value_band_calibration_adjustment"]
    ).clip(lower=0)
    result["combined_model_value"] = np.maximum(
        result["combined_model_value"],
        result[
            [
                "ultra_elite_current_value_floor",
                "established_elite_status_floor_value",
                "elite_non_attacking_role_floor_value",
            ]
        ].max(axis=1),
    )
    result["calibration_adjusted_value_delta"] = (
        result["combined_model_value"] - result["uncalibrated_combined_model_value"]
    )
    return result


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    scored = build_combined_scoring_frame()
    scoring_results = build_scoring_results(
        scored,
        scored["combined_model_value"].to_numpy(),
    )

    extra_columns = [
        "player_id",
        "football_ability_value",
        "football_ability_model_type",
        "elite_trajectory_probability",
        "elite_trajectory_reference_value",
        "elite_trajectory_floor_value",
        "elite_market_sanity_floor_value",
        "established_elite_status_floor_value",
        "ultra_elite_current_value_floor",
        "elite_non_attacking_role_floor_value",
        "new_high_value_signing_floor_value",
        "elite_trajectory_adjustment",
        "uncalibrated_combined_model_value",
        "calibration_band",
        "value_band_calibration_adjustment",
        "calibration_adjusted_value_delta",
        "combined_model_value",
        *PREMIERLEAGUE_STATS_FEATURES,
    ]
    scoring_results = scoring_results.merge(
        scored[extra_columns],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    scoring_results["predicted_value"] = scoring_results["combined_model_value"]
    scoring_results["valuation_gap"] = (
        scoring_results["predicted_value"]
        - scoring_results["current_market_value_in_eur"]
    )
    scoring_results["absolute_gap"] = scoring_results["valuation_gap"].abs()

    ranking_outputs = make_ranking_tables(scoring_results)
    manifest = save_ranking_outputs(ranking_outputs, PROCESSED_DIR)
    scoring_results.to_csv(REPORTS_DIR / "scoring_results_2025_26_combined.csv", index=False)
    manifest.to_csv(REPORTS_DIR / "saved_outputs_manifest.csv", index=False)

    print("Saved combined scoring outputs")
    print(manifest.to_string(index=False))
    print()
    print("Biggest elite-trajectory adjustments")
    print(
        scoring_results.sort_values("elite_trajectory_adjustment", ascending=False)
        [
            [
                "player_name",
                "current_club_name",
                "position",
                "age_at_season_end",
                "minutes_played",
                "current_market_value_in_eur",
                "football_ability_value",
                "elite_trajectory_probability",
                "elite_trajectory_adjustment",
                "predicted_value",
                "valuation_gap",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    print()
    print(f"Full combined report saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
