"""Evaluate position-specific valuation update models.

This tests whether advanced PL stats become more useful when the model does not
have to learn one relationship across every position at once.
"""

from __future__ import annotations

from pathlib import Path
import sys

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
    predict_value_from_log_change,
)
from scripts.evaluate_premierleague_stats_model import build_model_frame  # noqa: E402


POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]

POSITION_FEATURE_GROUPS = {
    "baseline": [],
    "attack_stats": [
        "pl_total_scoring_att",
        "pl_ontarget_scoring_att",
        "pl_big_chance_created",
        "pl_big_chance_missed",
    ],
    "midfield_stats": [
        "pl_total_pass",
        "pl_accurate_pass",
        "pl_touches",
        "pl_poss_lost_all",
        "pl_big_chance_created",
        "pl_duel_won",
        "pl_duel_lost",
    ],
    "defender_stats": [
        "pl_total_tackle",
        "pl_won_tackle",
        "pl_interception",
        "pl_total_clearance",
        "pl_outfielder_block",
        "pl_aerial_won",
        "pl_aerial_lost",
    ],
    "position_relevant_stats": None,
    "all_pl_stats": PREMIERLEAGUE_STATS_FEATURES,
}

POSITION_RELEVANT_MAP = {
    "Attack": POSITION_FEATURE_GROUPS["attack_stats"],
    "Midfield": POSITION_FEATURE_GROUPS["midfield_stats"],
    "Defender": POSITION_FEATURE_GROUPS["defender_stats"],
    "Goalkeeper": [],
}


def feature_group_for_position(group_name: str, position: str) -> list[str]:
    if group_name == "position_relevant_stats":
        return POSITION_RELEVANT_MAP[position]
    return POSITION_FEATURE_GROUPS[group_name] or []


def fit_predict_one_model(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    model = make_random_forest_model(
        numeric_features,
        SELECTED_CATEGORICAL_FEATURES,
    )
    model.fit(train[feature_columns], train["log_value_change"])
    predictions = predict_value_from_log_change(model, test[feature_columns])

    result = test[["season", "player_id", "player_name", "position", TARGET]].copy()
    result["predicted_value"] = predictions
    result["absolute_error"] = (result[TARGET] - result["predicted_value"]).abs()
    return result


def evaluate_global_model(
    data: pd.DataFrame,
    *,
    validation_season: int,
    group_name: str,
) -> tuple[dict, pd.DataFrame]:
    extra_features = []
    if group_name == "all_pl_stats":
        extra_features = PREMIERLEAGUE_STATS_FEATURES
    elif group_name != "baseline":
        # For global models, only use non-position-specific static groups.
        return {}, pd.DataFrame()

    numeric_features = TEAM_CONTEXT_NUMERIC_FEATURES + extra_features
    feature_columns = numeric_features + SELECTED_CATEGORICAL_FEATURES

    train = data.loc[
        data["season"].between(2016, validation_season - 1)
        & data["log_value_change"].notna()
        & data["previous_known_market_value_in_eur"].gt(0)
    ].copy()
    test = data.loc[
        data["season"].eq(validation_season)
        & data["previous_known_market_value_in_eur"].gt(0)
    ].copy()

    predictions = fit_predict_one_model(
        train=train,
        test=test,
        numeric_features=numeric_features,
        feature_columns=feature_columns,
    )
    metrics = {
        "model_type": "global",
        "feature_group": group_name,
        "validation_season": validation_season,
        "position": "All",
        "rows": len(predictions),
        **evaluate_predictions(predictions[TARGET], predictions["predicted_value"]),
    }
    return metrics, predictions


def evaluate_position_specific_model(
    data: pd.DataFrame,
    *,
    validation_season: int,
    group_name: str,
) -> tuple[list[dict], pd.DataFrame]:
    all_predictions = []
    metric_rows = []

    for position in POSITIONS:
        extra_features = feature_group_for_position(group_name, position)
        numeric_features = TEAM_CONTEXT_NUMERIC_FEATURES + extra_features
        feature_columns = numeric_features + SELECTED_CATEGORICAL_FEATURES

        train = data.loc[
            data["season"].between(2016, validation_season - 1)
            & data["position"].eq(position)
            & data["log_value_change"].notna()
            & data["previous_known_market_value_in_eur"].gt(0)
        ].copy()
        test = data.loc[
            data["season"].eq(validation_season)
            & data["position"].eq(position)
            & data["previous_known_market_value_in_eur"].gt(0)
        ].copy()

        if len(train) < 50 or test.empty:
            continue

        predictions = fit_predict_one_model(
            train=train,
            test=test,
            numeric_features=numeric_features,
            feature_columns=feature_columns,
        )
        all_predictions.append(predictions)
        metric_rows.append({
            "model_type": "position_specific",
            "feature_group": group_name,
            "validation_season": validation_season,
            "position": position,
            "rows": len(predictions),
            **evaluate_predictions(
                predictions[TARGET],
                predictions["predicted_value"],
            ),
        })

    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    metric_rows.append({
        "model_type": "position_specific",
        "feature_group": group_name,
        "validation_season": validation_season,
        "position": "All",
        "rows": len(combined_predictions),
        **evaluate_predictions(
            combined_predictions[TARGET],
            combined_predictions["predicted_value"],
        ),
    })
    return metric_rows, combined_predictions


def main() -> None:
    data = build_model_frame()
    validation_seasons = [2019, 2020, 2021, 2022, 2023]
    feature_groups = [
        "baseline",
        "position_relevant_stats",
        "all_pl_stats",
    ]

    rows = []
    for validation_season in validation_seasons:
        for group_name in ["baseline", "all_pl_stats"]:
            metrics, _ = evaluate_global_model(
                data,
                validation_season=validation_season,
                group_name=group_name,
            )
            if metrics:
                rows.append(metrics)

        for group_name in feature_groups:
            metrics, _ = evaluate_position_specific_model(
                data,
                validation_season=validation_season,
                group_name=group_name,
            )
            rows.extend(metrics)

    results = pd.DataFrame(rows)
    all_position_summary = (
        results.loc[results["position"].eq("All")]
        .groupby(["model_type", "feature_group"], as_index=False)
        .agg(
            folds=("validation_season", "nunique"),
            avg_rows=("rows", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .sort_values("mae")
    )

    position_summary = (
        results.loc[results["position"].ne("All")]
        .groupby(["feature_group", "position"], as_index=False)
        .agg(
            folds=("validation_season", "nunique"),
            avg_rows=("rows", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .sort_values(["position", "mae"])
    )

    print("Overall walk-forward summary")
    print(all_position_summary.to_string(index=False))
    print()
    print("Position-level walk-forward summary")
    print(position_summary.to_string(index=False))


if __name__ == "__main__":
    main()
