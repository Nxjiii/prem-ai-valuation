"""Analyze residuals for the selected valuation update model."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.features import (  # noqa: E402
    SELECTED_CATEGORICAL_FEATURES,
    TARGET,
    TEAM_CONTEXT_FEATURES_ALL,
    TEAM_CONTEXT_NUMERIC_FEATURES,
)
from prem_valuation.modeling import (  # noqa: E402
    evaluate_predictions,
    make_random_forest_model,
    predict_value_from_log_change,
)
from scripts.evaluate_premierleague_stats_model import build_model_frame  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports" / "model_analysis"


def feature_importance_table(model, feature_columns: list[str]) -> pd.DataFrame:
    """Extract fitted Random Forest feature importances from a sklearn pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out(feature_columns)
    feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": regressor.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def build_test_results(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit selected model and return 2024/25 test predictions + importances."""
    train = data.loc[
        data["season"].between(2016, 2023)
        & data["log_value_change"].notna()
        & data["previous_known_market_value_in_eur"].gt(0)
    ].copy()
    test = data.loc[
        data["season"].eq(2024)
        & data["previous_known_market_value_in_eur"].gt(0)
    ].copy()

    model = make_random_forest_model(
        TEAM_CONTEXT_NUMERIC_FEATURES,
        SELECTED_CATEGORICAL_FEATURES,
    )
    model.fit(train[TEAM_CONTEXT_FEATURES_ALL], train["log_value_change"])
    predictions = predict_value_from_log_change(
        model,
        test[TEAM_CONTEXT_FEATURES_ALL],
    )

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
            "previous_weighted_minutes",
            "previous_weighted_goals",
            "previous_weighted_assists",
            "team_points",
            "team_goal_difference",
            "team_final_position",
            "team_won_league",
            "player_minutes_share",
            TARGET,
        ]
    ].copy()
    results["predicted_value"] = predictions
    results["residual"] = results[TARGET] - results["predicted_value"]
    results["absolute_error"] = results["residual"].abs()
    results["actual_value_m"] = results[TARGET] / 1_000_000
    results["predicted_value_m"] = results["predicted_value"] / 1_000_000
    results["residual_m"] = results["residual"] / 1_000_000
    results["absolute_error_m"] = results["absolute_error"] / 1_000_000
    results["error_direction"] = "model_too_high"
    results.loc[results["residual"].gt(0), "error_direction"] = "model_too_low"

    importances = feature_importance_table(model, TEAM_CONTEXT_FEATURES_ALL)
    return results, importances


def summarise_errors(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create grouped residual summaries."""
    summaries = {}

    summaries["metrics"] = pd.DataFrame([
        evaluate_predictions(results[TARGET], results["predicted_value"])
    ])

    summaries["by_position"] = (
        results.groupby("position", as_index=False)
        .agg(
            players=("player_id", "count"),
            actual_mean=("actual_value_m", "mean"),
            predicted_mean=("predicted_value_m", "mean"),
            mae=("absolute_error", "mean"),
            mean_residual=("residual", "mean"),
        )
        .sort_values("mae", ascending=False)
    )

    results = results.copy()
    results["value_band"] = pd.cut(
        results[TARGET],
        bins=[0, 5_000_000, 15_000_000, 30_000_000, 60_000_000, float("inf")],
        labels=["€0–5m", "€5–15m", "€15–30m", "€30–60m", "€60m+"],
    )
    results["minutes_band"] = pd.cut(
        results["minutes_played"],
        bins=[-1, 449, 899, 1799, float("inf")],
        labels=["Under 450", "450–899", "900–1799", "1800+"],
    )
    results["age_band"] = pd.cut(
        results["age_at_season_end"],
        bins=[16, 20, 23, 26, 29, 32, 36, 50],
        right=False,
    )

    for group_name in ["value_band", "minutes_band", "age_band"]:
        summaries[f"by_{group_name}"] = (
            results.groupby(group_name, observed=False)
            .agg(
                players=("player_id", "count"),
                actual_mean=("actual_value_m", "mean"),
                predicted_mean=("predicted_value_m", "mean"),
                mae=("absolute_error", "mean"),
                mean_residual=("residual", "mean"),
            )
            .reset_index()
        )

    summaries["largest_errors"] = (
        results.sort_values("absolute_error", ascending=False)
        .head(30)
    )
    summaries["model_too_low"] = results.sort_values("residual", ascending=False).head(30)
    summaries["model_too_high"] = results.sort_values("residual", ascending=True).head(30)
    return summaries


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = build_model_frame()
    results, importances = build_test_results(data)
    summaries = summarise_errors(results)

    results.to_csv(REPORTS_DIR / "test_2024_residuals.csv", index=False)
    importances.to_csv(REPORTS_DIR / "selected_model_feature_importance.csv", index=False)

    for name, table in summaries.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", index=False)

    print("2024/25 test metrics")
    print(summaries["metrics"].to_string(index=False))
    print()
    print("Top feature importances")
    print(importances.head(15).to_string(index=False))
    print()
    print("Error by value band")
    print(summaries["by_value_band"].to_string(index=False))
    print()
    print("Largest errors")
    print(
        summaries["largest_errors"][
            [
                "player_name",
                "position",
                "minutes_played",
                "actual_value_m",
                "predicted_value_m",
                "residual_m",
                "absolute_error_m",
            ]
        ].head(15).to_string(index=False)
    )
    print()
    print(f"Saved analysis files to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
