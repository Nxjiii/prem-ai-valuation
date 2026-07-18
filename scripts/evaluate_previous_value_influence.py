"""Test how much exact previous market value drives the selected model."""

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
    walk_forward_evaluate_log_change,
)
from scripts.evaluate_premierleague_stats_model import build_model_frame  # noqa: E402


PREVIOUS_VALUE_COLUMN = "previous_known_market_value_in_eur"


def add_previous_value_band(data: pd.DataFrame) -> pd.DataFrame:
    """Add a softer categorical version of previous market value."""
    result = data.copy()
    result["previous_value_band"] = pd.cut(
        result[PREVIOUS_VALUE_COLUMN],
        bins=[
            0,
            2_000_000,
            5_000_000,
            10_000_000,
            20_000_000,
            40_000_000,
            70_000_000,
            float("inf"),
        ],
        labels=[
            "€0–2m",
            "€2–5m",
            "€5–10m",
            "€10–20m",
            "€20–40m",
            "€40–70m",
            "€70m+",
        ],
    )
    return result


def evaluate_variant(
    data: pd.DataFrame,
    *,
    name: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.Series, pd.Series]:
    feature_columns = numeric_features + categorical_features
    prediction_columns = feature_columns.copy()
    if PREVIOUS_VALUE_COLUMN not in prediction_columns:
        prediction_columns.append(PREVIOUS_VALUE_COLUMN)
    builder = lambda: make_random_forest_model(
        numeric_features,
        categorical_features,
    )

    cv = walk_forward_evaluate_log_change(
        data,
        prediction_columns,
        TARGET,
        "log_value_change",
        builder,
    )

    development = data.loc[data["season"].between(2016, 2023)].copy()
    test = data.loc[data["season"].eq(2024)].copy()
    train = development.loc[
        development["log_value_change"].notna()
        & development[PREVIOUS_VALUE_COLUMN].gt(0)
    ].copy()
    test = test.loc[test[PREVIOUS_VALUE_COLUMN].gt(0)].copy()

    model = builder()
    model.fit(train[prediction_columns], train["log_value_change"])
    predictions = predict_value_from_log_change(
        model,
        test[prediction_columns],
        previous_value_column=PREVIOUS_VALUE_COLUMN,
    )
    test_metrics = pd.Series(evaluate_predictions(test[TARGET], predictions))

    cv_metrics = cv[["mae", "rmse", "r2"]].mean()
    cv_metrics.name = f"{name}_cv"
    test_metrics.name = f"{name}_test"
    return cv_metrics, test_metrics


def main() -> None:
    data = add_previous_value_band(build_model_frame())

    numeric_without_exact_previous_value = [
        feature
        for feature in TEAM_CONTEXT_NUMERIC_FEATURES
        if feature != PREVIOUS_VALUE_COLUMN
    ]

    variants = {
        "current_exact_previous_value": {
            "numeric": TEAM_CONTEXT_NUMERIC_FEATURES,
            "categorical": SELECTED_CATEGORICAL_FEATURES,
        },
        "no_exact_previous_value_feature": {
            "numeric": numeric_without_exact_previous_value,
            "categorical": SELECTED_CATEGORICAL_FEATURES,
        },
        "previous_value_band_feature": {
            "numeric": numeric_without_exact_previous_value,
            "categorical": SELECTED_CATEGORICAL_FEATURES + ["previous_value_band"],
        },
    }

    rows = []
    for name, config in variants.items():
        cv, test = evaluate_variant(
            data,
            name=name,
            numeric_features=config["numeric"],
            categorical_features=config["categorical"],
        )
        rows.append({"variant": name, "split": "walk_forward_cv", **cv.to_dict()})
        rows.append({"variant": name, "split": "2024_test", **test.to_dict()})

    summary = pd.DataFrame(rows)
    print(
        summary.pivot(index="variant", columns="split", values="mae")
        .sort_values("walk_forward_cv")
    )
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
