"""Model construction and evaluation utilities."""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def evaluate_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
    }


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical columns."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ])


def make_ridge_model(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    alpha: float = 100,
) -> Pipeline:
    """Build the selected Ridge baseline."""
    return Pipeline(steps=[
        (
            "preprocessor",
            make_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=True,
            ),
        ),
        ("regressor", Ridge(alpha=alpha)),
    ])


def make_random_forest_model(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build the selected Random Forest configuration."""
    return Pipeline(steps=[
        (
            "preprocessor",
            make_preprocessor(numeric_features, categorical_features),
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])


def predict_non_negative(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """Predict and clip impossible negative market values to zero."""
    return np.clip(model.predict(data), 0, None)


def add_log_value_change_target(
    data: pd.DataFrame,
    *,
    target: str,
    previous_value_column: str = "previous_known_market_value_in_eur",
    output_column: str = "log_value_change",
) -> pd.DataFrame:
    """Add log post-season value change relative to latest known prior value."""
    result = data.copy()
    valid_previous_value = result[previous_value_column].gt(0)
    result[output_column] = np.nan
    result.loc[valid_previous_value, output_column] = np.log(
        result.loc[valid_previous_value, target]
        / result.loc[valid_previous_value, previous_value_column]
    )
    return result


def predict_value_from_log_change(
    model: Pipeline,
    data: pd.DataFrame,
    *,
    previous_value_column: str = "previous_known_market_value_in_eur",
    min_multiplier: float = 0.2,
    max_multiplier: float = 3.0,
) -> np.ndarray:
    """Predict market value by applying a predicted log-change multiplier."""
    log_change_predictions = model.predict(data)
    multipliers = np.clip(
        np.exp(log_change_predictions),
        min_multiplier,
        max_multiplier,
    )
    return data[previous_value_column].to_numpy() * multipliers


def walk_forward_evaluate(
    data: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_builder: Callable[[], Pipeline],
    *,
    first_train_season: int = 2016,
    validation_seasons: Iterable[int] = range(2019, 2024),
) -> pd.DataFrame:
    """Evaluate a model with expanding-window season folds."""
    rows = []

    for validation_season in validation_seasons:
        fold_train = data.loc[
            data["season"].between(first_train_season, validation_season - 1)
        ]
        fold_validation = data.loc[data["season"].eq(validation_season)]

        model = model_builder()
        model.fit(fold_train[feature_columns], fold_train[target])

        predictions = predict_non_negative(
            model,
            fold_validation[feature_columns],
        )

        rows.append({
            "validation_season": validation_season,
            **evaluate_predictions(fold_validation[target], predictions),
        })

    return pd.DataFrame(rows)


def walk_forward_evaluate_log_change(
    data: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    change_target: str,
    model_builder: Callable[[], Pipeline],
    *,
    previous_value_column: str = "previous_known_market_value_in_eur",
    first_train_season: int = 2016,
    validation_seasons: Iterable[int] = range(2019, 2024),
) -> pd.DataFrame:
    """Evaluate a value-update model with expanding-window season folds."""
    rows = []
    modelling_data = data.loc[
        data[previous_value_column].gt(0)
        & data[change_target].notna()
    ].copy()

    for validation_season in validation_seasons:
        fold_train = modelling_data.loc[
            modelling_data["season"].between(
                first_train_season,
                validation_season - 1,
            )
        ]
        fold_validation = modelling_data.loc[
            modelling_data["season"].eq(validation_season)
        ]

        model = model_builder()
        model.fit(fold_train[feature_columns], fold_train[change_target])

        predictions = predict_value_from_log_change(
            model,
            fold_validation[feature_columns],
            previous_value_column=previous_value_column,
        )

        rows.append({
            "validation_season": validation_season,
            **evaluate_predictions(fold_validation[target], predictions),
            "rows": len(fold_validation),
        })

    return pd.DataFrame(rows)
