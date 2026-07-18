"""Train/evaluate a young-player elite trajectory model.

This is deliberately separate from the valuation regressor. The valuation model
answers: "what value would football evidence imply?" This prospect model asks:
"does this young player look like someone who is/will be a high-value player?"

That second question is what the football-ability model struggles with for
players like Saka, Nwaneri, Mainoo, Yoro, and similar high-upside cases.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.data import (  # noqa: E402
    INTERIM_DIR,
    PROCESSED_DIR,
    load_appearances,
    load_competitions,
    load_games,
    load_players,
    load_valuations,
)
from prem_valuation.features import (  # noqa: E402
    TARGET,
    add_engineered_features,
    add_latest_history_to_scoring,
    add_latest_weighted_history_to_scoring,
    add_preseason_market_values,
    add_team_context_features,
    build_player_season_clubs,
    build_team_season_context,
    build_weighted_appearance_history,
)
from prem_valuation.modeling import make_preprocessor  # noqa: E402
from prem_valuation.rankings import attach_current_player_values  # noqa: E402
from scripts.evaluate_football_ability_model import (  # noqa: E402
    ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES,
    ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES,
    add_memory_and_potential_features,
    build_ability_model_data,
)


REPORTS_DIR = PROJECT_ROOT / "reports" / "prospect_model"
PROSPECT_AGE_LIMIT = 27
ELITE_VALUE_THRESHOLD = 50_000_000
FUTURE_WINDOW_DAYS = 730
MIN_MINUTES_FOR_TRAINING = 300
MIN_MINUTES_FOR_CURRENT_SCORING = 1

PROSPECT_NUMERIC_FEATURES = ABILITY_YOUTH_CONTEXT_NUMERIC_FEATURES
PROSPECT_CATEGORICAL_FEATURES = ABILITY_MARKET_CONTEXT_CATEGORICAL_FEATURES
PROSPECT_FEATURES = PROSPECT_NUMERIC_FEATURES + PROSPECT_CATEGORICAL_FEATURES


def make_prospect_classifier() -> Pipeline:
    """Build the first prospect classifier."""
    return Pipeline(steps=[
        (
            "preprocessor",
            make_preprocessor(
                PROSPECT_NUMERIC_FEATURES,
                PROSPECT_CATEGORICAL_FEATURES,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def add_future_elite_labels(
    data: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    """Label whether a player is/reaches a high valuation within two years."""
    result = data.copy()
    lookup = result[
        ["player_id", "season", "valuation_date", TARGET]
    ].copy()
    lookup["valuation_date"] = lookup["valuation_date"].fillna(
        result["season_end_date"]
    )

    candidates = lookup.merge(
        valuations[["player_id", "date", "market_value_in_eur"]],
        on="player_id",
        how="left",
        suffixes=("", "_future"),
    )
    candidates["days_after_valuation"] = (
        candidates["date"] - candidates["valuation_date"]
    ).dt.days
    candidates = candidates.loc[
        candidates["days_after_valuation"].between(0, FUTURE_WINDOW_DAYS)
    ].copy()

    future_peaks = (
        candidates.groupby(["player_id", "season"], as_index=False)
        .agg(future_peak_2yr_value=("market_value_in_eur_future", "max"))
    )
    result = result.merge(
        future_peaks,
        on=["player_id", "season"],
        how="left",
    )
    result["future_peak_2yr_value"] = result["future_peak_2yr_value"].fillna(
        result[TARGET]
    )
    result["future_elite_player"] = (
        result["future_peak_2yr_value"].ge(ELITE_VALUE_THRESHOLD)
    ).astype(int)
    result["future_value_gain_m"] = (
        result["future_peak_2yr_value"] - result[TARGET]
    ) / 1_000_000
    return result


def prospect_training_pool(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows where a potential model has a sensible job to do."""
    return data.loc[
        data["age_at_season_end"].lt(PROSPECT_AGE_LIMIT)
        & data["minutes_played"].ge(MIN_MINUTES_FOR_TRAINING)
        & data["future_peak_2yr_value"].notna()
    ].copy()


def evaluate_fold(
    data: pd.DataFrame,
    validation_season: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train on previous seasons and evaluate one season."""
    train = data.loc[data["season"].between(2016, validation_season - 1)].copy()
    validation = data.loc[data["season"].eq(validation_season)].copy()

    model = make_prospect_classifier()
    model.fit(train[PROSPECT_FEATURES], train["future_elite_player"])
    probabilities = model.predict_proba(validation[PROSPECT_FEATURES])[:, 1]

    predictions = validation[
        [
            "season",
            "player_id",
            "player_name",
            "position",
            "sub_position",
            "age_at_season_end",
            "minutes_played",
            TARGET,
            "future_peak_2yr_value",
            "future_value_gain_m",
            "future_elite_player",
        ]
    ].copy()
    predictions["prospect_probability"] = probabilities

    top_10 = predictions.nlargest(10, "prospect_probability")
    top_20 = predictions.nlargest(20, "prospect_probability")
    metrics = {
        "validation_season": validation_season,
        "rows": len(validation),
        "positive_rate": validation["future_elite_player"].mean(),
        "roc_auc": roc_auc_score(validation["future_elite_player"], probabilities),
        "average_precision": average_precision_score(
            validation["future_elite_player"],
            probabilities,
        ),
        "precision_at_10": top_10["future_elite_player"].mean(),
        "precision_at_20": top_20["future_elite_player"].mean(),
    }
    return metrics, predictions


def walk_forward_evaluate_prospects(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the prospect classifier with season-respecting folds."""
    metric_rows = []
    prediction_tables = []

    for validation_season in range(2019, 2024):
        metrics, predictions = evaluate_fold(data, validation_season)
        metric_rows.append(metrics)
        prediction_tables.append(predictions)

    return pd.DataFrame(metric_rows), pd.concat(prediction_tables, ignore_index=True)


def build_scoring_model_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Build the same feature frame for the current 2025/26 scoring season."""
    scoring = pd.read_csv(
        INTERIM_DIR / "scoring_2025_26_with_pl_stats.csv.gz",
        parse_dates=["season_end_date", "date_of_birth"],
    )
    scoring = scoring.loc[scoring["has_premierleague_stats"]].copy()
    scoring = add_engineered_features(scoring)

    valuations = load_valuations()
    scoring = add_preseason_market_values(scoring, valuations)
    scoring = add_latest_history_to_scoring(scoring, history, history_season=2024)

    appearances = load_appearances()
    games = load_games()
    competitions = load_competitions()
    appearance_history = build_weighted_appearance_history(
        appearances,
        games,
        competitions,
    )
    scoring = add_latest_weighted_history_to_scoring(
        scoring,
        appearance_history,
        history_season=2024,
    )

    team_context = build_team_season_context(games)
    player_season_clubs = build_player_season_clubs(appearances, games)
    scoring = add_team_context_features(
        scoring,
        player_season_clubs,
        team_context,
    )
    scoring = attach_current_player_values(scoring, load_players())
    scoring["contract_expiration_date"] = pd.to_datetime(
        scoring["contract_expiration_date"],
        errors="coerce",
    )
    scoring["current_contract_years_remaining"] = (
        scoring["contract_expiration_date"] - scoring["season_end_date"]
    ).dt.days / 365.25
    scoring[TARGET] = scoring["current_market_value_in_eur"]
    scoring["valuation_date"] = pd.NaT
    scoring["days_after_season"] = np.nan
    return scoring


def score_current_prospects(
    historical_data: pd.DataFrame,
    prospect_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all labelled history and score current young players."""
    scoring = build_scoring_model_frame(historical_data)
    combined = pd.concat([historical_data, scoring], ignore_index=True, sort=False)
    combined = add_memory_and_potential_features(combined)
    current = combined.loc[combined["season"].eq(2025)].copy()
    current_pool = current.loc[
        current["age_at_season_end"].lt(PROSPECT_AGE_LIMIT)
        & current["minutes_played"].ge(MIN_MINUTES_FOR_CURRENT_SCORING)
        & current["current_market_value_in_eur"].notna()
    ].copy()

    model = make_prospect_classifier()
    model.fit(prospect_data[PROSPECT_FEATURES], prospect_data["future_elite_player"])
    probabilities = model.predict_proba(current_pool[PROSPECT_FEATURES])[:, 1]

    current_predictions = current_pool[
        [
            "season",
            "player_id",
            "player_name",
            "season_club_id",
            "current_club_name",
            "position",
            "sub_position",
            "age_at_season_end",
            "minutes_played",
            "current_market_value_in_eur",
            "previous_known_market_value_in_eur",
            "recent_3yr_max_value_m",
            "team_final_position",
            "team_top_4",
            "team_won_league",
            "player_minutes_share",
        ]
    ].copy()
    current_predictions["prospect_probability"] = probabilities
    current_predictions["current_market_value_m"] = (
        current_predictions["current_market_value_in_eur"] / 1_000_000
    )

    feature_importances = (
        pd.DataFrame({
            "feature": model.named_steps["preprocessor"]
            .get_feature_names_out(PROSPECT_FEATURES),
            "importance": model.named_steps["classifier"].feature_importances_,
        })
        .assign(
            feature=lambda table: table["feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return current_predictions, feature_importances


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    historical_data = build_ability_model_data()
    labelled = add_future_elite_labels(historical_data, load_valuations())
    prospect_data = prospect_training_pool(labelled)
    cv_metrics, cv_predictions = walk_forward_evaluate_prospects(prospect_data)
    current_predictions, feature_importances = score_current_prospects(
        historical_data,
        prospect_data,
    )

    cv_metrics.to_csv(REPORTS_DIR / "walk_forward_cv.csv", index=False)
    cv_predictions.to_csv(REPORTS_DIR / "walk_forward_predictions.csv", index=False)
    current_predictions.sort_values(
        "prospect_probability",
        ascending=False,
    ).to_csv(REPORTS_DIR / "current_2025_26_prospect_predictions.csv", index=False)
    feature_importances.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    watchlist_names = [
        "Bukayo Saka",
        "Ethan Nwaneri",
        "Kobbie Mainoo",
        "Leny Yoro",
        "Myles Lewis-Skelly",
        "Adam Wharton",
        "Florian Wirtz",
        "Hugo Ekitiké",
        "Archie Gray",
    ]
    watchlist = current_predictions.loc[
        current_predictions["player_name"].isin(watchlist_names)
    ].sort_values("prospect_probability", ascending=False)
    watchlist.to_csv(REPORTS_DIR / "current_watchlist.csv", index=False)

    print("Prospect model CV metrics")
    print(cv_metrics.to_string(index=False))
    print()
    print("Mean CV metrics")
    print(cv_metrics.drop(columns="validation_season").mean(numeric_only=True).to_string())
    print()
    print("Top feature importances")
    print(feature_importances.head(20).to_string(index=False))
    print()
    print("Top current U27 elite-trajectory players")
    print(
        current_predictions.sort_values("prospect_probability", ascending=False)
        [
            [
                "player_name",
                "current_club_name",
                "position",
                "sub_position",
                "age_at_season_end",
                "minutes_played",
                "current_market_value_m",
                "prospect_probability",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    print()
    print("Watchlist")
    print(
        watchlist[
            [
                "player_name",
                "current_club_name",
                "position",
                "age_at_season_end",
                "minutes_played",
                "current_market_value_m",
                "prospect_probability",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved outputs to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
