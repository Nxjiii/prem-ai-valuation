"""Feature definitions and history feature helpers for valuation models."""

import pandas as pd


TARGET = "market_value_in_eur"

SELECTED_NUMERIC_FEATURES = [
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "clubs_played_for",
    "height_in_cm",
    "age_at_season_end",
    "age_distance_squared",
]

ORIGINAL_NUMERIC_FEATURES = [
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "clubs_played_for",
    "height_in_cm",
    "age_at_season_end",
]

ORIGINAL_CATEGORICAL_FEATURES = ["position", "foot"]
SELECTED_CATEGORICAL_FEATURES = [
    "position",
    "sub_position",
    "foot",
]

SELECTED_FEATURES = SELECTED_NUMERIC_FEATURES + SELECTED_CATEGORICAL_FEATURES
RATE_NUMERIC_FEATURES = SELECTED_NUMERIC_FEATURES + [
    "goals_per_90_stable",
    "assists_per_90_stable",
]

PREVIOUS_FEATURES = [
    "previous_minutes_played",
    "previous_goals",
    "previous_assists",
    "previous_appearances",
    "previous_market_value_in_eur",
]

HISTORY_NUMERIC_FEATURES = SELECTED_NUMERIC_FEATURES + PREVIOUS_FEATURES
HISTORY_FEATURES = HISTORY_NUMERIC_FEATURES + SELECTED_CATEGORICAL_FEATURES


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add model features derived from existing player-season columns."""
    result = data.copy()
    result["age_distance_squared"] = (
        result["age_at_season_end"] - 25
    ) ** 2
    stable_minutes = result["minutes_played"].clip(lower=450)
    result["goals_per_90_stable"] = (
        result["goals"] / stable_minutes * 90
    )
    result["assists_per_90_stable"] = (
        result["assists"] / stable_minutes * 90
    )
    return result


def add_previous_season_features(data: pd.DataFrame) -> pd.DataFrame:
    """Attach previous Premier League season stats and valuation by player."""
    history_columns = [
        "player_id",
        "season",
        "minutes_played",
        "goals",
        "assists",
        "appearances",
        "market_value_in_eur",
    ]
    player_history = data[history_columns].sort_values(
        ["player_id", "season"]
    ).copy()

    for column in [
        "minutes_played",
        "goals",
        "assists",
        "appearances",
        "market_value_in_eur",
    ]:
        player_history[f"previous_{column}"] = (
            player_history.groupby("player_id")[column].shift(1)
        )

    history_features = player_history[
        ["player_id", "season", *PREVIOUS_FEATURES]
    ]

    return data.merge(
        history_features,
        on=["player_id", "season"],
        how="left",
    )


def add_latest_history_to_scoring(
    scoring_data: pd.DataFrame,
    model_data: pd.DataFrame,
    history_season: int,
) -> pd.DataFrame:
    """Attach the latest available historical features to scoring rows."""
    latest_history = model_data.loc[
        model_data["season"].eq(history_season),
        [
            "player_id",
            "minutes_played",
            "goals",
            "assists",
            "appearances",
            "market_value_in_eur",
        ],
    ].rename(columns={
        "minutes_played": "previous_minutes_played",
        "goals": "previous_goals",
        "assists": "previous_assists",
        "appearances": "previous_appearances",
        "market_value_in_eur": "previous_market_value_in_eur",
    })

    return scoring_data.merge(
        latest_history,
        on="player_id",
        how="left",
    )
