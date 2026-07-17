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

PRESEASON_MARKET_VALUE_FEATURES = [
    "previous_known_market_value_in_eur",
]

WEIGHTED_HISTORY_FEATURES = [
    "previous_all_minutes",
    "previous_all_goals",
    "previous_all_assists",
    "previous_weighted_minutes",
    "previous_weighted_goals",
    "previous_weighted_assists",
    "previous_domestic_league_minutes",
    "previous_domestic_league_goals",
    "previous_domestic_league_assists",
    "previous_europe_minutes",
    "previous_europe_goals",
    "previous_europe_assists",
]

HISTORY_NUMERIC_FEATURES = (
    SELECTED_NUMERIC_FEATURES
    + PREVIOUS_FEATURES
    + PRESEASON_MARKET_VALUE_FEATURES
)
HISTORY_FEATURES = HISTORY_NUMERIC_FEATURES + SELECTED_CATEGORICAL_FEATURES
WEIGHTED_HISTORY_NUMERIC_FEATURES = (
    HISTORY_NUMERIC_FEATURES + WEIGHTED_HISTORY_FEATURES
)
WEIGHTED_HISTORY_FEATURES_ALL = (
    WEIGHTED_HISTORY_NUMERIC_FEATURES + SELECTED_CATEGORICAL_FEATURES
)


COMPETITION_WEIGHTS = {
    "GB1": 1.00,
    "ES1": 0.95,
    "L1": 0.90,
    "IT1": 0.90,
    "FR1": 0.85,
    "CL": 1.00,
    "EL": 0.80,
    "UCOL": 0.70,
    "PO1": 0.70,
    "NL1": 0.70,
    "BE1": 0.60,
    "TR1": 0.60,
    "SC1": 0.55,
    "SA1": 0.50,
    "FAC": 0.50,
    "CGB": 0.50,
}


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


def add_preseason_market_values(
    data: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    cutoff_month: int = 7,
    cutoff_day: int = 1,
) -> pd.DataFrame:
    """Attach latest known valuation before the season starts."""
    result = data.copy()
    result["season_start_cutoff"] = pd.to_datetime(
        result["season"].astype(str)
        + f"-{cutoff_month:02d}-{cutoff_day:02d}"
    )

    lookup_keys = result[["player_id", "season", "season_start_cutoff"]]
    valuation_lookup = valuations[
        ["player_id", "date", "market_value_in_eur"]
    ].copy()

    candidates = lookup_keys.merge(
        valuation_lookup,
        on="player_id",
        how="left",
    )
    candidates = candidates.loc[
        candidates["date"].lt(candidates["season_start_cutoff"])
    ].copy()

    latest_values = (
        candidates.sort_values("date")
        .drop_duplicates(["player_id", "season"], keep="last")
        [["player_id", "season", "market_value_in_eur"]]
        .rename(columns={
            "market_value_in_eur": "previous_known_market_value_in_eur",
        })
    )

    result = result.merge(
        latest_values,
        on=["player_id", "season"],
        how="left",
    )
    return result.drop(columns="season_start_cutoff")


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


def build_weighted_appearance_history(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    competitions: pd.DataFrame,
    *,
    competition_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate all-competition player history with simple league weights."""
    weights = competition_weights or COMPETITION_WEIGHTS
    game_lookup = games[
        ["game_id", "season", "competition_id", "competition_type"]
    ].rename(columns={
        "competition_id": "game_competition_id",
        "competition_type": "game_competition_type",
    })

    merged = appearances.merge(
        game_lookup,
        on="game_id",
        how="left",
        validate="many_to_one",
    )
    merged["competition_weight"] = (
        merged["game_competition_id"].map(weights).fillna(0.40)
    )
    merged["weighted_minutes"] = (
        merged["minutes_played"] * merged["competition_weight"]
    )
    merged["weighted_goals"] = (
        merged["goals"] * merged["competition_weight"]
    )
    merged["weighted_assists"] = (
        merged["assists"] * merged["competition_weight"]
    )

    competition_lookup = competitions[
        ["competition_id", "type", "sub_type"]
    ].rename(columns={
        "competition_id": "game_competition_id",
        "type": "competition_type",
        "sub_type": "competition_sub_type",
    })
    merged = merged.merge(
        competition_lookup,
        on="game_competition_id",
        how="left",
        validate="many_to_one",
    )

    merged["is_domestic_league"] = merged["competition_type"].eq(
        "domestic_league"
    )
    merged["is_europe"] = merged["game_competition_id"].isin(
        ["CL", "EL", "UCOL", "CLQ", "ELQ", "ECLQ"]
    )

    for prefix, mask in {
        "domestic_league": merged["is_domestic_league"],
        "europe": merged["is_europe"],
    }.items():
        merged[f"{prefix}_minutes"] = merged["minutes_played"].where(mask, 0)
        merged[f"{prefix}_goals"] = merged["goals"].where(mask, 0)
        merged[f"{prefix}_assists"] = merged["assists"].where(mask, 0)

    return (
        merged.groupby(["player_id", "season"], as_index=False)
        .agg(
            all_appearances=("appearance_id", "count"),
            all_minutes=("minutes_played", "sum"),
            all_goals=("goals", "sum"),
            all_assists=("assists", "sum"),
            weighted_minutes=("weighted_minutes", "sum"),
            weighted_goals=("weighted_goals", "sum"),
            weighted_assists=("weighted_assists", "sum"),
            domestic_league_minutes=("domestic_league_minutes", "sum"),
            domestic_league_goals=("domestic_league_goals", "sum"),
            domestic_league_assists=("domestic_league_assists", "sum"),
            europe_minutes=("europe_minutes", "sum"),
            europe_goals=("europe_goals", "sum"),
            europe_assists=("europe_assists", "sum"),
        )
    )


def add_weighted_previous_history(
    data: pd.DataFrame,
    appearance_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach previous-season all-competition weighted history to rows."""
    previous_history = appearance_history.copy().sort_values(
        ["player_id", "season"]
    )
    history_columns = [
        "all_minutes",
        "all_goals",
        "all_assists",
        "weighted_minutes",
        "weighted_goals",
        "weighted_assists",
        "domestic_league_minutes",
        "domestic_league_goals",
        "domestic_league_assists",
        "europe_minutes",
        "europe_goals",
        "europe_assists",
    ]

    for column in history_columns:
        previous_history[f"previous_{column}"] = (
            previous_history.groupby("player_id")[column].shift(1)
        )

    previous_columns = [
        "player_id",
        "season",
        *[f"previous_{column}" for column in history_columns],
    ]

    return data.merge(
        previous_history[previous_columns],
        on=["player_id", "season"],
        how="left",
    )


def add_latest_weighted_history_to_scoring(
    scoring_data: pd.DataFrame,
    appearance_history: pd.DataFrame,
    history_season: int,
) -> pd.DataFrame:
    """Attach latest all-competition weighted history to scoring rows."""
    latest_history = appearance_history.loc[
        appearance_history["season"].eq(history_season),
        [
            "player_id",
            "all_minutes",
            "all_goals",
            "all_assists",
            "weighted_minutes",
            "weighted_goals",
            "weighted_assists",
            "domestic_league_minutes",
            "domestic_league_goals",
            "domestic_league_assists",
            "europe_minutes",
            "europe_goals",
            "europe_assists",
        ],
    ].rename(columns={
        "all_minutes": "previous_all_minutes",
        "all_goals": "previous_all_goals",
        "all_assists": "previous_all_assists",
        "weighted_minutes": "previous_weighted_minutes",
        "weighted_goals": "previous_weighted_goals",
        "weighted_assists": "previous_weighted_assists",
        "domestic_league_minutes": "previous_domestic_league_minutes",
        "domestic_league_goals": "previous_domestic_league_goals",
        "domestic_league_assists": "previous_domestic_league_assists",
        "europe_minutes": "previous_europe_minutes",
        "europe_goals": "previous_europe_goals",
        "europe_assists": "previous_europe_assists",
    })

    return scoring_data.merge(
        latest_history,
        on="player_id",
        how="left",
    )
