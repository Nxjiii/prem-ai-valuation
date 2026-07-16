"""Build reproducible player-season datasets from the raw source tables."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load the raw tables needed by the player-season pipeline."""
    return {
        "players": pd.read_csv(RAW_DIR / "players.csv.gz"),
        "valuations": pd.read_csv(
            RAW_DIR / "player_valuations.csv.gz", parse_dates=["date"]
        ),
        "appearances": pd.read_csv(RAW_DIR / "appearances.csv.gz"),
        "games": pd.read_csv(RAW_DIR / "games.csv.gz", parse_dates=["date"]),
    }


def build_player_season_stats(
    appearances: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Premier League match appearances to player-season rows."""
    game_lookup = games[
        ["game_id", "season", "date", "competition_id"]
    ].rename(
        columns={
            "date": "game_date",
            "competition_id": "game_competition_id",
        }
    )

    merged = appearances.merge(
        game_lookup,
        on="game_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError("Some appearances do not match a game")
    if not merged["date"].eq(merged["game_date"]).all():
        raise ValueError("Appearance dates disagree with game dates")
    if not merged["competition_id"].eq(
        merged["game_competition_id"]
    ).all():
        raise ValueError("Appearance competitions disagree with games")

    pl_appearances = merged.loc[
        merged["competition_id"].eq("GB1")
    ].copy()

    player_season_stats = (
        pl_appearances.groupby(
            ["season", "player_id", "player_name"], as_index=False
        ).agg(
            appearances=("appearance_id", "count"),
            minutes_played=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            clubs_played_for=("player_club_id", "nunique"),
        )
    )

    pl_games = games.loc[games["competition_id"].eq("GB1")].copy()
    return player_season_stats, pl_games


def attach_postseason_valuations(
    player_season_stats: pd.DataFrame,
    pl_games: pd.DataFrame,
    valuations: pd.DataFrame,
    max_days_after_season: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each historical player-season to its first timely later valuation."""
    season_end_dates = pl_games.groupby("season", as_index=False).agg(
        season_end_date=("date", "max")
    )
    player_seasons = player_season_stats.merge(
        season_end_dates,
        on="season",
        how="left",
        validate="many_to_one",
    )

    valuation_lookup = valuations[
        ["player_id", "date", "market_value_in_eur"]
    ].rename(columns={"date": "valuation_date"})
    candidates = player_seasons.merge(
        valuation_lookup,
        on="player_id",
        how="left",
        validate="many_to_many",
    )
    candidates["days_after_season"] = (
        candidates["valuation_date"] - candidates["season_end_date"]
    ).dt.days

    labelled = (
        candidates.loc[candidates["days_after_season"].ge(0)]
        .sort_values("valuation_date")
        .drop_duplicates(["season", "player_id"], keep="first")
    )
    labelled = labelled.loc[
        labelled["days_after_season"].le(max_days_after_season)
    ].copy()

    scoring = player_seasons.loc[player_seasons["season"].eq(2025)].copy()
    return labelled, scoring


def build_player_attributes(players: pd.DataFrame) -> pd.DataFrame:
    """Select stable attributes and apply documented reference corrections."""
    attributes = players[
        [
            "player_id",
            "date_of_birth",
            "position",
            "sub_position",
            "foot",
            "height_in_cm",
            "country_of_citizenship",
        ]
    ].copy()
    attributes["date_of_birth"] = pd.to_datetime(
        attributes["date_of_birth"]
    )

    corrections = pd.read_csv(
        REFERENCE_DIR / "player_corrections.csv",
        parse_dates=["corrected_date_of_birth"],
    )
    correction_map = corrections.set_index("player_id")[
        "corrected_date_of_birth"
    ]
    attributes["date_of_birth"] = attributes["date_of_birth"].fillna(
        attributes["player_id"].map(correction_map)
    )
    return attributes


def add_player_attributes(
    player_seasons: pd.DataFrame, attributes: pd.DataFrame
) -> pd.DataFrame:
    """Add player attributes and calculate age at the historical cutoff."""
    enriched = player_seasons.merge(
        attributes,
        on="player_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not enriched["_merge"].eq("both").all():
        raise ValueError("Some player-season rows do not match a player profile")

    enriched = enriched.drop(columns="_merge")
    enriched["age_at_season_end"] = (
        enriched["season_end_date"] - enriched["date_of_birth"]
    ).dt.days / 365.25
    return enriched


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build labelled historical data and the unlabelled 2025/26 scoring set."""
    tables = load_raw_tables()
    stats, pl_games = build_player_season_stats(
        tables["appearances"], tables["games"]
    )
    labelled, scoring = attach_postseason_valuations(
        stats, pl_games, tables["valuations"]
    )
    attributes = build_player_attributes(tables["players"])
    return (
        add_player_attributes(labelled, attributes),
        add_player_attributes(scoring, attributes),
    )


def save_datasets(
    labelled: pd.DataFrame, scoring: pd.DataFrame
) -> tuple[Path, Path]:
    """Build and save interim datasets without modifying raw inputs."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    labelled_path = INTERIM_DIR / "labelled_player_seasons.csv.gz"
    scoring_path = INTERIM_DIR / "scoring_2025_26.csv.gz"
    labelled.to_csv(labelled_path, index=False)
    scoring.to_csv(scoring_path, index=False)
    return labelled_path, scoring_path
