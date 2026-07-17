"""Scoring, ranking, and output helpers for valuation gaps."""

from pathlib import Path

import numpy as np
import pandas as pd


RANKING_COLUMNS = [
    "player_name",
    "current_club_name",
    "position",
    "sub_position",
    "age_at_season_end",
    "minutes_played",
    "current_market_value_in_eur",
    "predicted_value",
    "valuation_gap",
    "has_previous_market_value",
]

HIGH_CONFIDENCE_COLUMNS = [
    "player_name",
    "current_club_name",
    "position",
    "sub_position",
    "age_at_season_end",
    "minutes_played",
    "previous_market_value_in_eur",
    "current_market_value_in_eur",
    "predicted_value",
    "valuation_gap",
]


def attach_current_player_values(
    scoring_data: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Attach current Transfermarkt values and current club metadata."""
    current_player_values = players[
        [
            "player_id",
            "market_value_in_eur",
            "current_club_name",
            "current_club_domestic_competition_id",
        ]
    ].rename(columns={
        "market_value_in_eur": "current_market_value_in_eur",
    })

    columns_to_replace = [
        "current_market_value_in_eur",
        "current_club_name",
        "current_club_domestic_competition_id",
    ]

    return scoring_data.drop(
        columns=[
            column for column in columns_to_replace
            if column in scoring_data.columns
        ]
    ).merge(
        current_player_values,
        on="player_id",
        how="left",
    )


def build_scoring_results(
    scoring_data: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create the scored player table used for valuation-gap rankings."""
    scoring_results = scoring_data[
        [
            "season",
            "player_id",
            "player_name",
            "current_club_name",
            "current_club_domestic_competition_id",
            "position",
            "sub_position",
            "age_at_season_end",
            "minutes_played",
            "previous_minutes_played",
            "current_market_value_in_eur",
            "previous_market_value_in_eur",
        ]
    ].copy()

    scoring_results["predicted_value"] = predictions
    scoring_results["valuation_gap"] = (
        scoring_results["predicted_value"]
        - scoring_results["current_market_value_in_eur"]
    )
    scoring_results["absolute_gap"] = scoring_results["valuation_gap"].abs()
    scoring_results["has_previous_market_value"] = (
        scoring_results["previous_market_value_in_eur"].notna()
    )
    scoring_results["minutes_band"] = pd.cut(
        scoring_results["minutes_played"],
        bins=[-1, 449, 899, np.inf],
        labels=["low_minutes", "medium_minutes", "regular_minutes"],
    )
    return scoring_results


def make_ranking_tables(
    scoring_results: pd.DataFrame,
    *,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """Build the standard 2025/26 valuation ranking outputs."""
    ranking_pool = scoring_results.loc[
        scoring_results["current_market_value_in_eur"].notna()
        & scoring_results["current_club_domestic_competition_id"].eq("GB1")
    ].copy()
    regular_minutes = ranking_pool["minutes_band"].eq("regular_minutes")

    undervalued_players = (
        ranking_pool.loc[regular_minutes]
        .sort_values("valuation_gap", ascending=False)
        [RANKING_COLUMNS]
        .head(top_n)
    )
    overvalued_players = (
        ranking_pool.loc[regular_minutes]
        .sort_values("valuation_gap", ascending=True)
        [RANKING_COLUMNS]
        .head(top_n)
    )

    high_confidence_pool = ranking_pool.loc[
        regular_minutes & ranking_pool["has_previous_market_value"]
    ].copy()
    high_confidence_undervalued = (
        high_confidence_pool
        .sort_values("valuation_gap", ascending=False)
        [HIGH_CONFIDENCE_COLUMNS]
        .head(top_n)
    )
    recruitment_bargains = (
        high_confidence_pool.loc[
            high_confidence_pool["age_at_season_end"].lt(27)
            & high_confidence_pool["current_market_value_in_eur"].le(50_000_000)
        ]
        .sort_values("valuation_gap", ascending=False)
        [HIGH_CONFIDENCE_COLUMNS]
        .head(top_n)
    )

    return {
        "scoring_results_2025_26": scoring_results,
        "undervalued_players_2025_26": undervalued_players,
        "overvalued_players_2025_26": overvalued_players,
        "high_confidence_undervalued_2025_26": high_confidence_undervalued,
        "recruitment_bargains_2025_26": recruitment_bargains,
    }


def save_ranking_outputs(
    ranking_outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Save ranking outputs as CSV files and return a small manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for name, output_table in ranking_outputs.items():
        output_path = output_dir / f"{name}.csv"
        output_table.to_csv(output_path, index=False)
        saved_files.append({
            "file": str(output_path),
            "rows": len(output_table),
            "columns": output_table.shape[1],
        })

    return pd.DataFrame(saved_files)
