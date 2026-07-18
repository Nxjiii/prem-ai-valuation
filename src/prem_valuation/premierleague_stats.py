"""Official Premier League stats loading and merge helpers."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


PL_STATS_COLUMNS = [
    "accurate_cross",
    "accurate_long_balls",
    "accurate_pass",
    "accurate_through_ball",
    "aerial_lost",
    "aerial_won",
    "appearances",
    "att_freekick_goal",
    "att_hd_goal",
    "att_lf_goal",
    "att_pen_goal",
    "att_pen_miss",
    "att_rf_goal",
    "big_chance_created",
    "big_chance_missed",
    "big_chance_scored",
    "blocked_scoring_att",
    "challenge_lost",
    "clean_sheet",
    "clearance_off_line",
    "dispossessed",
    "duel_lost",
    "duel_won",
    "effective_clearance",
    "error_lead_to_goal",
    "fouls",
    "goal_assist",
    "goal_kicks",
    "goals",
    "goals_conceded",
    "head_clearance",
    "hit_woodwork",
    "interception",
    "interception_won",
    "keeper_throws",
    "last_man_tackle",
    "mins_played",
    "ontarget_scoring_att",
    "outfielder_block",
    "own_goals",
    "penalty_conceded",
    "penalty_save",
    "penalty_won",
    "poss_lost_all",
    "red_card",
    "saves",
    "total_clearance",
    "total_corners_intobox",
    "total_cross",
    "total_fastbreak",
    "total_long_balls",
    "total_offside",
    "total_pass",
    "total_scoring_att",
    "total_tackle",
    "total_through_ball",
    "touches",
    "touches_in_opp_box",
    "won_tackle",
    "yellow_card",
]

PL_ID_COLUMNS = [
    "season",
    "comp_season_id",
    "pl_player_id",
    "pl_player_profile_id",
    "opta_player_id",
    "player_name",
    "position",
    "position_info",
]


def normalise_player_name(value: str) -> str:
    """Create a forgiving player-name key across football data sources."""
    value = str(value).lower()
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    value = (
        value.replace("ø", "o")
        .replace("đ", "d")
        .replace("ð", "d")
        .replace("þ", "th")
        .replace("ı", "i")
        .replace("ł", "l")
    )
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_premierleague_aliases(path: Path) -> pd.DataFrame:
    """Load manual name aliases for official Premier League stats."""
    if not path.exists():
        return pd.DataFrame(
            columns=["player_id", "scoring_player_name", "premierleague_player_name"]
        )
    return pd.read_csv(path)


def prepare_premierleague_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Prepare official PL stats for a season/name-key merge."""
    columns = PL_ID_COLUMNS + PL_STATS_COLUMNS
    prepared = stats[columns].copy()
    prepared["premierleague_join_key"] = prepared["player_name"].map(
        normalise_player_name
    )
    prepared = prepared.rename(
        columns={
            "player_name": "premierleague_player_name",
            "position": "premierleague_position",
            "position_info": "premierleague_position_info",
            **{
                column: f"pl_{column}"
                for column in PL_STATS_COLUMNS
            },
        }
    )
    return prepared


def attach_premierleague_stats(
    player_seasons: pd.DataFrame,
    premierleague_stats: pd.DataFrame,
    aliases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach official Premier League stats to player-season rows."""
    data = player_seasons.copy()
    stats = prepare_premierleague_stats(premierleague_stats)

    data["premierleague_join_key"] = data["player_name"].map(
        normalise_player_name
    )

    if aliases is not None and not aliases.empty:
        alias_lookup = aliases.assign(
            premierleague_join_key=aliases["premierleague_player_name"].map(
                normalise_player_name
            )
        )[["player_id", "premierleague_join_key"]]

        data = data.merge(
            alias_lookup,
            on="player_id",
            how="left",
            suffixes=("", "_alias"),
        )
        data["premierleague_join_key"] = data[
            "premierleague_join_key_alias"
        ].fillna(data["premierleague_join_key"])
        data = data.drop(columns=["premierleague_join_key_alias"])

    enriched = data.merge(
        stats,
        on=["season", "premierleague_join_key"],
        how="left",
        validate="many_to_one",
    )
    enriched["has_premierleague_stats"] = enriched["pl_player_id"].notna()
    return enriched.drop(columns=["premierleague_join_key"])


def add_premierleague_rate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add simple per-90 and percentage features from official PL stats."""
    result = data.copy()
    stable_minutes = result["pl_mins_played"].clip(lower=450)

    per_90_columns = [
        "pl_total_scoring_att",
        "pl_ontarget_scoring_att",
        "pl_big_chance_created",
        "pl_big_chance_scored",
        "pl_big_chance_missed",
        "pl_hit_woodwork",
        "pl_penalty_won",
        "pl_total_fastbreak",
        "pl_touches_in_opp_box",
        "pl_total_pass",
        "pl_accurate_pass",
        "pl_total_long_balls",
        "pl_accurate_long_balls",
        "pl_total_through_ball",
        "pl_accurate_through_ball",
        "pl_total_cross",
        "pl_accurate_cross",
        "pl_total_corners_intobox",
        "pl_touches",
        "pl_poss_lost_all",
        "pl_dispossessed",
        "pl_fouls",
        "pl_total_offside",
        "pl_total_tackle",
        "pl_won_tackle",
        "pl_last_man_tackle",
        "pl_interception",
        "pl_interception_won",
        "pl_total_clearance",
        "pl_effective_clearance",
        "pl_head_clearance",
        "pl_clearance_off_line",
        "pl_outfielder_block",
        "pl_blocked_scoring_att",
        "pl_duel_won",
        "pl_duel_lost",
        "pl_challenge_lost",
        "pl_aerial_won",
        "pl_aerial_lost",
        "pl_penalty_conceded",
        "pl_own_goals",
        "pl_error_lead_to_goal",
        "pl_saves",
        "pl_penalty_save",
        "pl_goals_conceded",
        "pl_clean_sheet",
        "pl_keeper_throws",
        "pl_goal_kicks",
    ]

    for column in per_90_columns:
        if column in result.columns:
            result[f"{column}_per_90"] = result[column] / stable_minutes * 90

    result["pl_pass_completion_rate"] = (
        result["pl_accurate_pass"] / result["pl_total_pass"].replace(0, pd.NA)
    )
    result["pl_cross_completion_rate"] = (
        result["pl_accurate_cross"] / result["pl_total_cross"].replace(0, pd.NA)
    )
    result["pl_tackle_success_rate"] = (
        result["pl_won_tackle"] / result["pl_total_tackle"].replace(0, pd.NA)
    )
    result["pl_duel_win_rate"] = (
        result["pl_duel_won"]
        / (result["pl_duel_won"] + result["pl_duel_lost"]).replace(0, pd.NA)
    )
    result["pl_aerial_win_rate"] = (
        result["pl_aerial_won"]
        / (result["pl_aerial_won"] + result["pl_aerial_lost"]).replace(0, pd.NA)
    )
    result["pl_long_ball_completion_rate"] = (
        result["pl_accurate_long_balls"]
        / result["pl_total_long_balls"].replace(0, pd.NA)
    )
    result["pl_through_ball_completion_rate"] = (
        result["pl_accurate_through_ball"]
        / result["pl_total_through_ball"].replace(0, pd.NA)
    )
    result["pl_save_rate"] = (
        result["pl_saves"]
        / (result["pl_saves"] + result["pl_goals_conceded"]).replace(0, pd.NA)
    )
    return result
