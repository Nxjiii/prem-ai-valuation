"""Create readable explanation cards for 2025/26 player valuations.

This is intentionally a practical first explainability layer, not a full SHAP
decomposition. It turns the model output into reason codes by combining:

- current/model valuation gap
- player context features
- position-relative percentiles
- guardrails/floors applied by the combined valuation model
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.data import PROCESSED_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports" / "explanations"
SCORING_PATH = PROCESSED_DIR / "scoring_results_2025_26.csv"

GUARDRAIL_COLUMNS = {
    "elite_trajectory_floor_value": "elite trajectory floor",
    "elite_market_sanity_floor_value": "elite market sanity floor",
    "established_elite_status_floor_value": "established elite status floor",
    "ultra_elite_current_value_floor": "ultra-elite current value floor",
    "elite_non_attacking_role_floor_value": "elite non-attacking role floor",
    "new_high_value_signing_floor_value": "new high-value signing floor",
}

SIGNAL_FEATURES_BY_POSITION = {
    "Attack": [
        ("goals", "goals", "high"),
        ("assists", "assists", "high"),
        ("previous_weighted_goals", "recent goal history", "high"),
        ("previous_weighted_assists", "recent assist history", "high"),
        ("pl_total_scoring_att_per_90", "shot volume", "high"),
        ("pl_ontarget_scoring_att_per_90", "shots on target", "high"),
        ("pl_big_chance_created_per_90", "chance creation", "high"),
        ("pl_touches_in_opp_box_per_90", "box threat", "high"),
        ("pl_duel_win_rate", "duel success", "high"),
        ("pl_dispossessed_per_90", "being dispossessed", "low"),
    ],
    "Midfield": [
        ("goals", "goals", "high"),
        ("assists", "assists", "high"),
        ("previous_weighted_assists", "recent assist history", "high"),
        ("pl_big_chance_created_per_90", "chance creation", "high"),
        ("pl_pass_completion_rate", "pass completion", "high"),
        ("pl_total_long_balls_per_90", "long passing volume", "high"),
        ("pl_total_through_ball_per_90", "through-ball volume", "high"),
        ("pl_total_tackle_per_90", "tackling volume", "high"),
        ("pl_interception_per_90", "interceptions", "high"),
        ("pl_duel_win_rate", "duel success", "high"),
        ("pl_poss_lost_all_per_90", "possession losses", "low"),
    ],
    "Defender": [
        ("pl_tackle_success_rate", "tackle success", "high"),
        ("pl_interception_per_90", "interceptions", "high"),
        ("pl_total_clearance_per_90", "clearance volume", "high"),
        ("pl_effective_clearance_per_90", "effective clearances", "high"),
        ("pl_outfielder_block_per_90", "blocks", "high"),
        ("pl_aerial_win_rate", "aerial success", "high"),
        ("pl_duel_win_rate", "duel success", "high"),
        ("pl_error_lead_to_goal_per_90", "errors leading to goals", "low"),
        ("pl_penalty_conceded_per_90", "penalties conceded", "low"),
        ("pl_own_goals_per_90", "own goals", "low"),
    ],
    "Goalkeeper": [
        ("pl_saves_per_90", "saves", "high"),
        ("pl_save_rate", "save rate", "high"),
        ("pl_clean_sheet_per_90", "clean sheets", "high"),
        ("pl_goals_conceded_per_90", "goals conceded", "low"),
        ("pl_error_lead_to_goal_per_90", "errors leading to goals", "low"),
        ("pl_pass_completion_rate", "pass completion", "high"),
    ],
}


def euros(value: float | int | None) -> str:
    """Format a value as a compact euro amount."""
    if value is None or pd.isna(value):
        return "unknown"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000:
        return f"{sign}€{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{sign}€{value / 1_000:.0f}k"
    return f"{sign}€{value:.0f}"


def as_percentile(value: float | int | None) -> str:
    """Format percentile values from 0-1."""
    if value is None or pd.isna(value):
        return "unknown percentile"
    return f"{float(value) * 100:.0f}th percentile"


def normalise_name(value: str) -> str:
    """Normalise names for forgiving lookups like 'saka' or 'ekitike'."""
    normalised = unicodedata.normalize("NFKD", str(value))
    ascii_name = "".join(
        character for character in normalised
        if not unicodedata.combining(character)
    )
    return " ".join(ascii_name.lower().split())


def slugify(value: str) -> str:
    """Create a simple filename-safe slug."""
    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in normalise_name(value)
    )
    return "_".join(part for part in cleaned.split("_") if part)


def add_position_percentiles(scoring: pd.DataFrame) -> pd.DataFrame:
    """Add percentile columns for signal features within each broad position."""
    result = scoring.copy()
    seen_features = {
        feature
        for feature_group in SIGNAL_FEATURES_BY_POSITION.values()
        for feature, _, _ in feature_group
    }
    seen_features.update([
        "minutes_played",
        "player_minutes_share",
        "previous_known_market_value_in_eur",
        "previous_market_value_in_eur",
    ])

    for feature in sorted(seen_features):
        if feature not in result.columns:
            continue
        percentile_column = f"{feature}_position_percentile"
        result[percentile_column] = (
            result.groupby("position", observed=False)[feature]
            .rank(pct=True, na_option="keep")
        )
    return result


def add_signal(signals: list[tuple[float, str]], strength: float, text: str) -> None:
    """Collect signal text with a sortable strength."""
    if text:
        signals.append((float(strength), text))


def player_guardrails(row: pd.Series) -> list[str]:
    """Describe active valuation floors/guardrails."""
    guardrails = []
    football_value = row.get("football_ability_value", np.nan)
    predicted_value = row.get("predicted_value", np.nan)

    for column, label in GUARDRAIL_COLUMNS.items():
        value = row.get(column, 0)
        if pd.notna(value) and value > 0:
            guardrails.append(f"{label}: {euros(value)}")

    calibration_delta = row.get("calibration_adjusted_value_delta", 0)
    if pd.notna(calibration_delta) and abs(calibration_delta) >= 500_000:
        direction = "up" if calibration_delta > 0 else "down"
        guardrails.append(
            f"value-band calibration moved estimate {direction} by {euros(abs(calibration_delta))}"
        )

    if (
        pd.notna(football_value)
        and pd.notna(predicted_value)
        and predicted_value - football_value >= 5_000_000
    ):
        guardrails.append(
            f"final estimate is {euros(predicted_value - football_value)} above pure football-ability value"
        )

    return guardrails


def player_signals(row: pd.Series) -> tuple[list[str], list[str]]:
    """Build positive and negative reason-code signals for one player."""
    positives: list[tuple[float, str]] = []
    negatives: list[tuple[float, str]] = []

    age = row.get("age_at_season_end")
    minutes = row.get("minutes_played")
    minutes_share = row.get("player_minutes_share")
    current_value = row.get("current_market_value_in_eur")
    previous_known_value = row.get("previous_known_market_value_in_eur")
    previous_pl_value = row.get("previous_market_value_in_eur")
    contract_years = row.get("current_contract_years_remaining")
    elite_probability = row.get("elite_trajectory_probability")
    football_value = row.get("football_ability_value")

    if pd.notna(previous_known_value) and previous_known_value >= 75_000_000:
        add_signal(
            positives,
            0.98,
            f"elite market memory: previous known value was {euros(previous_known_value)}",
        )
    elif pd.notna(previous_known_value) and previous_known_value >= 40_000_000:
        add_signal(
            positives,
            0.75,
            f"strong market memory: previous known value was {euros(previous_known_value)}",
        )

    if pd.notna(previous_pl_value) and pd.notna(current_value):
        value_change = current_value - previous_pl_value
        if value_change >= 10_000_000:
            add_signal(
                positives,
                0.68,
                f"Transfermarkt has already moved him up from {euros(previous_pl_value)}",
            )
        elif value_change <= -10_000_000:
            add_signal(
                negatives,
                0.68,
                f"recent market value fell from {euros(previous_pl_value)} to {euros(current_value)}",
            )

    if pd.notna(elite_probability) and elite_probability >= 0.75:
        add_signal(
            positives,
            0.95,
            f"elite trajectory probability is high ({elite_probability:.0%})",
        )
    elif pd.notna(elite_probability) and elite_probability <= 0.20 and pd.notna(age) and age < 25:
        add_signal(
            negatives,
            0.55,
            f"elite trajectory probability is low ({elite_probability:.0%}) for a young player",
        )

    if pd.notna(age):
        if age < 22:
            add_signal(positives, 0.78, f"young age profile ({age:.1f}) supports upside")
        elif age < 26:
            add_signal(positives, 0.62, f"prime-development age profile ({age:.1f})")
        elif age >= 32:
            add_signal(negatives, 0.72, f"older resale age profile ({age:.1f})")

    if pd.notna(minutes):
        if minutes >= 2_500:
            add_signal(positives, 0.82, f"heavy league usage: {minutes:.0f} minutes")
        elif minutes < 900:
            add_signal(
                negatives,
                0.82,
                f"limited sample/current-season minutes: {minutes:.0f}",
            )

    if pd.notna(minutes_share):
        if minutes_share >= 0.70:
            add_signal(positives, 0.76, f"major squad role: {minutes_share:.0%} of team minutes")
        elif minutes_share < 0.25:
            add_signal(negatives, 0.65, f"small squad role: {minutes_share:.0%} of team minutes")

    if row.get("team_won_league", 0) == 1 and pd.notna(minutes) and minutes >= 1_200:
        add_signal(positives, 0.82, "meaningful minutes in a title-winning side")
    elif row.get("team_top_4", 0) == 1 and pd.notna(minutes) and minutes >= 1_800:
        add_signal(positives, 0.70, "regular minutes in a top-four side")

    if pd.notna(contract_years):
        if contract_years >= 4:
            add_signal(positives, 0.64, f"long contract runway ({contract_years:.1f} years)")
        elif contract_years <= 1.5:
            add_signal(negatives, 0.56, f"shorter contract runway ({contract_years:.1f} years)")

    for feature, label, good_direction in SIGNAL_FEATURES_BY_POSITION.get(
        row.get("position"),
        [],
    ):
        if feature not in row.index:
            continue
        percentile = row.get(f"{feature}_position_percentile")
        value = row.get(feature)
        if pd.isna(percentile) or pd.isna(value):
            continue

        if good_direction == "high":
            if percentile >= 0.85:
                add_signal(
                    positives,
                    percentile,
                    f"strong {label} for his position ({as_percentile(percentile)})",
                )
            elif percentile <= 0.15:
                add_signal(
                    negatives,
                    1 - percentile,
                    f"low {label} for his position ({as_percentile(percentile)})",
                )
        else:
            if percentile <= 0.15:
                add_signal(
                    positives,
                    1 - percentile,
                    f"low {label} for his position ({as_percentile(percentile)})",
                )
            elif percentile >= 0.85:
                add_signal(
                    negatives,
                    percentile,
                    f"high {label} for his position ({as_percentile(percentile)})",
                )

    if (
        pd.notna(current_value)
        and pd.notna(football_value)
        and current_value >= 50_000_000
        and football_value <= current_value * 0.65
    ):
        add_signal(
            negatives,
            0.90,
            (
                "football-ability estimate is well below current market price "
                f"({euros(football_value)} vs {euros(current_value)})"
            ),
        )
    elif (
        pd.notna(current_value)
        and pd.notna(football_value)
        and current_value >= 50_000_000
        and football_value <= current_value * 0.85
    ):
        add_signal(
            negatives,
            0.70,
            (
                "model likes him, but not enough to fully match current market price "
                f"({euros(football_value)} vs {euros(current_value)})"
            ),
        )

    if (
        pd.notna(current_value)
        and current_value >= 50_000_000
        and pd.isna(previous_pl_value)
        and pd.notna(minutes)
        and minutes < 1_800
    ):
        add_signal(
            negatives,
            0.74,
            "high-value player with limited current PL sample and no previous PL value anchor",
        )

    positives_text = [
        text for _, text in sorted(positives, reverse=True)[:6]
    ]
    negatives_text = [
        text for _, text in sorted(negatives, reverse=True)[:6]
    ]
    return positives_text, negatives_text


def interpretation(row: pd.Series, guardrails: list[str]) -> str:
    """Write a short human-readable interpretation of the valuation."""
    gap = row.get("valuation_gap")
    player = row.get("player_name", "This player")
    football_value = row.get("football_ability_value")
    predicted_value = row.get("predicted_value")

    if pd.isna(gap):
        return f"{player} cannot be compared because current Transfermarkt value is missing."

    if gap >= 10_000_000:
        verdict = "the model sees him as undervalued versus Transfermarkt"
    elif gap <= -10_000_000:
        verdict = "the model sees him as overvalued versus Transfermarkt"
    else:
        verdict = "the model sees him as broadly fairly valued"

    if (
        guardrails
        and pd.notna(football_value)
        and pd.notna(predicted_value)
        and predicted_value - football_value >= 5_000_000
    ):
        return (
            f"{player}: {verdict}. The raw football-ability estimate was "
            f"{euros(football_value)}, but guardrail logic protected the final "
            f"valuation at {euros(predicted_value)}."
        )

    return f"{player}: {verdict}. Final model value is {euros(predicted_value)}."


def build_explanations(scoring: pd.DataFrame) -> pd.DataFrame:
    """Create one explanation row per player."""
    scoring = add_position_percentiles(scoring)
    explanation_rows = []

    for _, row in scoring.iterrows():
        positives, negatives = player_signals(row)
        guardrails = player_guardrails(row)
        explanation_rows.append({
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "season_club_name": row.get("season_club_name"),
            "current_club_name": row.get("current_club_name"),
            "position": row.get("position"),
            "sub_position": row.get("sub_position"),
            "age_at_season_end": row.get("age_at_season_end"),
            "minutes_played": row.get("minutes_played"),
            "current_market_value_in_eur": row.get("current_market_value_in_eur"),
            "football_ability_value": row.get("football_ability_value"),
            "predicted_value": row.get("predicted_value"),
            "valuation_gap": row.get("valuation_gap"),
            "elite_trajectory_probability": row.get("elite_trajectory_probability"),
            "positive_signals": " | ".join(positives),
            "negative_signals": " | ".join(negatives),
            "guardrails_applied": " | ".join(guardrails),
            "interpretation": interpretation(row, guardrails),
        })

    return pd.DataFrame(explanation_rows)


def player_card(row: pd.Series) -> str:
    """Render one Markdown valuation card."""
    positives = [
        signal for signal in str(row.get("positive_signals", "")).split(" | ")
        if signal and signal != "nan"
    ]
    negatives = [
        signal for signal in str(row.get("negative_signals", "")).split(" | ")
        if signal and signal != "nan"
    ]
    guardrails = [
        signal for signal in str(row.get("guardrails_applied", "")).split(" | ")
        if signal and signal != "nan"
    ]

    lines = [
        f"## {row['player_name']}",
        "",
        f"- Club: {row.get('season_club_name')} / current: {row.get('current_club_name')}",
        f"- Position: {row.get('position')} — {row.get('sub_position')}",
        f"- Age: {row.get('age_at_season_end'):.1f}",
        f"- Minutes: {row.get('minutes_played'):.0f}",
        f"- Current Transfermarkt value: {euros(row.get('current_market_value_in_eur'))}",
        f"- Pure football-ability value: {euros(row.get('football_ability_value'))}",
        f"- Final model value: {euros(row.get('predicted_value'))}",
        f"- Gap: {euros(row.get('valuation_gap'))}",
        "",
        "Main positive signals:",
    ]

    lines.extend([f"- {signal}" for signal in positives] or ["- None flagged"])
    lines.extend(["", "Main negative signals:"])
    lines.extend([f"- {signal}" for signal in negatives] or ["- None flagged"])
    lines.extend(["", "Guardrails / calibration applied:"])
    lines.extend([f"- {signal}" for signal in guardrails] or ["- None"])
    lines.extend(["", "Interpretation:", "", str(row.get("interpretation")), ""])
    return "\n".join(lines)


def match_player(explanations: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return the rows matching one player-name query."""
    lookup = explanations.copy()
    normalised_query = normalise_name(query)
    lookup["_normalised_player_name"] = lookup["player_name"].map(normalise_name)

    exact_matches = lookup.loc[
        lookup["_normalised_player_name"].eq(normalised_query)
    ]
    if len(exact_matches) == 1:
        return exact_matches.drop(columns="_normalised_player_name")

    surname_matches = lookup.loc[
        lookup["_normalised_player_name"].eq(normalised_query)
        | lookup["_normalised_player_name"].str.endswith(
            f" {normalised_query}",
            na=False,
        )
    ]
    if len(surname_matches) == 1:
        return surname_matches.drop(columns="_normalised_player_name")

    contains_matches = lookup.loc[
        lookup["_normalised_player_name"].str.contains(
            normalised_query,
            regex=False,
            na=False,
        )
    ]
    if len(contains_matches) == 1:
        return contains_matches.drop(columns="_normalised_player_name")

    if contains_matches.empty:
        sample = (
            lookup.loc[
                lookup["_normalised_player_name"].str.startswith(
                    normalised_query[:3],
                    na=False,
                ),
                "player_name",
            ]
            .drop_duplicates()
            .head(10)
            .to_list()
        )
        suggestion_text = (
            "\nPossible close starts:\n- " + "\n- ".join(sample)
            if sample
            else ""
        )
        raise SystemExit(f'No player matched "{query}".{suggestion_text}')

    choices = (
        contains_matches[
            ["player_name", "season_club_name", "current_club_name", "position"]
        ]
        .drop_duplicates()
        .sort_values("player_name")
    )
    raise SystemExit(
        f'Multiple players matched "{query}". Be more specific:\n'
        + choices.to_string(index=False)
    )


def match_team(explanations: pd.DataFrame, query: str) -> tuple[str, pd.DataFrame]:
    """Return the players matching one team-name query."""
    lookup = explanations.copy()
    normalised_query = normalise_name(query)
    lookup["_normalised_season_club_name"] = lookup["season_club_name"].map(
        normalise_name
    )
    lookup["_normalised_current_club_name"] = lookup["current_club_name"].map(
        normalise_name
    )

    club_names = (
        pd.concat([
            lookup["season_club_name"],
            lookup["current_club_name"],
        ])
        .dropna()
        .drop_duplicates()
        .sort_values()
        .to_frame(name="club_name")
    )
    club_names["_normalised_club_name"] = club_names["club_name"].map(normalise_name)

    exact_clubs = club_names.loc[
        club_names["_normalised_club_name"].eq(normalised_query)
    ]
    contains_clubs = club_names.loc[
        club_names["_normalised_club_name"].str.contains(
            normalised_query,
            regex=False,
            na=False,
        )
    ]

    if len(exact_clubs) == 1:
        matched_club = exact_clubs.iloc[0]["club_name"]
    elif len(contains_clubs) == 1:
        matched_club = contains_clubs.iloc[0]["club_name"]
    elif contains_clubs.empty:
        sample = (
            club_names.loc[
                club_names["_normalised_club_name"].str.startswith(
                    normalised_query[:3],
                    na=False,
                ),
                "club_name",
            ]
            .head(10)
            .to_list()
        )
        suggestion_text = (
            "\nPossible close starts:\n- " + "\n- ".join(sample)
            if sample
            else ""
        )
        raise SystemExit(f'No team matched "{query}".{suggestion_text}')
    else:
        choices = contains_clubs[["club_name"]].sort_values("club_name")
        raise SystemExit(
            f'Multiple teams matched "{query}". Be more specific:\n'
            + choices.to_string(index=False)
        )

    normalised_club = normalise_name(matched_club)
    team_players = lookup.loc[
        lookup["_normalised_season_club_name"].eq(normalised_club)
        | lookup["_normalised_current_club_name"].eq(normalised_club)
    ].copy()

    return matched_club, team_players.drop(
        columns=[
            "_normalised_season_club_name",
            "_normalised_current_club_name",
        ],
    )


def sort_team_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Sort team cards in a useful reading order."""
    position_order = {
        "Goalkeeper": 0,
        "Defender": 1,
        "Midfield": 2,
        "Attack": 3,
    }
    result = cards.copy()
    result["_position_order"] = result["position"].map(position_order).fillna(9)
    return (
        result.sort_values(
            ["_position_order", "minutes_played", "absolute_gap"],
            ascending=[True, False, False],
        )
        .drop(columns="_position_order")
    )


def select_cards(
    explanations: pd.DataFrame,
    player_queries: list[str],
    team_queries: list[str],
    extra_top_gaps: int,
) -> pd.DataFrame:
    """Select requested players plus optional biggest valuation gaps."""
    selected_parts = []
    for query in player_queries:
        selected_parts.append(match_player(explanations, query))
    for query in team_queries:
        _, team_players = match_team(explanations, query)
        selected_parts.append(team_players)

    if extra_top_gaps > 0:
        selected_parts.append(
            explanations.sort_values("valuation_gap", ascending=False).head(extra_top_gaps)
        )
        selected_parts.append(
            explanations.sort_values("valuation_gap", ascending=True).head(extra_top_gaps)
        )

    if not selected_parts:
        return explanations.head(0)

    return (
        pd.concat(selected_parts, ignore_index=True)
        .drop_duplicates("player_id")
        .sort_values(["absolute_gap", "player_name"], ascending=[False, True])
        if "absolute_gap" in explanations.columns
        else pd.concat(selected_parts, ignore_index=True).drop_duplicates("player_id")
    )


def save_markdown_cards(
    cards: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "2025/26 Player Valuation Explanation Cards",
) -> None:
    """Save selected player explanation cards as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# {title}",
        "",
        (
            "These are reason-code explanations built from model inputs, "
            "position-relative percentiles, and valuation guardrails. They are "
            "not exact SHAP-style feature attributions."
        ),
        "",
    ]
    for _, row in cards.iterrows():
        body.append(player_card(row))
    output_path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain selected 2025/26 player valuations.",
    )
    parser.add_argument(
        "--player",
        action="append",
        default=[],
        help=(
            "Player name or partial name to explain. Can be used multiple times, "
            'for example: --player saka --player "Eberechi Eze".'
        ),
    )
    parser.add_argument(
        "--players",
        nargs="*",
        default=[],
        help="Alternative way to pass several player names after one flag.",
    )
    parser.add_argument(
        "--team",
        action="append",
        default=[],
        help=(
            "Team name or partial name. Saves one Markdown report containing "
            'every player for that team, for example: --team arsenal.'
        ),
    )
    parser.add_argument(
        "--teams",
        nargs="*",
        default=[],
        help="Alternative way to pass several team names after one flag.",
    )
    parser.add_argument(
        "--top-gaps",
        type=int,
        default=0,
        help="Also include this many biggest positive and negative valuation gaps.",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Also save the full explanation table for every player.",
    )
    args = parser.parse_args()

    player_queries = [*args.player, *args.players]
    team_queries = [*args.team, *args.teams]
    if not player_queries and not team_queries and args.top_gaps <= 0:
        parser.error(
            (
                'choose at least one player/team, e.g. --player saka, '
                'or --team arsenal, or use --top-gaps 10'
            )
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scoring = pd.read_csv(SCORING_PATH)
    explanations = build_explanations(scoring)
    explanations["absolute_gap"] = explanations["valuation_gap"].abs()

    report_title = "2025/26 Player Valuation Explanation Cards"
    output_slug = "selected_player"
    if team_queries and not player_queries and args.top_gaps == 0:
        matched_team_names = [
            match_team(explanations, query)[0]
            for query in team_queries
        ]
        if len(matched_team_names) == 1:
            report_title = f"2025/26 {matched_team_names[0]} Valuation Explanation Cards"
            output_slug = f"team_{slugify(matched_team_names[0])}"
        else:
            report_title = "2025/26 Team Valuation Explanation Cards"
            output_slug = "selected_teams"

    selected_explanations_path = (
        REPORTS_DIR / f"{output_slug}_valuation_explanations.csv"
    )
    cards_path = REPORTS_DIR / f"{output_slug}_valuation_cards.md"

    cards = select_cards(
        explanations,
        player_queries=player_queries,
        team_queries=team_queries,
        extra_top_gaps=args.top_gaps,
    )
    if team_queries and not player_queries and args.top_gaps == 0:
        cards = sort_team_cards(cards)

    cards.to_csv(selected_explanations_path, index=False)
    save_markdown_cards(cards, cards_path, title=report_title)

    if args.save_all:
        explanations_path = REPORTS_DIR / "player_valuation_explanations.csv"
        explanations.to_csv(explanations_path, index=False)
        print(f"Saved all player explanations to {explanations_path}")

    print(f"Saved selected explanations to {selected_explanations_path}")
    print(f"Saved selected Markdown cards to {cards_path}")
    print()
    if len(cards) <= 5:
        print("\n---\n".join(player_card(row) for _, row in cards.iterrows()))
        print()
    else:
        print(f"Selected {len(cards)} players. Full explanations are in the Markdown file.")
        print()
    print(
        cards[
            [
                "player_name",
                "season_club_name",
                "current_club_name",
                "current_market_value_in_eur",
                "predicted_value",
                "valuation_gap",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
