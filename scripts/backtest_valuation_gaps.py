"""Backtest whether model valuation gaps predict future market movement.

This script tests the project claim we actually care about:

    Do players the model calls "undervalued" tend to rise in value later,
    and do players it calls "overvalued" tend to fall?

For each historical validation season, the script:

1. trains the valuation model only on earlier seasons,
2. scores that season's players,
3. compares model value to the observed post-season Transfermarkt value,
4. checks each player's Transfermarkt value 6/12 months later,
5. summarises whether the top undervalued/overvalued groups moved as expected.

The first version deliberately skips value-band calibration to avoid leaking a
future calibration season into earlier backtests.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.data import load_valuations  # noqa: E402
from scripts.build_combined_valuation_outputs import (  # noqa: E402
    add_combined_value,
    score_current_elite_trajectory,
    score_current_football_ability,
)
from scripts.evaluate_football_ability_model import build_ability_model_data  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports" / "backtests"
DEFAULT_HORIZONS_MONTHS = [6, 12]


def add_future_value(
    scored: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    horizon_months: int,
    max_days_after_horizon: int = 120,
) -> pd.DataFrame:
    """Attach the first valuation at/after a future horizon for each row."""
    result = scored.copy()
    result["target_future_date"] = (
        pd.to_datetime(result["valuation_date"])
        + pd.DateOffset(months=horizon_months)
    )

    valuation_lookup = valuations[
        ["player_id", "date", "market_value_in_eur"]
    ].rename(columns={
        "date": "future_valuation_date",
        "market_value_in_eur": f"future_value_{horizon_months}m",
    })

    candidates = result[
        ["season", "player_id", "valuation_date", "target_future_date"]
    ].merge(
        valuation_lookup,
        on="player_id",
        how="left",
        validate="many_to_many",
    )
    candidates["days_after_target"] = (
        candidates["future_valuation_date"] - candidates["target_future_date"]
    ).dt.days
    valid_candidates = candidates.loc[
        candidates["days_after_target"].between(0, max_days_after_horizon)
    ].copy()

    future_values = (
        valid_candidates.sort_values("future_valuation_date")
        .drop_duplicates(["season", "player_id"], keep="first")
        [
            [
                "season",
                "player_id",
                "future_valuation_date",
                f"future_value_{horizon_months}m",
            ]
        ]
        .rename(columns={
            "future_valuation_date": f"future_valuation_date_{horizon_months}m",
        })
    )

    result = result.merge(
        future_values,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    future_value_column = f"future_value_{horizon_months}m"
    result[f"future_change_{horizon_months}m"] = (
        result[future_value_column] - result["market_value_in_eur"]
    )
    result[f"future_change_pct_{horizon_months}m"] = (
        result[f"future_change_{horizon_months}m"]
        / result["market_value_in_eur"].replace(0, np.nan)
    )
    return result


def score_backtest_season(
    historical_data: pd.DataFrame,
    *,
    validation_season: int,
) -> pd.DataFrame:
    """Train before one season and score that season."""
    train = historical_data.loc[historical_data["season"].lt(validation_season)].copy()
    validation = historical_data.loc[
        historical_data["season"].eq(validation_season)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(f"Cannot backtest season {validation_season}")

    validation["current_market_value_in_eur"] = validation["market_value_in_eur"]
    scored = score_current_football_ability(train, validation)
    scored = score_current_elite_trajectory(train, scored)
    scored = add_combined_value(scored)
    scored["predicted_value"] = scored["combined_model_value"]
    scored["valuation_gap"] = scored["predicted_value"] - scored["market_value_in_eur"]
    scored["valuation_gap_pct"] = (
        scored["valuation_gap"] / scored["market_value_in_eur"].replace(0, np.nan)
    )
    scored["absolute_gap"] = scored["valuation_gap"].abs()
    scored["backtest_train_start"] = int(train["season"].min())
    scored["backtest_train_end"] = int(train["season"].max())
    return scored


def summarise_group(
    data: pd.DataFrame,
    *,
    season: int,
    horizon_months: int,
    group_name: str,
) -> dict[str, float | int | str]:
    """Summarise future value movement for one selected group."""
    future_change = f"future_change_{horizon_months}m"
    future_change_pct = f"future_change_pct_{horizon_months}m"
    available = data.dropna(subset=[future_change, future_change_pct]).copy()

    return {
        "season": season,
        "horizon_months": horizon_months,
        "group": group_name,
        "players": len(data),
        "players_with_future_value": len(available),
        "mean_current_value": available["market_value_in_eur"].mean(),
        "mean_model_value": available["predicted_value"].mean(),
        "mean_gap": available["valuation_gap"].mean(),
        "median_gap": available["valuation_gap"].median(),
        "mean_future_change": available[future_change].mean(),
        "median_future_change": available[future_change].median(),
        "mean_future_change_pct": available[future_change_pct].mean(),
        "median_future_change_pct": available[future_change_pct].median(),
        "share_future_value_up": available[future_change].gt(0).mean(),
        "share_future_value_down": available[future_change].lt(0).mean(),
    }


def build_backtest(
    *,
    validation_seasons: list[int],
    top_n: int,
    min_minutes: int,
    horizons_months: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all backtest seasons and return player rows plus summary rows."""
    historical_data = build_ability_model_data()
    valuations = load_valuations()

    scored_seasons = []
    summary_rows = []

    for season in validation_seasons:
        scored = score_backtest_season(
            historical_data,
            validation_season=season,
        )
        for horizon in horizons_months:
            scored = add_future_value(
                scored,
                valuations,
                horizon_months=horizon,
            )

        scored_seasons.append(scored)

        eligible = scored.loc[
            scored["minutes_played"].ge(min_minutes)
            & scored["market_value_in_eur"].notna()
            & scored["predicted_value"].notna()
        ].copy()
        top_undervalued = eligible.sort_values(
            "valuation_gap",
            ascending=False,
        ).head(top_n)
        recruitment_pool = eligible.loc[
            eligible["age_at_season_end"].lt(27)
            & eligible["market_value_in_eur"].le(60_000_000)
        ].copy()
        top_recruitment_undervalued = recruitment_pool.sort_values(
            "valuation_gap",
            ascending=False,
        ).head(top_n)
        young_high_confidence_undervalued = eligible.loc[
            eligible["age_at_season_end"].lt(27)
            & eligible["previous_known_market_value_in_eur"].notna()
        ].sort_values(
            "valuation_gap",
            ascending=False,
        ).head(top_n)
        top_overvalued = eligible.sort_values(
            "valuation_gap",
            ascending=True,
        ).head(top_n)
        middle = eligible.drop(
            index=top_undervalued.index.union(top_overvalued.index)
        )

        for horizon in horizons_months:
            summary_rows.extend([
                summarise_group(
                    eligible,
                    season=season,
                    horizon_months=horizon,
                    group_name="all_eligible",
                ),
                summarise_group(
                    top_undervalued,
                    season=season,
                    horizon_months=horizon,
                    group_name=f"top_{top_n}_undervalued",
                ),
                summarise_group(
                    top_recruitment_undervalued,
                    season=season,
                    horizon_months=horizon,
                    group_name=f"top_{top_n}_recruitment_undervalued",
                ),
                summarise_group(
                    young_high_confidence_undervalued,
                    season=season,
                    horizon_months=horizon,
                    group_name=f"top_{top_n}_young_known_value_undervalued",
                ),
                summarise_group(
                    top_overvalued,
                    season=season,
                    horizon_months=horizon,
                    group_name=f"top_{top_n}_overvalued",
                ),
                summarise_group(
                    middle,
                    season=season,
                    horizon_months=horizon,
                    group_name="middle_remainder",
                ),
            ])

    player_rows = pd.concat(scored_seasons, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return player_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest whether valuation gaps predict future value moves.",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=[2019, 2020, 2021, 2022, 2023],
        help="Validation seasons to backtest.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of most undervalued/overvalued players per season.",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=900,
        help="Minimum minutes for players to enter ranking groups.",
    )
    parser.add_argument(
        "--horizons-months",
        nargs="*",
        type=int,
        default=DEFAULT_HORIZONS_MONTHS,
        help="Future valuation horizons in months.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    player_rows, summary = build_backtest(
        validation_seasons=args.seasons,
        top_n=args.top_n,
        min_minutes=args.min_minutes,
        horizons_months=args.horizons_months,
    )

    player_rows_path = REPORTS_DIR / "valuation_gap_backtest_player_rows.csv"
    summary_path = REPORTS_DIR / "valuation_gap_backtest_summary.csv"
    player_rows.to_csv(player_rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved player-level backtest rows to {player_rows_path}")
    print(f"Saved backtest summary to {summary_path}")
    print()
    print("Summary:")
    display_columns = [
        "season",
        "horizon_months",
        "group",
        "players_with_future_value",
        "mean_gap",
        "mean_future_change",
        "mean_future_change_pct",
        "share_future_value_up",
        "share_future_value_down",
    ]
    print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
