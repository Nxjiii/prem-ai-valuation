"""Scrape historical Premier League player-season stats from the official API.

The Premier League website is backed by the Pulse Live football API. This script
pulls one ranked stat table at a time, then pivots those tables into one
player-season row.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "premierleague" / "player_stats_2016_2025.csv"

COMPETITION_ID = 1
PAGE_SIZE = 1000

SEASONS = {
    2016: 54,
    2017: 79,
    2018: 210,
    2019: 274,
    2020: 363,
    2021: 418,
    2022: 489,
    2023: 578,
    2024: 719,
    2025: 777,
}

STAT_IDS = [
    "appearances",
    "mins_played",
    "goals",
    "goal_assist",
    "total_scoring_att",
    "ontarget_scoring_att",
    "big_chance_created",
    "big_chance_missed",
    "total_pass",
    "accurate_pass",
    "total_cross",
    "accurate_cross",
    "touches",
    "poss_lost_all",
    "total_tackle",
    "won_tackle",
    "interception",
    "total_clearance",
    "outfielder_block",
    "duel_won",
    "duel_lost",
    "aerial_won",
    "aerial_lost",
    "error_lead_to_goal",
    "yellow_card",
    "red_card",
]


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Origin": "https://www.premierleague.com",
            "User-Agent": "Mozilla/5.0",
        }
    )
    return session


def fetch_ranked_stat(
    *,
    stat_id: str,
    season_start_year: int,
    comp_season_id: int,
    max_retries: int = 3,
) -> list[dict]:
    session = make_session()
    url = f"https://footballapi.pulselive.com/football/stats/ranked/players/{stat_id}"
    params = {
        "page": 0,
        "pageSize": PAGE_SIZE,
        "comps": COMPETITION_ID,
        "compSeasons": comp_season_id,
        "altIds": "true",
    }
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=20)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(1 + attempt)

    payload = response.json()

    rows = []
    for item in payload["stats"]["content"]:
        owner = item["owner"]
        name = owner.get("name", {})
        current_team = owner.get("currentTeam", {})
        alt_ids = owner.get("altIds", {})
        info = owner.get("info", {})

        rows.append(
            {
                "season": season_start_year,
                "comp_season_id": comp_season_id,
                "pl_player_id": owner.get("id"),
                "pl_player_profile_id": owner.get("playerId"),
                "opta_player_id": alt_ids.get("opta"),
                "player_name": name.get("display"),
                "first_name": name.get("first"),
                "last_name": name.get("last"),
                "position": info.get("position"),
                "position_info": info.get("positionInfo"),
                "current_team_name": current_team.get("name"),
                "current_team_id": current_team.get("id"),
                "current_team_abbr": current_team.get("club", {}).get("abbr"),
                "stat_id": stat_id,
                "stat_value": item.get("value"),
            }
        )

    return rows


def scrape_stats(
    *,
    seasons: dict[int, int] = SEASONS,
    stat_ids: list[str] = STAT_IDS,
    sleep_seconds: float = 0.15,
    max_workers: int = 8,
) -> pd.DataFrame:
    long_rows = []
    jobs = [
        (season_start_year, comp_season_id, stat_id)
        for season_start_year, comp_season_id in seasons.items()
        for stat_id in stat_ids
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_ranked_stat,
                stat_id=stat_id,
                season_start_year=season_start_year,
                comp_season_id=comp_season_id,
            ): (season_start_year, stat_id)
            for season_start_year, comp_season_id, stat_id in jobs
        }

        for future in as_completed(futures):
            season_start_year, stat_id = futures[future]
            rows = future.result()
            long_rows.extend(rows)
            print(
                f"{season_start_year} {stat_id}: {len(rows):,} rows",
                flush=True,
            )
            time.sleep(sleep_seconds)

    long_stats = pd.DataFrame(long_rows)
    id_columns = [
        "season",
        "comp_season_id",
        "pl_player_id",
        "pl_player_profile_id",
        "opta_player_id",
        "player_name",
        "first_name",
        "last_name",
        "position",
        "position_info",
        "current_team_name",
        "current_team_id",
        "current_team_abbr",
    ]
    long_stats[id_columns] = long_stats[id_columns].fillna("unknown")

    wide_stats = (
        long_stats.pivot_table(
            index=id_columns,
            columns="stat_id",
            values="stat_value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    stat_columns = [column for column in wide_stats.columns if column not in id_columns]
    wide_stats[stat_columns] = wide_stats[stat_columns].fillna(0)
    return wide_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="CSV path for the scraped official Premier League player stats.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = scrape_stats(max_workers=args.max_workers)
    stats.to_csv(output_path, index=False)

    print(f"Saved {len(stats):,} rows to {output_path}")


if __name__ == "__main__":
    main()
