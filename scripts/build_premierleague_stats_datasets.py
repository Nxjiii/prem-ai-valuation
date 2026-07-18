"""Build interim datasets enriched with official Premier League stats."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from prem_valuation.data import INTERIM_DIR, RAW_DIR, REFERENCE_DIR, load_interim_tables
from prem_valuation.premierleague_stats import (
    add_premierleague_rate_features,
    attach_premierleague_stats,
    load_premierleague_aliases,
)


PL_STATS_PATH = RAW_DIR / "premierleague" / "player_stats_2016_2025.csv"
ALIASES_PATH = REFERENCE_DIR / "premierleague_player_aliases.csv"


def main() -> None:
    labelled, scoring = load_interim_tables()
    premierleague_stats = pd.read_csv(PL_STATS_PATH)
    aliases = load_premierleague_aliases(ALIASES_PATH)

    labelled_enriched = add_premierleague_rate_features(
        attach_premierleague_stats(labelled, premierleague_stats, aliases)
    )
    scoring_enriched = add_premierleague_rate_features(
        attach_premierleague_stats(scoring, premierleague_stats, aliases)
    )

    labelled_path = INTERIM_DIR / "labelled_player_seasons_with_pl_stats.csv.gz"
    scoring_path = INTERIM_DIR / "scoring_2025_26_with_pl_stats.csv.gz"
    labelled_enriched.to_csv(labelled_path, index=False)
    scoring_enriched.to_csv(scoring_path, index=False)

    labelled_recent = labelled_enriched.loc[labelled_enriched["season"].ge(2016)]
    print(f"Saved: {labelled_path}")
    print(f"Saved: {scoring_path}")
    print(
        "Labelled coverage from 2016: "
        f"{labelled_recent['has_premierleague_stats'].mean():.1%} "
        f"({int(labelled_recent['has_premierleague_stats'].sum()):,}/"
        f"{len(labelled_recent):,})"
    )
    print(
        "Scoring coverage: "
        f"{scoring_enriched['has_premierleague_stats'].mean():.1%} "
        f"({int(scoring_enriched['has_premierleague_stats'].sum()):,}/"
        f"{len(scoring_enriched):,})"
    )


if __name__ == "__main__":
    main()
