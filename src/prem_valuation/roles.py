"""Player role-group helpers.

The model still trains on broad positions for sample-size stability, but reports
and explanations can use more detailed football roles for fairer comparisons.
"""

from __future__ import annotations

import pandas as pd


ROLE_GROUP_BY_SUB_POSITION = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Centre-Back",
    "Left-Back": "Left-Back",
    "Right-Back": "Right-Back",
    "Defensive Midfield": "Defensive Midfield",
    "Central Midfield": "Central Midfield",
    "Attacking Midfield": "Attacking Midfield",
    "Left Midfield": "Wide Midfield",
    "Right Midfield": "Wide Midfield",
    "Left Winger": "Left Winger",
    "Right Winger": "Right Winger",
    "Centre-Forward": "Centre-Forward",
    "Second Striker": "Second Striker",
}


ROLE_GROUP_ORDER = {
    "Goalkeeper": 0,
    "Centre-Back": 1,
    "Left-Back": 2,
    "Right-Back": 3,
    "Defensive Midfield": 4,
    "Central Midfield": 5,
    "Attacking Midfield": 6,
    "Wide Midfield": 7,
    "Left Winger": 8,
    "Right Winger": 9,
    "Second Striker": 10,
    "Centre-Forward": 11,
    "Other": 99,
}


def add_role_group(data: pd.DataFrame) -> pd.DataFrame:
    """Attach detailed role groups derived from Transfermarkt sub-positions."""
    result = data.copy()
    result["role_group"] = (
        result["sub_position"]
        .map(ROLE_GROUP_BY_SUB_POSITION)
        .fillna(result["position"])
        .fillna("Other")
    )
    result["role_group"] = result["role_group"].where(
        result["role_group"].isin(ROLE_GROUP_ORDER),
        "Other",
    )
    return result
