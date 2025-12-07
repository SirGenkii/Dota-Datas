from __future__ import annotations

import polars as pl


def objectives_timeline(objectives: pl.DataFrame, match_id: int) -> pl.DataFrame:
    """Filter objectives for a match and sort by time for plotting."""
    if "match_id" not in objectives.columns:
        raise ValueError("objectives table must include match_id")
    return objectives.filter(pl.col("match_id") == match_id).sort("time")


def teamfights_for_match(teamfights: pl.DataFrame, match_id: int) -> pl.DataFrame:
    """Filter teamfights for a match."""
    return teamfights.filter(pl.col("match_id") == match_id).sort("teamfight_index")
