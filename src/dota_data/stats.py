from __future__ import annotations

import polars as pl


def kill_participation(players: pl.DataFrame) -> pl.DataFrame:
    """Compute kill participation per player."""
    if "kills" not in players.columns or "assists" not in players.columns:
        raise pl.ColumnNotFoundError("kills/assists columns required for kill participation")

    team_kills = players.group_by(["match_id", "is_radiant"]).agg(
        pl.col("kills").sum().alias("team_kills")
    )
    base = players.select(
        "match_id",
        "player_slot",
        "is_radiant",
        "hero_id",
        "kills",
        "assists",
    )
    joined = base.join(team_kills, on=["match_id", "is_radiant"], how="left")
    return joined.with_columns(
        ((pl.col("kills") + pl.col("assists")) / pl.col("team_kills")).alias("kill_participation")
    )


def vision_stats(players: pl.DataFrame) -> pl.DataFrame:
    """Basic vision stats per player."""
    cols = [c for c in ["obs_placed", "sen_placed", "observer_kills", "sentry_kills"] if c in players.columns]
    return players.select(
        "match_id",
        "player_slot",
        "is_radiant",
        "hero_id",
        *cols,
    ).with_columns(
        (pl.col("obs_placed") + pl.col("sen_placed")).alias("wards_placed")
        if "obs_placed" in players.columns and "sen_placed" in players.columns
        else pl.lit(None).alias("wards_placed")
    )


def team_scores(matches: pl.DataFrame) -> pl.DataFrame:
    """Return match_id with team scores and winner flag."""
    cols = [c for c in ["match_id", "radiant_score", "dire_score", "radiant_win", "duration"] if c in matches.columns]
    return matches.select(cols)
