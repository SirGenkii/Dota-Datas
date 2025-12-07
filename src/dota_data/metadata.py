from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import polars as pl


def _parse_team_field(val: Any) -> dict:
    """Parse team json string/dict into a python dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            loaded = json.loads(val)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def build_team_dictionary(matches: pl.DataFrame) -> pl.DataFrame:
    """Return unique team entries with id/name/tag/logo."""
    rows = []
    for row in matches.iter_rows(named=True):
        for side in ("radiant", "dire"):
            team_id = row.get(f"{side}_team_id")
            name = row.get(f"{side}_name")
            logo = row.get(f"{side}_logo")
            team_blob = _parse_team_field(row.get(f"{side}_team"))
            rows.append(
                {
                    "team_id": team_id or team_blob.get("team_id"),
                    "name": name or team_blob.get("name"),
                    "tag": team_blob.get("tag"),
                    "logo_url": team_blob.get("logo_url") or logo,
                    "side_sampled": side,
                }
            )
    df = pl.DataFrame(rows, strict=False)
    df = df.filter(pl.any_horizontal(~pl.col(["team_id", "name"]).is_null()))
    return df.unique(subset=["team_id", "name", "tag", "logo_url"])


def build_player_dictionary(players: pl.DataFrame) -> pl.DataFrame:
    """Return unique players with first known names and match counts."""
    if "account_id" not in players.columns:
        raise pl.ColumnNotFoundError("account_id column required to build player dictionary")
    return (
        players.group_by("account_id")
        .agg(
            pl.len().alias("matches_played"),
            pl.col("personaname").drop_nulls().first().alias("personaname"),
            pl.col("name").drop_nulls().first().alias("name"),
        )
        .sort("matches_played", descending=True)
    )


def build_hero_counts(players: pl.DataFrame) -> pl.DataFrame:
    """Return hero_id with counts from the dataset."""
    if "hero_id" not in players.columns:
        raise pl.ColumnNotFoundError("hero_id column required to build hero counts")
    return (
        players.group_by("hero_id")
        .agg(pl.len().alias("matches_played"))
        .sort("matches_played", descending=True)
    )


def load_hero_dictionary(paths: Optional[Iterable[Path | str]] = None) -> pl.DataFrame:
    """
    Load hero dictionary if available.
    Accepts JSON (list of objects with id/name/localized_name) or CSV with hero_id,name.
    """
    candidates = list(paths) if paths else [Path("data/dictionaries/heroes.json"), Path("data/dictionaries/heroes.csv")]
    for cand in candidates:
        p = Path(cand)
        if not p.exists():
            continue
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                data = data.get("heroes") or data.values()
            rows = []
            for item in data:
                if isinstance(item, dict) and "id" in item:
                    rows.append(
                        {
                            "hero_id": item.get("id"),
                            "name": item.get("name"),
                            "localized_name": item.get("localized_name"),
                        }
                    )
            return pl.DataFrame(rows, strict=False)
        if p.suffix == ".csv":
            return pl.read_csv(p)
    raise FileNotFoundError(f"No hero dictionary found in {candidates}")
