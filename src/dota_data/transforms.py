from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl


def read_processed_tables(processed_dir: str = "data/processed") -> Dict[str, pl.DataFrame]:
    """Load parquet tables produced by io.write_parquet_tables."""
    base = Path(processed_dir)
    paths = {
        "matches": base / "matches.parquet",
        "players": base / "players.parquet",
        "objectives": base / "objectives.parquet",
        "teamfights": base / "teamfights.parquet",
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Processed parquet missing: {missing_list}. Run `python -m src.dota_data.io --raw data/raw/data.json --out {processed_dir}` first.")
    return {name: pl.read_parquet(path) for name, path in paths.items()}


def _parse_json_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, str):
        try:
            loaded = json.loads(val)
        except json.JSONDecodeError:
            return []
        if isinstance(loaded, list):
            return loaded
        return []
    if isinstance(val, list):
        return val
    return []


def _value_at_minute(seq: List[Any], minute: int) -> Optional[Any]:
    if not seq:
        return None
    idx = min(minute, len(seq) - 1)
    return seq[idx]


def lane_phase_features(players: pl.DataFrame, minute: int = 10) -> pl.DataFrame:
    """Extract lane-phase metrics at a given minute from players table."""
    rows = []
    for row in players.iter_rows(named=True):
        lh_t = _parse_json_list(row.get("lh_t"))
        dn_t = _parse_json_list(row.get("dn_t"))
        gold_t = _parse_json_list(row.get("gold_t"))
        xp_t = _parse_json_list(row.get("xp_t"))
        rows.append(
            {
                "match_id": row.get("match_id"),
                "player_slot": row.get("player_slot"),
                "is_radiant": row.get("is_radiant"),
                "hero_id": row.get("hero_id"),
                "lane": row.get("lane"),
                "lane_role": row.get("lane_role"),
                "gpm": row.get("gold_per_min"),
                "xpm": row.get("xp_per_min"),
                "lh_m{m}".format(m=minute): _value_at_minute(lh_t, minute),
                "dn_m{m}".format(m=minute): _value_at_minute(dn_t, minute),
                "gold_m{m}".format(m=minute): _value_at_minute(gold_t, minute),
                "xp_m{m}".format(m=minute): _value_at_minute(xp_t, minute),
            }
        )
    return pl.DataFrame(rows, strict=False)


def match_header(matches: pl.DataFrame) -> pl.DataFrame:
    """Return a lean match-level table useful for joins and modelling."""
    columns = [
        "match_id",
        "start_time",
        "duration",
        "radiant_win",
        "patch",
        "game_mode",
        "lobby_type",
        "radiant_score",
        "dire_score",
        "cluster",
        "region",
    ]
    present_cols = [c for c in columns if c in matches.columns]
    return matches.select(present_cols)
