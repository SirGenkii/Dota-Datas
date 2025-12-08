from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import polars as pl


def _to_df(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """Create a DataFrame with wide schema inference to avoid type conflicts."""
    infer_len = len(rows) if rows else 0
    return pl.DataFrame(rows, strict=False, infer_schema_length=infer_len if infer_len > 0 else None)

# Keys that are nested/timeline-heavy and should be handled separately from the match header table.
EXCLUDED_MATCH_KEYS = {
    "players",
    "objectives",
    "teamfights",
    "radiant_gold_adv",
    "radiant_xp_adv",
    "picks_bans",
    "draft_timings",
    "cosmetics",
    "chat",
    "pauses",
}

def _serialize_value(value: Any) -> Any:
    """Convert non-scalar values to JSON strings to keep parquet simple."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def load_raw_matches(path: Path | str) -> List[Dict[str, Any]]:
    """Load raw JSON array from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def matches_table(raw: Iterable[Dict[str, Any]]) -> pl.DataFrame:
    """Flatten match-level info (no heavy nested arrays)."""
    rows: List[Dict[str, Any]] = []
    for entry in raw:
        match = entry.get("json", {})
        row = {k: _serialize_value(v) for k, v in match.items() if k not in EXCLUDED_MATCH_KEYS}
        row["players_count"] = len(match.get("players", []))
        row["objectives_count"] = len(match.get("objectives", []))
        row["teamfights_count"] = len(match.get("teamfights", []))
        rows.append(row)
    return _to_df(rows)


def players_table(raw: Iterable[Dict[str, Any]]) -> pl.DataFrame:
    """Flatten per-player data with match_id/is_radiant attached."""
    rows: List[Dict[str, Any]] = []
    for entry in raw:
        match = entry.get("json", {})
        match_id = match.get("match_id")
        for player in match.get("players", []):
            row = {k: _serialize_value(v) for k, v in player.items()}
            row["match_id"] = match_id
            slot = player.get("player_slot", 0)
            row["is_radiant"] = slot < 128
            rows.append(row)
    return _to_df(rows)


def objectives_table(raw: Iterable[Dict[str, Any]]) -> pl.DataFrame:
    """Flatten objectives/timeline events."""
    rows: List[Dict[str, Any]] = []
    for entry in raw:
        match = entry.get("json", {})
        match_id = match.get("match_id")
        for idx, obj in enumerate(match.get("objectives", [])):
            row = dict(obj)
            row["match_id"] = match_id
            row["objective_index"] = idx
            rows.append(row)
    return _to_df(rows)


def teamfights_table(raw: Iterable[Dict[str, Any]]) -> pl.DataFrame:
    """Flatten teamfight summaries with per-player stats."""
    rows: List[Dict[str, Any]] = []
    for entry in raw:
        match = entry.get("json", {})
        match_id = match.get("match_id")
        for tf_idx, tf in enumerate(match.get("teamfights", [])):
            base = {
                "match_id": match_id,
                "teamfight_index": tf_idx,
                "start": tf.get("start"),
                "end": tf.get("end"),
                "last_death": tf.get("last_death"),
            }
            for p_idx, player in enumerate(tf.get("players", [])):
                row: Dict[str, Any] = dict(base)
                row["teamfight_player_index"] = p_idx
                row["deaths"] = player.get("deaths")
                row["buybacks"] = player.get("buybacks")
                row["gold_delta"] = player.get("gold_delta")
                row["xp_delta"] = player.get("xp_delta")
                row["xp_start"] = player.get("xp_start")
                row["xp_end"] = player.get("xp_end")
                row["healing"] = player.get("healing")
                row["damage"] = player.get("damage")
                # Keep nested/action info for later analysis.
                row["ability_uses"] = _serialize_value(player.get("ability_uses"))
                row["item_uses"] = _serialize_value(player.get("item_uses"))
                row["ability_targets"] = _serialize_value(player.get("ability_targets"))
                row["deaths_pos"] = _serialize_value(player.get("deaths_pos"))
                row["killed"] = _serialize_value(player.get("killed"))
                rows.append(row)
    return _to_df(rows)


def summarize_raw(raw: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return quick stats about the dataset."""
    raw_list = list(raw)
    matches_count = len(raw_list)
    top_keys = set()
    player_keys = set()
    objective_keys = set()
    max_objectives = 0
    for entry in raw_list:
        match = entry.get("json", {})
        top_keys.update(match.keys())
        players = match.get("players", [])
        for p in players:
            player_keys.update(p.keys())
        objs = match.get("objectives", []) or []
        max_objectives = max(max_objectives, len(objs))
        for obj in objs:
            objective_keys.update(obj.keys())
    return {
        "matches": matches_count,
        "top_level_keys": sorted(top_keys),
        "player_keys_count": len(player_keys),
        "objective_fields": sorted(objective_keys),
        "max_objectives_per_match": max_objectives,
    }


def write_parquet_tables(raw_path: Path | str, output_dir: Path | str) -> Dict[str, Path]:
    """Generate parquet tables for matches, players, objectives, and teamfights."""
    raw = load_raw_matches(raw_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "matches": matches_table(raw),
        "players": players_table(raw),
        "objectives": objectives_table(raw),
        "teamfights": teamfights_table(raw),
    }

    paths: Dict[str, Path] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        paths[name] = path
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw Dota matches JSON to parquet tables.")
    parser.add_argument("--raw", default="data/raw/data.json", help="Path to raw JSON file.")
    parser.add_argument("--out", default="data/processed", help="Output directory for parquet tables.")
    args = parser.parse_args()

    paths = write_parquet_tables(args.raw, args.out)
    print("Parquet tables written:")
    for name, path in paths.items():
        print(f"- {name}: {path}")

    summary = summarize_raw(load_raw_matches(args.raw))
    print("Dataset summary:")
    for key, val in summary.items():
        print(f"{key}: {val}")
