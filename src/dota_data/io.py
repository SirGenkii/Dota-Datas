from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _clean_name(name: str | None) -> Optional[str]:
    """Normalize whitespace: trim + collapse multiple spaces."""
    if not name:
        return None
    return " ".join(str(name).strip().split())


def _slugify(text: str) -> Optional[str]:
    """Lightweight slug for tournament names."""
    if not text:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug.strip("-") or None


DEFAULT_TIER = "regular"
DEFAULT_LOCATION = "online"
LEAGUE_MAPPING_PATH = Path("data/mappings/league_mapping.csv")
QUAL_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"qualifier",
        r"closed qualifier",
        r"open qualifier",
        r"regional qualifier",
        r"road to",
        r"rtti",
    ]
]
MAJOR_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"the international",
        r"riyadh masters",
        r"esl one",
        r"dreamleague season",
        r"pgl wallachia",
        r"betboom dacha",
        r"esports world cup",
        r"apac predator league",
        r"blast slam",
        r"elite league",
        r"games of future",
    ]
]


@lru_cache()
def _load_league_mapping(path: str = str(LEAGUE_MAPPING_PATH)) -> Dict[int, Dict[str, str]]:
    """Load league mapping {leagueid: {tier, location}}."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pl.read_csv(p)
    except Exception:
        return {}
    out: Dict[int, Dict[str, str]] = {}
    for row in df.iter_rows(named=True):
        lid = row.get("leagueid")
        if lid is None:
            continue
        try:
            lid_int = int(lid)
        except Exception:
            continue
        tier = (row.get("tier_inferred") or row.get("tier") or DEFAULT_TIER) or DEFAULT_TIER
        location = (row.get("location_inferred") or row.get("location") or DEFAULT_LOCATION) or DEFAULT_LOCATION
        out[lid_int] = {"tier": str(tier).lower(), "location": str(location).lower()}
    return out


def _infer_tournament_tags(
    league_id: Any, tournament_name: Optional[str], league_raw: Any, league_map: Dict[int, Dict[str, str]]
) -> tuple[str, str]:
    """Infer tier/location from mapping, then heuristics, else defaults."""
    # Mapping by league id
    try:
        lid = int(league_id) if league_id is not None else None
    except Exception:
        lid = None
    if lid is not None and lid in league_map:
        m = league_map[lid]
        return m.get("tier", DEFAULT_TIER), m.get("location", DEFAULT_LOCATION)

    # Use raw tier field if present
    tier_field = None
    if isinstance(league_raw, dict):
        tier_field = league_raw.get("tier")
    if tier_field == "premium":
        return "major", "lan"

    # Heuristics on name
    name = tournament_name or ""
    if any(p.search(name) for p in QUAL_PATTERNS):
        return "qualifier", "online"
    if any(p.search(name) for p in MAJOR_PATTERNS):
        # DreamLeague est online; autres majors -> lan
        if "dreamleague" in name.lower():
            return "major", "online"
        return "major", "lan"

    return DEFAULT_TIER, DEFAULT_LOCATION


def _extract_league_name(league_value: Any) -> Optional[str]:
    """Extract and clean the league/tournament name from the raw league field."""
    raw_name: Optional[str] = None
    if isinstance(league_value, dict):
        raw_name = league_value.get("name")
    elif isinstance(league_value, str):
        # If already JSON-encoded, try to decode; otherwise use as-is.
        try:
            decoded = json.loads(league_value)
            if isinstance(decoded, dict):
                raw_name = decoded.get("name")
            else:
                raw_name = league_value
        except Exception:
            raw_name = league_value
    return _clean_name(raw_name)


def load_raw_matches(path: Path | str) -> List[Dict[str, Any]]:
    """Load raw JSON array from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_alias_map(path: Optional[Path]) -> Dict[int, int]:
    """Load alias->canonical mapping from CSV with columns alias_team_id,canonical_team_id."""
    mapping: Dict[int, int] = {}
    if path is None or not path.exists():
        return mapping
    try:
        df = pl.read_csv(path)
    except Exception:
        return mapping
    if not {"alias_team_id", "canonical_team_id"}.issubset(set(df.columns)):
        return mapping
    for row in df.iter_rows(named=True):
        try:
            alias = int(row["alias_team_id"])
            canon = int(row["canonical_team_id"])
            if alias != canon:
                mapping[alias] = canon
        except Exception:
            continue
    return mapping


def _map_team_id(team_id: Any, alias_map: Dict[int, int]) -> Any:
    try:
        tid = int(team_id)
    except Exception:
        return team_id
    return alias_map.get(tid, tid)


def matches_table(raw: Iterable[Dict[str, Any]], alias_map: Optional[Dict[int, int]] = None) -> pl.DataFrame:
    """Flatten match-level info (no heavy nested arrays)."""
    alias_map = alias_map or {}
    league_map = _load_league_mapping()
    rows: List[Dict[str, Any]] = []
    for entry in raw:
        match = entry.get("json", {})
        row = {k: _serialize_value(v) for k, v in match.items() if k not in EXCLUDED_MATCH_KEYS}
        league_name = _extract_league_name(match.get("league"))
        tier, location = _infer_tournament_tags(match.get("leagueid"), league_name, match.get("league"), league_map)
        row["league_name"] = league_name
        row["tournament_name"] = league_name
        row["tournament_slug"] = _slugify(league_name)
        row["tournament_tier"] = tier
        row["tournament_location"] = location
        # Map team ids to canonical if provided
        row["radiant_team_id"] = _map_team_id(row.get("radiant_team_id"), alias_map)
        row["dire_team_id"] = _map_team_id(row.get("dire_team_id"), alias_map)
        row["players_count"] = len(match.get("players", []))
        row["objectives_count"] = len(match.get("objectives", []))
        row["teamfights_count"] = len(match.get("teamfights", []))
        rows.append(row)
    df = _to_df(rows)
    return infer_series_fields(df)


def infer_series_fields(matches: pl.DataFrame) -> pl.DataFrame:
    """
    Infer best-of type (bo_type) and normalized series_type for matches.

    This is computed per (series_id, leagueid) using:
    - number of maps actually played (match_count)
    - wins per team derived from radiant_win
    """
    if matches is None or matches.is_empty():
        return matches

    df = matches
    if "series_id" in df.columns and "leagueid" in df.columns:
        if "series_type" in df.columns and "series_type_raw" not in df.columns:
            df = df.with_columns(pl.col("series_type").alias("series_type_raw"))

        # winner_team_id per match (None if radiant_win is null)
        if "winner_team_id" not in df.columns and {"radiant_win", "radiant_team_id", "dire_team_id"}.issubset(set(df.columns)):
            df = df.with_columns(
                pl.when(pl.col("radiant_win") == True)
                .then(pl.col("radiant_team_id"))
                .when(pl.col("radiant_win") == False)
                .then(pl.col("dire_team_id"))
                .otherwise(None)
                .alias("winner_team_id")
            )

        valid = df.filter(pl.col("series_id").is_not_null())
        series_rows = []
        for _, g in valid.group_by(["series_id", "leagueid"], maintain_order=True):
            rows_g = g.to_dicts()
            if not rows_g:
                continue
            match_count = len(rows_g)
            teams = []
            for r in rows_g:
                for k in ("radiant_team_id", "dire_team_id"):
                    if r.get(k) is not None:
                        tid = int(r[k])
                        if tid not in teams:
                            teams.append(tid)
            teams_in_series = len(teams)
            wins = {t: 0 for t in teams}
            for r in rows_g:
                w = r.get("winner_team_id")
                if w is not None:
                    wins[w] = wins.get(w, 0) + 1
            max_wins = max(wins.values()) if wins else 0
            series_type_raw = rows_g[0].get("series_type_raw")

            # Infer bo_type based on wins and match_count
            if match_count >= 5 or max_wins >= 3:
                bo_type = 5
            elif match_count == 4:
                bo_type = 5
            elif match_count == 3:
                bo_type = 5 if max_wins == 3 else 3
            elif match_count == 2:
                if wins and list(wins.values()).count(1) == 2:
                    bo_type = 2  # 1-1 likely BO2
                elif series_type_raw == 3:
                    bo_type = 2
                else:
                    bo_type = 3
            elif match_count == 1:
                bo_type = 1
            else:
                bo_type = None
            series_rows.append(
                {
                    "series_id": rows_g[0].get("series_id"),
                    "leagueid": rows_g[0].get("leagueid"),
                    "bo_type": bo_type,
                    "teams_in_series": teams_in_series,
                    "series_type_raw": series_type_raw,
                }
            )

        if series_rows:
            series_df = pl.DataFrame(series_rows, strict=False)
            df = df.join(series_df, on=["series_id", "leagueid"], how="left")
            df = df.with_columns(pl.coalesce([pl.col("bo_type"), pl.col("series_type_raw")]).alias("series_type"))

    cast_cols = {}
    for col in ["bo_type", "series_type", "series_type_raw"]:
        if col in df.columns:
            cast_cols[col] = pl.col(col).cast(pl.Int64, strict=False)
    if cast_cols:
        df = df.with_columns(list(cast_cols.values()))
    return df


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


def series_summary_table(matches: pl.DataFrame) -> pl.DataFrame:
    """Build a per-series summary (score, bo_type, teams, timings)."""
    if matches is None or matches.is_empty() or "series_id" not in matches.columns:
        return pl.DataFrame([], schema={})
    rows = []
    for _, g in matches.filter(pl.col("series_id").is_not_null()).group_by(["series_id", "leagueid"], maintain_order=True):
        rows_g = g.to_dicts()
        if not rows_g:
            continue
        match_count = len(rows_g)
        leagueid = rows_g[0].get("leagueid")
        league_name = rows_g[0].get("league_name")
        tournament_name = rows_g[0].get("tournament_name")
        tournament_slug = rows_g[0].get("tournament_slug")
        tournament_tier = rows_g[0].get("tournament_tier")
        tournament_location = rows_g[0].get("tournament_location")
        series_type_raw = rows_g[0].get("series_type_raw")
        bo_type = rows_g[0].get("bo_type")
        start_min = min(r.get("start_time", 0) or 0 for r in rows_g)
        start_max = max(r.get("start_time", 0) or 0 for r in rows_g)
        teams = []
        for r in rows_g:
            for k in ("radiant_team_id", "dire_team_id"):
                if r.get(k) is not None:
                    tid = int(r[k])
                    if tid not in teams:
                        teams.append(tid)
        teams_in_series = len(teams)
        wins = {t: 0 for t in teams}
        for r in rows_g:
            win_tid = None
            if r.get("radiant_win") is True:
                win_tid = r.get("radiant_team_id")
            elif r.get("radiant_win") is False:
                win_tid = r.get("dire_team_id")
            if win_tid is not None:
                wins[int(win_tid)] = wins.get(int(win_tid), 0) + 1
        winner_team_id = None
        max_wins = 0
        if wins:
            winner_team_id = max(wins, key=wins.get)
            max_wins = wins[winner_team_id]
        score_team_a = score_team_b = None
        team_a = teams[0] if teams else None
        team_b = teams[1] if teams_in_series >= 2 else None
        if team_a is not None and team_b is not None:
            score_team_a = wins.get(team_a, 0)
            score_team_b = wins.get(team_b, 0)
        rows.append(
            {
                "series_id": rows_g[0].get("series_id"),
                "leagueid": leagueid,
                "league_name": league_name,
                "tournament_name": tournament_name,
                "tournament_slug": tournament_slug,
                "tournament_tier": tournament_tier,
                "tournament_location": tournament_location,
                "series_type_raw": series_type_raw,
                "bo_type": bo_type,
                "match_count": match_count,
                "teams": teams,
                "teams_in_series": teams_in_series,
                "team_a": team_a,
                "team_b": team_b,
                "score_team_a": score_team_a,
                "score_team_b": score_team_b,
                "winner_team_id": winner_team_id,
                "max_wins": max_wins,
                "start_time_min": start_min,
                "start_time_max": start_max,
            }
        )
    return pl.DataFrame(rows, strict=False)


def write_parquet_tables(raw_path: Path | str, output_dir: Path | str, alias_path: Optional[Path] = None) -> Dict[str, Path]:
    """Generate parquet tables for matches, players, objectives, teamfights, and series summaries."""
    raw = load_raw_matches(raw_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alias_map = load_alias_map(alias_path)

    tables = {
        "matches": matches_table(raw, alias_map=alias_map),
        "players": players_table(raw),
        "objectives": objectives_table(raw),
        "teamfights": teamfights_table(raw),
    }
    tables["series"] = series_summary_table(tables["matches"])

    paths: Dict[str, Path] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        paths[name] = path
    return paths


def load_raw_matches_from_files(paths: Iterable[Path | str]) -> List[Dict[str, Any]]:
    """Load and concatenate multiple raw JSON files (each containing an array of wrapped matches)."""
    out: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows = load_raw_matches(p)
        except Exception:
            continue
        if isinstance(rows, list):
            out.extend([r for r in rows if isinstance(r, dict)])
    return out


def append_parquet_tables(
    raw: Iterable[Dict[str, Any]],
    output_dir: Path | str,
    alias_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Append a batch of raw matches to existing processed parquet tables.

    This updates:
    - matches.parquet (dedup by match_id, then re-infer series fields)
    - players.parquet / objectives.parquet / teamfights.parquet (append)
    - series.parquet (recomputed from the updated matches table)
    """
    raw_list = list(raw)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alias_map = load_alias_map(alias_path)

    new_tables = {
        "matches": matches_table(raw_list, alias_map=alias_map),
        "players": players_table(raw_list),
        "objectives": objectives_table(raw_list),
        "teamfights": teamfights_table(raw_list),
    }

    paths: Dict[str, Path] = {}

    # Matches: union + dedup + recompute series fields.
    matches_path = out_dir / "matches.parquet"
    if matches_path.exists():
        old_matches = pl.read_parquet(matches_path)
        merged_matches = pl.concat([old_matches, new_tables["matches"]], how="diagonal_relaxed")
    else:
        merged_matches = new_tables["matches"]
    if "match_id" in merged_matches.columns:
        merged_matches = merged_matches.unique(subset=["match_id"], keep="last")
    merged_matches = infer_series_fields(merged_matches)
    merged_matches.write_parquet(matches_path)
    paths["matches"] = matches_path

    # Other append-only tables.
    for name in ["players", "objectives", "teamfights"]:
        path = out_dir / f"{name}.parquet"
        if path.exists():
            old = pl.read_parquet(path)
            merged = pl.concat([old, new_tables[name]], how="diagonal_relaxed")
        else:
            merged = new_tables[name]
        merged.write_parquet(path)
        paths[name] = path

    # Series: recompute from full matches table.
    series_df = series_summary_table(merged_matches)
    series_path = out_dir / "series.parquet"
    series_df.write_parquet(series_path)
    paths["series"] = series_path

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw Dota matches JSON to parquet tables.")
    parser.add_argument("--raw", default="data/raw/data.json", help="Path to raw JSON file.")
    parser.add_argument("--out", default="data/processed", help="Output directory for parquet tables.")
    parser.add_argument("--aliases", default="data/team_aliases.csv", help="CSV file mapping alias_team_id -> canonical_team_id.")
    args = parser.parse_args()

    alias_path = Path(args.aliases) if args.aliases else None
    paths = write_parquet_tables(args.raw, args.out, alias_path=alias_path)
    print("Parquet tables written:")
    for name, path in paths.items():
        print(f"- {name}: {path}")

    summary = summarize_raw(load_raw_matches(args.raw))
    print("Dataset summary:")
    for key, val in summary.items():
        print(f"{key}: {val}")
