from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import polars as pl

from .api import write_json
from .io import load_raw_matches_from_files


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run_dir(updates_dir: Path) -> Optional[Path]:
    if not updates_dir.exists():
        return None
    runs = sorted([p for p in updates_dir.glob("run_*") if p.is_dir()])
    return runs[-1] if runs else None


def _run_chunk_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("matches_chunk*.json"))


def _existing_match_ids(processed_dir: Path) -> Set[int]:
    file_path = processed_dir / "matches.parquet"
    dir_path = processed_dir / "matches"
    if file_path.exists():
        df = pl.read_parquet(file_path, columns=["match_id"])
    elif dir_path.exists():
        df = pl.read_parquet(str(dir_path / "**/*.parquet"), columns=["match_id"])
    else:
        return set()
    if "match_id" not in df.columns:
        return set()
    return {int(v) for v in df["match_id"].drop_nulls().to_list() if v is not None}


def _match_rows_from_raw(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in raw:
        match = entry.get("json") if isinstance(entry, dict) and "json" in entry else entry
        if not isinstance(match, dict):
            continue
        mid = match.get("match_id")
        if not isinstance(mid, int):
            continue
        st = match.get("start_time")
        start_time = int(st) if isinstance(st, int) else None
        rows.append(
            {
                "match_id": mid,
                "start_time": start_time,
                "start_dt": datetime.fromtimestamp(start_time, tz=timezone.utc) if start_time is not None else None,
                "leagueid": match.get("leagueid"),
                "series_id": match.get("series_id"),
                "map_num": match.get("map_num"),
                "radiant_team_id": match.get("radiant_team_id"),
                "dire_team_id": match.get("dire_team_id"),
                "radiant_name": match.get("radiant_name"),
                "dire_name": match.get("dire_name"),
                "radiant_win": match.get("radiant_win"),
            }
        )
    return rows


def write_run_traces(run_dir: Path, raw: List[Dict[str, Any]], processed_dir: Optional[Path]) -> Dict[str, str]:
    """
    Persist human-friendly run traces:
    - matches_trace.parquet: match-level info for downloaded matches
    - series_trace.parquet: series-level view (final score if processed series.parquet is available)
    """
    trace_paths: Dict[str, str] = {}

    match_rows = _match_rows_from_raw(raw)
    matches_schema = {
        "match_id": pl.Int64,
        "start_time": pl.Int64,
        "start_dt": pl.Datetime(time_zone="UTC"),
        "leagueid": pl.Int64,
        "series_id": pl.Int64,
        "map_num": pl.Int64,
        "radiant_team_id": pl.Int64,
        "dire_team_id": pl.Int64,
        "radiant_name": pl.Utf8,
        "dire_name": pl.Utf8,
        "radiant_win": pl.Boolean,
    }
    matches_df = pl.DataFrame(match_rows, strict=False, schema=matches_schema)
    if not matches_df.is_empty():
        matches_df = matches_df.unique(subset=["match_id"], keep="last").sort("start_time", descending=True, nulls_last=True)
    matches_path = run_dir / "matches_trace.parquet"
    matches_df.write_parquet(matches_path)
    trace_paths["matches_trace_parquet"] = str(matches_path)
    # CSV for quick human inspection
    matches_csv = run_dir / "matches_trace.csv"
    matches_df.select(
        [c for c in ["start_dt", "match_id", "radiant_name", "dire_name", "radiant_win", "leagueid", "series_id", "map_num"] if c in matches_df.columns]
    ).write_csv(matches_csv)
    trace_paths["matches_trace_csv"] = str(matches_csv)

    # Build series trace
    series_rows: List[Dict[str, Any]] = []
    if not matches_df.is_empty() and {"series_id", "leagueid"}.issubset(set(matches_df.columns)):
        valid = matches_df.filter(pl.col("series_id").is_not_null() & pl.col("leagueid").is_not_null())
        if not valid.is_empty():
            for key, g in valid.partition_by(["series_id", "leagueid"], as_dict=True, maintain_order=True).items():
                series_id, leagueid = key
                rows = g.to_dicts()
                # teams in this run's subset
                teams: List[int] = []
                for r in rows:
                    for k in ("radiant_team_id", "dire_team_id"):
                        v = r.get(k)
                        if v is None:
                            continue
                        try:
                            tid = int(v)
                        except Exception:
                            continue
                        if tid not in teams:
                            teams.append(tid)
                wins = {t: 0 for t in teams}
                for r in rows:
                    rw = r.get("radiant_win")
                    if rw is None:
                        continue
                    if rw is True and r.get("radiant_team_id") is not None:
                        try:
                            wins[int(r["radiant_team_id"])] += 1
                        except Exception:
                            pass
                    if rw is False and r.get("dire_team_id") is not None:
                        try:
                            wins[int(r["dire_team_id"])] += 1
                        except Exception:
                            pass
                team_a = teams[0] if teams else None
                team_b = teams[1] if len(teams) >= 2 else None
                score_a_run = wins.get(team_a) if team_a is not None else None
                score_b_run = wins.get(team_b) if team_b is not None else None
                start_time_min = None
                start_times = [r.get("start_time") for r in rows if r.get("start_time") is not None]
                if start_times:
                    start_time_min = int(min(start_times))
                series_rows.append(
                    {
                        "series_id": series_id,
                        "leagueid": leagueid,
                        "start_time_min_run": start_time_min,
                        "start_dt_run": datetime.fromtimestamp(start_time_min, tz=timezone.utc) if start_time_min is not None else None,
                        "team_a_id_run": team_a,
                        "team_b_id_run": team_b,
                        "score_team_a_run": score_a_run,
                        "score_team_b_run": score_b_run,
                        "maps_downloaded": len(rows),
                    }
                )

    series_schema = {
        "series_id": pl.Int64,
        "leagueid": pl.Int64,
        "start_time_min_run": pl.Int64,
        "start_dt_run": pl.Datetime(time_zone="UTC"),
        "team_a_id_run": pl.Int64,
        "team_b_id_run": pl.Int64,
        "score_team_a_run": pl.Int64,
        "score_team_b_run": pl.Int64,
        "maps_downloaded": pl.Int64,
        # final series table fields (optional, but predeclare for stable CSV headers)
        "tournament_name": pl.Utf8,
        "tournament_slug": pl.Utf8,
        "tournament_tier": pl.Utf8,
        "tournament_location": pl.Utf8,
        "team_a": pl.Int64,
        "team_b": pl.Int64,
        "score_team_a": pl.Int64,
        "score_team_b": pl.Int64,
        "winner_team_id": pl.Int64,
        "start_time_min": pl.Int64,
        "start_time_max": pl.Int64,
        "team_a_name": pl.Utf8,
        "team_b_name": pl.Utf8,
        "winner_team_name": pl.Utf8,
    }
    series_df = pl.DataFrame(series_rows, strict=False, schema=series_schema)

    # If processed series table exists, attach final score + tournament metadata.
    if processed_dir is not None:
        series_path = processed_dir / "series.parquet"
        if series_path.exists() and not series_df.is_empty():
            proc_series = pl.read_parquet(series_path)
            join_cols = [c for c in ["series_id", "leagueid"] if c in proc_series.columns and c in series_df.columns]
            if join_cols:
                keep_cols = [c for c in proc_series.columns if c in {"series_id", "leagueid", "tournament_name", "tournament_slug", "tournament_tier", "tournament_location", "team_a", "team_b", "score_team_a", "score_team_b", "winner_team_id", "start_time_min", "start_time_max"}]
                proc_series = proc_series.select(keep_cols)
                series_df = series_df.join(proc_series, on=join_cols, how="left", suffix="_final")

            # Attach team names if possible from processed matches table
            matches_path = processed_dir / "matches.parquet"
            if matches_path.exists():
                cols = pl.scan_parquet(matches_path).collect_schema().names()
                m = pl.read_parquet(
                    matches_path,
                    columns=[c for c in ["radiant_team_id", "dire_team_id", "radiant_name", "dire_name"] if c in cols],
                )
                long = pl.concat(
                    [
                        m.select(pl.col("radiant_team_id").cast(pl.Int64, strict=False).alias("team_id"), pl.col("radiant_name").alias("name")),
                        m.select(pl.col("dire_team_id").cast(pl.Int64, strict=False).alias("team_id"), pl.col("dire_name").alias("name")),
                    ],
                    how="vertical",
                )
                name_map = (
                    long.filter(pl.col("team_id").is_not_null() & pl.col("name").is_not_null())
                    .group_by("team_id")
                    .agg(pl.first("name").alias("team_name"))
                )
                if "team_a" in series_df.columns:
                    joined = series_df.join(name_map, left_on="team_a", right_on="team_id", how="left")
                    if "team_name" in joined.columns:
                        joined = joined.rename({"team_name": "team_a_name_join"})
                        joined = joined.with_columns(pl.coalesce([pl.col("team_a_name_join"), pl.col("team_a_name")]).alias("team_a_name"))
                        drop_cols = [c for c in ["team_a_name_join", "team_id"] if c in joined.columns]
                        joined = joined.drop(drop_cols) if drop_cols else joined
                    else:
                        joined = joined.drop(["team_id"]) if "team_id" in joined.columns else joined
                    series_df = joined
                if "team_b" in series_df.columns:
                    joined = series_df.join(name_map, left_on="team_b", right_on="team_id", how="left")
                    if "team_name" in joined.columns:
                        joined = joined.rename({"team_name": "team_b_name_join"})
                        joined = joined.with_columns(pl.coalesce([pl.col("team_b_name_join"), pl.col("team_b_name")]).alias("team_b_name"))
                        drop_cols = [c for c in ["team_b_name_join", "team_id"] if c in joined.columns]
                        joined = joined.drop(drop_cols) if drop_cols else joined
                    else:
                        joined = joined.drop(["team_id"]) if "team_id" in joined.columns else joined
                    series_df = joined
                if "winner_team_id" in series_df.columns:
                    joined = series_df.join(name_map, left_on="winner_team_id", right_on="team_id", how="left")
                    if "team_name" in joined.columns:
                        joined = joined.rename({"team_name": "winner_team_name_join"})
                        joined = joined.with_columns(
                            pl.coalesce([pl.col("winner_team_name_join"), pl.col("winner_team_name")]).alias("winner_team_name")
                        )
                        drop_cols = [c for c in ["winner_team_name_join", "team_id"] if c in joined.columns]
                        joined = joined.drop(drop_cols) if drop_cols else joined
                    else:
                        joined = joined.drop(["team_id"]) if "team_id" in joined.columns else joined
                    series_df = joined

    series_path_out = run_dir / "series_trace.parquet"
    series_df.write_parquet(series_path_out)
    trace_paths["series_trace_parquet"] = str(series_path_out)
    # CSV for quick human inspection
    series_csv = run_dir / "series_trace.csv"
    series_df.select(
        [
            c
            for c in [
                "start_dt_run",
                "tournament_name",
                "team_a_name",
                "team_b_name",
                "score_team_a",
                "score_team_b",
                "winner_team_name",
                "series_id",
                "leagueid",
            ]
            if c in series_df.columns
        ]
    ).write_csv(series_csv)
    trace_paths["series_trace_csv"] = str(series_csv)

    return trace_paths


def healthcheck_run(
    run_dir: Path,
    *,
    processed_dir: Optional[Path] = None,
    write_report: bool = True,
) -> Dict[str, Any]:
    meta_path = run_dir / "run_metadata.json"
    errors_path = run_dir / "errors.json"

    meta = _read_json(meta_path) if meta_path.exists() else {}
    errors = _read_json(errors_path) if errors_path.exists() else []

    chunk_files = _run_chunk_files(run_dir)
    raw = load_raw_matches_from_files(chunk_files)

    match_ids: List[int] = []
    missing_match_id = 0
    has_gold_adv = has_xp_adv = has_picks_bans = 0
    for entry in raw:
        match = entry.get("json") if isinstance(entry, dict) and "json" in entry else entry
        if not isinstance(match, dict):
            continue
        mid = match.get("match_id")
        if isinstance(mid, int):
            match_ids.append(mid)
        else:
            missing_match_id += 1
        gold = match.get("radiant_gold_adv")
        xp = match.get("radiant_xp_adv")
        pb = match.get("picks_bans")
        if isinstance(gold, list) and len(gold) > 0:
            has_gold_adv += 1
        if isinstance(xp, list) and len(xp) > 0:
            has_xp_adv += 1
        if isinstance(pb, list) and len(pb) > 0:
            has_picks_bans += 1

    unique_ids = set(match_ids)
    duplicates = len(match_ids) - len(unique_ids)

    expected_ids = meta.get("new_match_ids_selected") or meta.get("new_match_ids") or []
    expected_set = {int(v) for v in expected_ids if isinstance(v, int) or (isinstance(v, str) and v.isdigit())}
    expected_found = len(expected_set & unique_ids) if expected_set else None
    expected_missing = len(expected_set - unique_ids) if expected_set else None

    processed_missing = None
    processed_found = None
    if processed_dir is not None:
        processed_ids = _existing_match_ids(processed_dir)
        processed_found = len(unique_ids & processed_ids)
        processed_missing = len(unique_ids - processed_ids)

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "chunk_files": len(chunk_files),
        "raw_matches": len(raw),
        "match_ids_total": len(match_ids),
        "match_ids_unique": len(unique_ids),
        "match_ids_duplicates": duplicates,
        "entries_missing_match_id": missing_match_id,
        "errors_count": len(errors) if isinstance(errors, list) else None,
        "expected_selected_count": len(expected_set) if expected_set else None,
        "expected_found_in_chunks": expected_found,
        "expected_missing_in_chunks": expected_missing,
        "gold_adv_present": has_gold_adv,
        "xp_adv_present": has_xp_adv,
        "picks_bans_present": has_picks_bans,
        "processed_dir": str(processed_dir) if processed_dir is not None else None,
        "processed_found": processed_found,
        "processed_missing": processed_missing,
    }

    if write_report:
        trace_paths = write_run_traces(run_dir, raw=raw, processed_dir=processed_dir)
        report["trace_paths"] = trace_paths
        write_json(report, run_dir / "healthcheck.json")
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print(f"[healthcheck] run_dir={report.get('run_dir')}")
    print(
        "[healthcheck] chunks={chunks} raw_matches={raw} unique_match_ids={uniq} dup={dup} errors={err}".format(
            chunks=report.get("chunk_files"),
            raw=report.get("raw_matches"),
            uniq=report.get("match_ids_unique"),
            dup=report.get("match_ids_duplicates"),
            err=report.get("errors_count"),
        )
    )
    sel = report.get("expected_selected_count")
    if sel is not None:
        print(
            "[healthcheck] expected_selected={sel} found_in_chunks={found} missing_in_chunks={miss}".format(
                sel=sel,
                found=report.get("expected_found_in_chunks"),
                miss=report.get("expected_missing_in_chunks"),
            )
        )
    print(
        "[healthcheck] gold_adv={g}/{n} xp_adv={x}/{n} picks_bans={pb}/{n}".format(
            g=report.get("gold_adv_present"),
            x=report.get("xp_adv_present"),
            pb=report.get("picks_bans_present"),
            n=report.get("raw_matches") or 0,
        )
    )
    if report.get("processed_dir") is not None:
        print(
            "[healthcheck] processed_found={f} processed_missing={m}".format(
                f=report.get("processed_found"),
                m=report.get("processed_missing"),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks for incremental raw runs (chunks).")
    parser.add_argument("--run-dir", default=None, help="Path to a run directory (data/raw/updates/run_...).")
    parser.add_argument("--updates-dir", default="data/raw/updates", help="Root updates dir; used to pick latest run if --run-dir omitted.")
    parser.add_argument("--processed", default=None, help="Optional processed parquet dir to verify match_ids presence.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(Path(args.updates_dir))
    if run_dir is None:
        raise SystemExit("No run directory found. Provide --run-dir or ensure updates dir exists.")
    report = healthcheck_run(run_dir, processed_dir=Path(args.processed) if args.processed else None, write_report=True)
    _print_report(report)
