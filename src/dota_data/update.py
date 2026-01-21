from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import polars as pl

from .api import (
    build_session,
    fetch_matches_chunked,
    fetch_team_matches,
    load_api_key,
    load_team_list,
    write_json,
)
from .io import append_parquet_tables, load_raw_matches_from_files


def _scan_parquet_any(path: Path) -> pl.LazyFrame:
    if path.exists() and path.is_dir():
        return pl.scan_parquet(str(path / "**/*.parquet"))
    return pl.scan_parquet(path)


def _existing_match_ids(processed_dir: Path) -> Set[int]:
    file_path = processed_dir / "matches.parquet"
    dir_path = processed_dir / "matches"
    if file_path.exists():
        lf = _scan_parquet_any(file_path)
    elif dir_path.exists():
        lf = _scan_parquet_any(dir_path)
    else:
        return set()
    if "match_id" not in lf.columns:
        return set()
    df = lf.select(pl.col("match_id").cast(pl.Int64, strict=False)).collect()
    return {int(v) for v in df["match_id"].drop_nulls().to_list() if v is not None}


def _last_start_time_by_team(processed_dir: Path, team_ids: Sequence[int]) -> Dict[int, int]:
    file_path = processed_dir / "matches.parquet"
    dir_path = processed_dir / "matches"
    if file_path.exists():
        lf = pl.scan_parquet(file_path)
    elif dir_path.exists():
        lf = pl.scan_parquet(str(dir_path / "**/*.parquet"))
    else:
        return {int(t): 0 for t in team_ids}

    needed = {"start_time", "radiant_team_id", "dire_team_id"}
    if not needed.issubset(set(lf.columns)):
        return {int(t): 0 for t in team_ids}

    rows = lf.select(["start_time", "radiant_team_id", "dire_team_id"]).collect()
    long = pl.concat(
        [
            rows.select(pl.col("radiant_team_id").cast(pl.Int64, strict=False).alias("team_id"), pl.col("start_time")),
            rows.select(pl.col("dire_team_id").cast(pl.Int64, strict=False).alias("team_id"), pl.col("start_time")),
        ],
        how="vertical",
    )
    long = long.filter(pl.col("team_id").is_in([int(t) for t in team_ids]) & pl.col("start_time").is_not_null())
    if long.is_empty():
        return {int(t): 0 for t in team_ids}
    agg = long.group_by("team_id").agg(pl.col("start_time").max().alias("last_start_time"))
    out = {int(r["team_id"]): int(r["last_start_time"] or 0) for r in agg.iter_rows(named=True)}
    for tid in team_ids:
        out.setdefault(int(tid), 0)
    return out


def _parse_since(since: Optional[str]) -> Optional[int]:
    if since is None or not str(since).strip():
        return None
    s = str(since).strip()
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            continue
    raise ValueError(f"Invalid --since value: {since!r}. Use unix timestamp or YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.")


@dataclass(frozen=True)
class TeamCursor:
    team_id: int
    team_name: str
    last_start_time: int


def _new_match_ids_for_team(
    team: TeamCursor,
    session: Any,
    existing_ids: Set[int],
    limit: int,
    max_pages: int,
    since_start_time: Optional[int],
    sleep_pages: float,
) -> List[int]:
    new_ids: List[int] = []
    stop_time = max(team.last_start_time, since_start_time or 0)
    for page in range(max_pages):
        offset = page * limit
        page_rows = fetch_team_matches(team.team_id, session=session, limit=limit, offset=offset)
        if not page_rows:
            break
        page_times = [m.get("start_time") for m in page_rows if isinstance(m, dict) and isinstance(m.get("start_time"), int)]
        for m in page_rows:
            if not isinstance(m, dict):
                continue
            mid = m.get("match_id")
            if not isinstance(mid, int):
                continue
            if mid in existing_ids:
                continue
            st = m.get("start_time")
            if isinstance(st, int) and st <= stop_time:
                continue
            new_ids.append(mid)
        if stop_time and page_times and min(page_times) <= stop_time:
            break
        if sleep_pages and sleep_pages > 0:
            time.sleep(sleep_pages)
    return new_ids


def discover_new_match_ids(
    teams: Sequence[Dict[str, Any]],
    session: Any,
    existing_ids: Set[int],
    last_start_time_by_team: Dict[int, int],
    *,
    limit: int = 100,
    max_pages: int = 3,
    since_start_time: Optional[int] = None,
    sleep_pages: float = 0.0,
) -> Tuple[List[int], Dict[str, Any]]:
    per_team: Dict[int, List[int]] = {}
    for t in teams:
        try:
            team_id = int(t.get("TeamID") or t.get("team_id") or t.get("id") or t.get("_source_team_id"))
        except Exception:
            continue
        team_name = str(t.get("TeamName") or t.get("name") or "")
        cursor = TeamCursor(team_id=team_id, team_name=team_name, last_start_time=int(last_start_time_by_team.get(team_id, 0) or 0))
        ids = _new_match_ids_for_team(
            cursor,
            session=session,
            existing_ids=existing_ids,
            limit=limit,
            max_pages=max_pages,
            since_start_time=since_start_time,
            sleep_pages=sleep_pages,
        )
        per_team[team_id] = ids

    flat = []
    for _, ids in per_team.items():
        flat.extend(ids)
    deduped = list(dict.fromkeys(flat))
    summary = {
        "teams": len(per_team),
        "per_team_new_counts": {str(k): len(v) for k, v in per_team.items()},
        "new_match_ids_total": len(deduped),
    }
    return deduped, summary


def download_new_matches(
    *,
    teams_csv: Path,
    processed_dir: Path,
    raw_updates_dir: Path,
    api_key_env: str = "OPENDOTA_KEY",
    limit: int = 100,
    max_pages: int = 3,
    since: Optional[str] = None,
    chunk_size: int = 100,
    sleep_match_detail: float = 1.0,
    sleep_pages: float = 0.0,
    timeout: int = 60,
    dry_run: bool = False,
) -> Dict[str, Any]:
    teams = load_team_list(teams_csv)
    team_ids = [int(t["TeamID"]) for t in teams if "TeamID" in t]

    existing_ids = _existing_match_ids(processed_dir)
    last_times = _last_start_time_by_team(processed_dir, team_ids=team_ids)
    since_ts = _parse_since(since)

    api_key = load_api_key(env_var=api_key_env, load_env_file=True)
    session = build_session(api_key=api_key)

    new_ids, discover_summary = discover_new_match_ids(
        teams,
        session=session,
        existing_ids=existing_ids,
        last_start_time_by_team=last_times,
        limit=limit,
        max_pages=max_pages,
        since_start_time=since_ts,
        sleep_pages=sleep_pages,
    )

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = raw_updates_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "teams_csv": str(teams_csv),
        "processed_dir": str(processed_dir),
        "raw_updates_dir": str(raw_updates_dir),
        "limit": limit,
        "max_pages": max_pages,
        "since": since,
        "chunk_size": chunk_size,
        "sleep_match_detail": sleep_match_detail,
        "sleep_pages": sleep_pages,
        "timeout": timeout,
        "existing_match_ids": len(existing_ids),
        **discover_summary,
        "new_match_ids": new_ids,
    }
    write_json(meta, run_dir / "run_metadata.json")

    if dry_run or not new_ids:
        return {"run_dir": str(run_dir), "downloaded": 0, "new_match_ids": len(new_ids), "dry_run": dry_run}

    summary = fetch_matches_chunked(
        new_ids,
        session=session,
        out_dir=run_dir,
        chunk_size=chunk_size,
        resume=False,
        sleep=sleep_match_detail,
        timeout=timeout,
        prefix="matches_chunk",
        retry_failed=True,
    )
    return {"run_dir": str(run_dir), "downloaded": summary.get("total_ids", 0), "summary": summary}


def _raw_chunk_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("matches_chunk*.json"))


def apply_raw_run_to_processed(
    *,
    run_dir: Path,
    processed_dir: Path,
    aliases: Optional[Path] = None,
) -> Dict[str, Any]:
    raw_files = _raw_chunk_files(run_dir)
    raw = load_raw_matches_from_files(raw_files)
    paths = append_parquet_tables(raw, processed_dir, alias_path=aliases)
    return {
        "run_dir": str(run_dir),
        "raw_files": len(raw_files),
        "raw_matches": len(raw),
        "processed_paths": {k: str(v) for k, v in paths.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download missing recent matches for tracked teams (incremental sync).")
    parser.add_argument("--teams", default="data/teams_to_look.csv", help="CSV of tracked teams (TeamName, TeamID).")
    parser.add_argument("--processed", default="data/processed", help="Processed parquet dir to detect existing matches.")
    parser.add_argument("--raw-updates", default="data/raw/updates", help="Directory where new raw chunks will be stored.")
    parser.add_argument("--aliases", default="data/team_aliases.csv", help="CSV mapping alias_team_id -> canonical_team_id.")
    parser.add_argument("--api-key-env", default="OPENDOTA_KEY", help="Env var holding OpenDota API key.")
    parser.add_argument("--limit", type=int, default=100, help="Per-page limit for /teams/{id}/matches.")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages per team to scan for new matches.")
    parser.add_argument("--since", default=None, help="Optional lower bound (unix ts or YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size for match detail download.")
    parser.add_argument("--sleep-match-detail", type=float, default=1.0, help="Sleep seconds between match detail requests.")
    parser.add_argument("--sleep-pages", type=float, default=0.0, help="Sleep seconds between team match pages.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout for match detail requests.")
    parser.add_argument("--dry-run", action="store_true", help="Only discover new match_ids; do not download match details.")
    parser.add_argument("--apply-parquet", action="store_true", help="After download, append the batch to processed parquet tables.")
    args = parser.parse_args()

    result = download_new_matches(
        teams_csv=Path(args.teams),
        processed_dir=Path(args.processed),
        raw_updates_dir=Path(args.raw_updates),
        api_key_env=args.api_key_env,
        limit=args.limit,
        max_pages=args.max_pages,
        since=args.since,
        chunk_size=args.chunk_size,
        sleep_match_detail=args.sleep_match_detail,
        sleep_pages=args.sleep_pages,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    print(result)

    if args.apply_parquet and not args.dry_run:
        applied = apply_raw_run_to_processed(
            run_dir=Path(result["run_dir"]),
            processed_dir=Path(args.processed),
            aliases=Path(args.aliases) if args.aliases else None,
        )
        print(applied)
