from __future__ import annotations

import argparse
import json
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
    cols = lf.collect_schema().names()
    if "match_id" not in cols:
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
    cols = set(lf.collect_schema().names())
    if not needed.issubset(cols):
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


def _global_last_day_start(processed_dir: Path) -> Optional[int]:
    """
    Return the UTC day start timestamp of the latest match currently stored in processed data.
    Used to enforce an incremental window and prevent downloading historical matches.
    """
    matches_path = processed_dir / "matches.parquet"
    if not matches_path.exists():
        return None
    lf = pl.scan_parquet(matches_path)
    cols = lf.collect_schema().names()
    if "start_time" not in cols:
        return None
    max_ts = lf.select(pl.col("start_time").max()).collect().item()
    if max_ts is None:
        return None
    try:
        max_ts_int = int(max_ts)
    except Exception:
        return None
    day_start = datetime.fromtimestamp(max_ts_int, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(day_start.timestamp())


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
    timeout_team_matches: int,
    retries_team_matches: int,
    backoff_team_matches: float,
) -> List[Tuple[int, int]]:
    new_rows: List[Tuple[int, int]] = []
    # When a global since_start_time is provided, enforce that window (and allow missing matches
    # earlier in the same day). Otherwise fall back to team cursor.
    stop_time = int(since_start_time) if since_start_time is not None else int(team.last_start_time)
    for page in range(max_pages):
        offset = page * limit
        page_rows: List[Dict[str, Any]] = []
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, retries_team_matches + 1)):
            try:
                page_rows = fetch_team_matches(
                    team.team_id, session=session, limit=limit, offset=offset, timeout=timeout_team_matches
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries_team_matches:
                    time.sleep(max(0.0, backoff_team_matches) * (2**attempt))
        if last_exc is not None and not page_rows:
            # Give up on this team; continue with others.
            break
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
            new_rows.append((mid, int(st) if isinstance(st, int) else 0))
        if stop_time and page_times and min(page_times) <= stop_time:
            break
        if sleep_pages and sleep_pages > 0:
            time.sleep(sleep_pages)
    return new_rows


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
    timeout_team_matches: int = 60,
    retries_team_matches: int = 2,
    backoff_team_matches: float = 1.5,
) -> Tuple[List[int], Dict[str, Any], Dict[int, int]]:
    per_team: Dict[int, List[Tuple[int, int]]] = {}
    match_start_times: Dict[int, int] = {}
    team_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for t in teams:
        try:
            team_id = int(t.get("TeamID") or t.get("team_id") or t.get("id") or t.get("_source_team_id"))
        except Exception:
            continue
        team_name = str(t.get("TeamName") or t.get("name") or "")
        cursor = TeamCursor(team_id=team_id, team_name=team_name, last_start_time=int(last_start_time_by_team.get(team_id, 0) or 0))
        team_error = None
        try:
            ids = _new_match_ids_for_team(
                cursor,
                session=session,
                existing_ids=existing_ids,
                limit=limit,
                max_pages=max_pages,
                since_start_time=since_start_time,
                sleep_pages=sleep_pages,
                timeout_team_matches=timeout_team_matches,
                retries_team_matches=retries_team_matches,
                backoff_team_matches=backoff_team_matches,
            )
        except Exception as exc:  # noqa: BLE001
            ids = []
            team_error = str(exc)
        if team_error:
            errors.append({"team_id": team_id, "team_name": team_name, "error": team_error})
        per_team[team_id] = ids
        team_rows.append(
            {
                "team_id": team_id,
                "team_name": team_name,
                "last_start_time": int(cursor.last_start_time),
                "new_matches_found": len(ids),
            }
        )

    flat: List[int] = []
    for _, rows in per_team.items():
        for mid, st in rows:
            flat.append(mid)
            if mid not in match_start_times or st > match_start_times[mid]:
                match_start_times[mid] = st
    deduped = list(dict.fromkeys(flat))
    summary = {
        "teams": len(per_team),
        "per_team_new_counts": {str(k): len(v) for k, v in per_team.items()},
        "new_match_ids_total": len(deduped),
        "teams_failed": len(errors),
        "teams_discovery_rows": team_rows,
        "teams_discovery_errors": errors,
    }
    return deduped, summary, match_start_times


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
    max_new: Optional[int] = None,
    timeout_team_matches: int = 60,
    retries_team_matches: int = 2,
    backoff_team_matches: float = 1.5,
    allow_older_than_latest_day: bool = False,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    teams = load_team_list(teams_csv)
    team_ids = [int(t["TeamID"]) for t in teams if "TeamID" in t]

    existing_ids = _existing_match_ids(processed_dir)
    last_times = _last_start_time_by_team(processed_dir, team_ids=team_ids)
    since_ts_user = _parse_since(since)
    latest_day_start = _global_last_day_start(processed_dir)
    since_ts_effective = since_ts_user
    if not allow_older_than_latest_day and latest_day_start is not None:
        since_ts_effective = max(since_ts_effective or 0, latest_day_start)

    api_key = load_api_key(env_var=api_key_env, load_env_file=True)
    session = build_session(api_key=api_key)

    new_ids, discover_summary, match_start_times = discover_new_match_ids(
        teams,
        session=session,
        existing_ids=existing_ids,
        last_start_time_by_team=last_times,
        limit=limit,
        max_pages=max_pages,
        since_start_time=since_ts_effective,
        sleep_pages=sleep_pages,
        timeout_team_matches=timeout_team_matches,
        retries_team_matches=retries_team_matches,
        backoff_team_matches=backoff_team_matches,
    )

    new_ids_discovered = list(new_ids)
    if max_new is not None and max_new > 0 and len(new_ids) > max_new:
        new_ids = sorted(new_ids, key=lambda mid: (match_start_times.get(int(mid), 0), int(mid)), reverse=True)[:max_new]

    discover_seconds = time.monotonic() - t0

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
        "since_ts_user": since_ts_user,
        "since_ts_effective": since_ts_effective,
        "latest_day_start_ts": latest_day_start,
        "allow_older_than_latest_day": allow_older_than_latest_day,
        "chunk_size": chunk_size,
        "sleep_match_detail": sleep_match_detail,
        "sleep_pages": sleep_pages,
        "timeout": timeout,
        "existing_match_ids": len(existing_ids),
        **discover_summary,
        "max_new": max_new,
        "new_match_ids_discovered": new_ids_discovered,
        "new_match_ids_selected": new_ids,
        "new_match_start_time_max": max(match_start_times.values()) if match_start_times else None,
        "discover_seconds": discover_seconds,
    }
    write_json(meta, run_dir / "run_metadata.json")
    # Write team-level discovery info for easy inspection.
    try:
        pl.DataFrame(discover_summary.get("teams_discovery_rows") or [], strict=False).write_parquet(run_dir / "teams_discovery.parquet")
    except Exception:
        pass
    if discover_summary.get("teams_discovery_errors"):
        write_json(discover_summary["teams_discovery_errors"], run_dir / "teams_discovery_errors.json")

    if dry_run or not new_ids:
        elapsed_seconds = time.monotonic() - t0
        write_json(
            {
                "run_id": run_id,
                "elapsed_seconds": elapsed_seconds,
                "discover_seconds": discover_seconds,
                "download_seconds": 0.0,
                "downloaded": 0,
            },
            run_dir / "run_result.json",
        )
        return {
            "run_dir": str(run_dir),
            "run_metadata_path": str(run_dir / "run_metadata.json"),
            "run_result_path": str(run_dir / "run_result.json"),
            "downloaded": 0,
            "new_match_ids": len(new_ids),
            "dry_run": dry_run,
        }

    t_dl = time.monotonic()
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
    download_seconds = time.monotonic() - t_dl
    elapsed_seconds = time.monotonic() - t0
    write_json(
        {
            "run_id": run_id,
            "elapsed_seconds": elapsed_seconds,
            "discover_seconds": discover_seconds,
            "download_seconds": download_seconds,
            "download_summary": summary,
        },
        run_dir / "run_result.json",
    )
    return {
        "run_dir": str(run_dir),
        "run_metadata_path": str(run_dir / "run_metadata.json"),
        "run_result_path": str(run_dir / "run_result.json"),
        "downloaded": summary.get("total_ids", 0),
        "summary": summary,
    }


def _raw_chunk_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("matches_chunk*.json"))


def apply_raw_run_to_processed(
    *,
    run_dir: Path,
    processed_dir: Path,
    aliases: Optional[Path] = None,
) -> Dict[str, Any]:
    def dataset_stats(processed_dir: Path) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"processed_dir": str(processed_dir)}
        matches_path = processed_dir / "matches.parquet"
        if matches_path.exists():
            lf = pl.scan_parquet(matches_path)
            cols = lf.collect_schema().names()
            if "match_id" in cols:
                stats["matches_unique"] = int(lf.select(pl.col("match_id").n_unique()).collect().item())
                stats["matches_rows"] = int(lf.select(pl.len()).collect().item())
            if "start_time" in cols:
                stats["start_time_max"] = lf.select(pl.col("start_time").max()).collect().item()
            if "series_id" in cols and "leagueid" in cols:
                stats["series_pairs_unique"] = int(
                    lf.filter(pl.col("series_id").is_not_null() & pl.col("leagueid").is_not_null())
                    .select(pl.struct(["series_id", "leagueid"]).n_unique())
                    .collect()
                    .item()
                )
        series_path = processed_dir / "series.parquet"
        if series_path.exists():
            lf_s = pl.scan_parquet(series_path)
            stats["series_rows"] = int(lf_s.select(pl.len()).collect().item())
        extras_path = processed_dir / "extras.parquet"
        if extras_path.exists():
            lf_e = pl.scan_parquet(extras_path)
            cols_e = lf_e.collect_schema().names()
            if "match_id" in cols_e:
                stats["extras_unique"] = int(lf_e.select(pl.col("match_id").n_unique()).collect().item())
        return stats

    raw_files = _raw_chunk_files(run_dir)
    raw = load_raw_matches_from_files(raw_files)

    # Safety: enforce the run's effective since window when applying, to avoid accidentally
    # appending historical matches from a run_dir.
    filtered_out = 0
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        since_ts_effective = meta.get("since_ts_effective")
        allow_older = bool(meta.get("allow_older_than_latest_day"))
        if not allow_older and isinstance(since_ts_effective, int):
            kept = []
            for entry in raw:
                match = entry.get("json") if isinstance(entry, dict) and "json" in entry else None
                if not isinstance(match, dict):
                    filtered_out += 1
                    continue
                st = match.get("start_time")
                if isinstance(st, int) and st >= since_ts_effective:
                    kept.append(entry)
                else:
                    filtered_out += 1
            raw = kept

    before = dataset_stats(processed_dir)
    paths = append_parquet_tables(raw, processed_dir, alias_path=aliases)
    after = dataset_stats(processed_dir)

    write_json(before, run_dir / "dataset_stats_before.json")
    write_json(after, run_dir / "dataset_stats_after.json")
    delta = {
        "matches_unique_delta": (after.get("matches_unique", 0) - before.get("matches_unique", 0))
        if after.get("matches_unique") is not None and before.get("matches_unique") is not None
        else None,
        "series_rows_delta": (after.get("series_rows", 0) - before.get("series_rows", 0))
        if after.get("series_rows") is not None and before.get("series_rows") is not None
        else None,
        "series_pairs_unique_delta": (after.get("series_pairs_unique", 0) - before.get("series_pairs_unique", 0))
        if after.get("series_pairs_unique") is not None and before.get("series_pairs_unique") is not None
        else None,
    }
    write_json(delta, run_dir / "dataset_stats_delta.json")
    return {
        "run_dir": str(run_dir),
        "raw_files": len(raw_files),
        "raw_matches": len(raw),
        "raw_filtered_out": filtered_out,
        "processed_paths": {k: str(v) for k, v in paths.items()},
        "dataset_stats_before": before,
        "dataset_stats_after": after,
        "dataset_stats_delta": delta,
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
    parser.add_argument("--timeout-team-matches", type=int, default=60, help="HTTP timeout for /teams/{id}/matches discovery calls.")
    parser.add_argument("--retries-team-matches", type=int, default=2, help="Retries for /teams/{id}/matches discovery calls.")
    parser.add_argument("--backoff-team-matches", type=float, default=1.5, help="Backoff base (seconds) for discovery retries.")
    parser.add_argument("--allow-older-than-latest-day", action="store_true", help="Allow downloading matches older than the latest day present in processed data.")
    parser.add_argument("--max-new", type=int, default=None, help="Cap the number of new matches to download (newest first).")
    parser.add_argument("--dry-run", action="store_true", help="Only discover new match_ids; do not download match details.")
    parser.add_argument("--apply-parquet", action="store_true", help="After download, append the batch to processed parquet tables.")
    args = parser.parse_args()

    t0 = time.monotonic()
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
        max_new=args.max_new,
        timeout_team_matches=args.timeout_team_matches,
        retries_team_matches=args.retries_team_matches,
        backoff_team_matches=args.backoff_team_matches,
        allow_older_than_latest_day=args.allow_older_than_latest_day,
    )
    run_dir = Path(result["run_dir"])
    elapsed = time.monotonic() - t0
    print(f"[sync] run_dir={run_dir} dry_run={bool(args.dry_run)} elapsed={elapsed:.1f}s")
    if args.dry_run:
        print(f"[sync] new_match_ids_selected={result.get('new_match_ids')}")
    elif "summary" not in result:
        print(f"[sync] new_match_ids_selected={result.get('new_match_ids')}")
        print("[sync] nothing to download")
    else:
        summary = result.get("summary") or {}
        print(
            "[sync] downloaded={d} chunks_written={cw} errors_remaining={er}".format(
                d=summary.get("total_ids"),
                cw=summary.get("chunks_written"),
                er=summary.get("errors_remaining"),
            )
        )
        try:
            from .healthcheck import healthcheck_run  # noqa: PLC0415

            report = healthcheck_run(run_dir, processed_dir=None, write_report=True)
            print(
                "[sync] healthcheck unique_match_ids={u} gold_adv={g}/{n} xp_adv={x}/{n}".format(
                    u=report.get("match_ids_unique"),
                    g=report.get("gold_adv_present"),
                    x=report.get("xp_adv_present"),
                    n=report.get("raw_matches") or 0,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sync] healthcheck failed: {exc}")

    if args.apply_parquet and not args.dry_run:
        applied = apply_raw_run_to_processed(
            run_dir=Path(result["run_dir"]),
            processed_dir=Path(args.processed),
            aliases=Path(args.aliases) if args.aliases else None,
        )
        print(f"[sync] parquet appended: {applied.get('processed_paths')}")
        delta = applied.get("dataset_stats_delta") or {}
        print(
            "[sync] dataset delta matches={m} series_rows={s} series_pairs={sp}".format(
                m=delta.get("matches_unique_delta"),
                s=delta.get("series_rows_delta"),
                sp=delta.get("series_pairs_unique_delta"),
            )
        )
        try:
            from .healthcheck import healthcheck_run  # noqa: PLC0415

            report = healthcheck_run(run_dir, processed_dir=Path(args.processed), write_report=True)
            print(
                "[sync] processed_found={f} processed_missing={m}".format(
                    f=report.get("processed_found"),
                    m=report.get("processed_missing"),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sync] processed healthcheck failed: {exc}")
