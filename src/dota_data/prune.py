from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set

import polars as pl

from .io import infer_series_fields, series_summary_table


def _parse_min_date(val: str) -> int:
    dt = datetime.strptime(val, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _day_start(ts: int) -> int:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(dt.timestamp())


def _max_start_time(processed_dir: Path) -> Optional[int]:
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
        return int(max_ts)
    except Exception:
        return None


def prune_processed(
    *,
    processed_dir: Path,
    out_dir: Path,
    min_start_time: int,
) -> Dict[str, str]:
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches_path = processed_dir / "matches.parquet"
    if not matches_path.exists():
        raise FileNotFoundError(f"Missing {matches_path}")
    matches = pl.read_parquet(matches_path)
    if "match_id" not in matches.columns or "start_time" not in matches.columns:
        raise ValueError("matches.parquet must contain match_id and start_time")

    matches_new = matches.filter(pl.col("start_time").is_not_null() & (pl.col("start_time") >= min_start_time))
    match_ids: Set[int] = set(int(v) for v in matches_new["match_id"].drop_nulls().to_list() if v is not None)

    def filter_table(name: str) -> Optional[pl.DataFrame]:
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            return None
        df = pl.read_parquet(path)
        if "match_id" not in df.columns:
            return df
        return df.filter(pl.col("match_id").is_in(list(match_ids)))

    players = filter_table("players")
    objectives = filter_table("objectives")
    teamfights = filter_table("teamfights")
    extras = filter_table("extras")

    matches_new = infer_series_fields(matches_new)
    series = series_summary_table(matches_new)

    out_paths: Dict[str, str] = {}
    (out_dir / "matches.parquet").write_bytes(b"")  # ensure writable early
    (out_dir / "matches.parquet").unlink()
    matches_new.write_parquet(out_dir / "matches.parquet")
    out_paths["matches"] = str(out_dir / "matches.parquet")
    if players is not None:
        players.write_parquet(out_dir / "players.parquet")
        out_paths["players"] = str(out_dir / "players.parquet")
    if objectives is not None:
        objectives.write_parquet(out_dir / "objectives.parquet")
        out_paths["objectives"] = str(out_dir / "objectives.parquet")
    if teamfights is not None:
        teamfights.write_parquet(out_dir / "teamfights.parquet")
        out_paths["teamfights"] = str(out_dir / "teamfights.parquet")
    if extras is not None:
        extras.write_parquet(out_dir / "extras.parquet")
        out_paths["extras"] = str(out_dir / "extras.parquet")
    series.write_parquet(out_dir / "series.parquet")
    out_paths["series"] = str(out_dir / "series.parquet")
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune processed parquet to a time window.")
    parser.add_argument("--processed", default="data/processed", help="Input processed directory.")
    parser.add_argument("--out", default="data/processed_pruned", help="Output directory for pruned parquet tables.")
    parser.add_argument("--min-date", default=None, help="Keep matches with start_time >= this UTC date (YYYY-MM-DD).")
    parser.add_argument("--keep-days", type=int, default=None, help="Keep last N days based on max(start_time).")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    max_ts = _max_start_time(processed_dir)
    if args.min_date:
        min_ts = _parse_min_date(args.min_date)
    elif args.keep_days is not None:
        if max_ts is None:
            raise SystemExit("Cannot use --keep-days without matches.parquet start_time.")
        min_ts = _day_start(max_ts - int(args.keep_days) * 86400)
    else:
        raise SystemExit("Provide --min-date YYYY-MM-DD or --keep-days N")

    paths = prune_processed(processed_dir=processed_dir, out_dir=Path(args.out), min_start_time=min_ts)
    print({"min_start_time": min_ts, "out_paths": paths})
