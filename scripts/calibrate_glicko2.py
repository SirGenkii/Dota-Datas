from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dota_data import read_processed_tables
from src.dota_data.ratings import Glicko2Config, build_series_results, evaluate_glicko2_series

SERIES_OVERRIDE_PATH = Path("data/mappings/series_overrides.csv")


def apply_series_overrides(matches: pl.DataFrame, overrides_path: Path = SERIES_OVERRIDE_PATH) -> pl.DataFrame:
    """
    Apply manual overrides for specific (leagueid, series_id).
    Kept in sync with scripts/precompute_metrics.py.
    """
    if not overrides_path.exists():
        return matches
    overrides = pl.read_csv(overrides_path)
    if overrides.is_empty():
        return matches
    needed = {"leagueid", "series_id"}
    if not needed.issubset(overrides.columns):
        return matches
    ov_cols = [
        c
        for c in overrides.columns
        if c
        in {
            "leagueid",
            "series_id",
            "override_bo_type",
            "override_series_type",
            "override_tournament_tier",
            "override_tournament_location",
        }
    ]
    ov = overrides.select(ov_cols)
    base = matches
    for col in ["bo_type", "series_type_raw", "series_type"]:
        if col not in base.columns:
            base = base.with_columns(pl.lit(None).alias(col))
    joined = base.join(ov, on=["leagueid", "series_id"], how="left")
    base = joined.drop(["series_type"]) if "series_type" in joined.columns else joined
    out = base.with_columns(
        pl.coalesce([pl.col("override_bo_type"), pl.col("override_series_type"), pl.col("bo_type")]).alias("bo_type"),
        pl.coalesce([pl.col("override_series_type"), pl.col("series_type_raw"), pl.col("bo_type")]).alias("series_type_tmp"),
        pl.coalesce([pl.col("override_tournament_tier"), pl.col("tournament_tier")]).alias("tournament_tier"),
        pl.coalesce([pl.col("override_tournament_location"), pl.col("tournament_location")]).alias("tournament_location"),
    )
    out = out.with_columns(pl.coalesce([pl.col("bo_type"), pl.col("series_type_tmp")]).alias("series_type"))
    out = out.with_columns(
        pl.col("bo_type").cast(pl.Int64, strict=False),
        pl.col("series_type").cast(pl.Int64, strict=False),
    )
    drop_cols = [
        c
        for c in [
            "override_bo_type",
            "override_series_type",
            "override_tournament_tier",
            "override_tournament_location",
            "series_type_tmp",
        ]
        if c in out.columns
    ]
    return out.drop(drop_cols)


def _parse_floats(xs: Sequence[str]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if x is None:
            continue
        for part in str(x).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
    return out


def _grid(configs: Dict[str, List]) -> List[Dict[str, object]]:
    keys = list(configs.keys())
    if not keys:
        return [{}]
    combos: List[Dict[str, object]] = [{}]
    for k in keys:
        nxt: List[Dict[str, object]] = []
        for base in combos:
            for v in configs[k]:
                row = dict(base)
                row[k] = v
                nxt.append(row)
        combos = nxt
    return combos


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-search Glicko-2 params on series outcomes.")
    p.add_argument("--processed", default="data/processed", help="Processed parquet directory.")
    p.add_argument("--teams", default="data/teams_to_look.csv", help="Tracked teams CSV (TeamID).")
    p.add_argument("--out", default="data/metrics/glicko2_calibration.json", help="Output JSON with best params.")
    p.add_argument("--scope", choices=["all", "tracked_v_any", "tracked_only"], default="tracked_v_any")
    p.add_argument("--warmup-frac", type=float, default=0.2, help="Fraction of earliest series ignored for scoring.")
    p.add_argument("--periods", nargs="*", default=["day", "week"], choices=["day", "week", "series"])
    p.add_argument("--taus", nargs="*", default=["0.3", "0.5", "0.8"])
    p.add_argument("--init-rds", nargs="*", default=["300", "350"])
    p.add_argument("--init-sigmas", nargs="*", default=["0.06"])
    p.add_argument("--score-rd-mults", nargs="*", default=["2.0"])
    args = p.parse_args()

    processed_dir = Path(args.processed)
    tables = read_processed_tables(processed_dir)
    matches = tables["matches"]
    matches = apply_series_overrides(matches, overrides_path=SERIES_OVERRIDE_PATH)

    teams_csv = pl.read_csv(args.teams)
    teams_csv = teams_csv.rename({c: c.strip() for c in teams_csv.columns})
    tracked = set(int(x) for x in teams_csv["TeamID"].to_list() if x is not None)

    series_results = build_series_results(matches)
    if series_results.is_empty():
        raise RuntimeError("No series results could be built from matches.")

    if args.scope == "tracked_only":
        series_results = series_results.filter(pl.col("team_a_id").is_in(list(tracked)) & pl.col("team_b_id").is_in(list(tracked)))
    elif args.scope == "tracked_v_any":
        series_results = series_results.filter(pl.col("team_a_id").is_in(list(tracked)) | pl.col("team_b_id").is_in(list(tracked)))

    if series_results.is_empty():
        raise RuntimeError(f"No series left after scope={args.scope}.")

    grid = _grid(
        {
            "period": list(args.periods),
            "tau": _parse_floats(args.taus),
            "init_rd": _parse_floats(args.init_rds),
            "init_sigma": _parse_floats(args.init_sigmas),
            "score_rd_mult": _parse_floats(args.score_rd_mults),
        }
    )

    best = None
    results: List[Dict[str, object]] = []
    for g in grid:
        cfg = Glicko2Config(
            base_rating=1500.0,
            init_rd=float(g["init_rd"]),
            init_sigma=float(g["init_sigma"]),
            tau=float(g["tau"]),
            period=str(g["period"]),
            score_rd_mult=float(g["score_rd_mult"]),
        )
        metrics = evaluate_glicko2_series(series_results, config=cfg, warmup_frac=float(args.warmup_frac))
        row = {**asdict(cfg), **metrics}
        results.append(row)
        if best is None or (row.get("logloss") is not None and row["logloss"] < best["logloss"]):
            best = row

    results_sorted = sorted(results, key=lambda r: float(r.get("logloss") or 1e18))
    print("Top configs (by logloss):")
    for r in results_sorted[:10]:
        print(
            f"- period={r['period']} tau={r['tau']} init_rd={r['init_rd']} init_sigma={r['init_sigma']} "
            f"logloss={r['logloss']:.6f} brier={r['brier']:.6f} scored={int(r.get('scored', 0))}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": args.scope,
        "warmup_frac": float(args.warmup_frac),
        "best": best,
        "candidates": results_sorted,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
