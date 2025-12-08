from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import polars as pl

from src.dota_data import (
    read_processed_tables,
    build_team_dictionary,
)


def load_lookup(path: Path) -> List[int]:
    """Load tracked team IDs from CSV."""
    df = pl.read_csv(path)
    return df["TeamID"].to_list()


def extract_team_id(row: Dict[str, object], team_ids: List[int]) -> Tuple[int | None, int | None, int | None]:
    """Return (tracked_team_id, opp_id, team_is_radiant) for matches involving tracked teams."""
    r_id = row.get("radiant_team_id")
    d_id = row.get("dire_team_id")
    r_tracked = r_id if r_id in team_ids else None
    d_tracked = d_id if d_id in team_ids else None
    tracked = r_tracked or d_tracked
    if tracked is None:
        return None, None, None
    opp = d_id if tracked == r_id else r_id
    team_is_radiant = tracked == r_id
    return tracked, opp, 1 if team_is_radiant else 0


def compute_elo(matches: pl.DataFrame, team_ids: List[int], k: float = 24.0, base_elo: float = 1500.0, side_adv: float = 25.0) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute Elo for tracked teams.
    - Only tracked teams are updated.
    - Opponents not tracked are assumed fixed at base_elo.
    - side_adv applies to Radiant (advantage).
    """
    ratings: Dict[int, float] = {tid: base_elo for tid in team_ids}
    history_rows = []

    # Sort matches by start_time
    matches_sorted = matches.sort("start_time")
    for row in matches_sorted.iter_rows(named=True):
        tracked_id, opp_id, team_is_radiant = extract_team_id(row, team_ids)
        if tracked_id is None:
            continue
        team_rating = ratings.get(tracked_id, base_elo)
        opp_rating = ratings.get(opp_id, base_elo) if opp_id is not None and opp_id in team_ids else base_elo

        # Determine score for tracked team
        radiant_win = row.get("radiant_win")
        if radiant_win is None:
            continue
        team_win = radiant_win if team_is_radiant else (1 - int(radiant_win))

        # Side adjustment: Radiant gets +side_adv
        adj = side_adv if team_is_radiant else -side_adv
        expected = 1 / (1 + 10 ** (((opp_rating - team_rating) - adj) / 400))
        new_rating = team_rating + k * (team_win - expected)

        history_rows.append(
            {
                "match_id": row.get("match_id"),
                "start_time": row.get("start_time"),
                "start_dt": datetime.fromtimestamp(row.get("start_time", 0), tz=timezone.utc),
                "team_id": tracked_id,
                "opponent_id": opp_id,
                "team_is_radiant": bool(team_is_radiant),
                "team_win": team_win,
                "rating_pre": team_rating,
                "rating_post": new_rating,
                "expected": expected,
            }
        )

        ratings[tracked_id] = new_rating

    hist_df = pl.DataFrame(history_rows, strict=False)
    latest_rows = [{"team_id": tid, "elo": elo} for tid, elo in ratings.items()]
    latest_df = pl.DataFrame(latest_rows, strict=False).sort("elo", descending=True)
    return hist_df, latest_df


def compute_firsts(matches: pl.DataFrame, objectives: pl.DataFrame, players: pl.DataFrame, tracked_ids: List[int]) -> pl.DataFrame:
    """
    Compute first blood / first tower / first roshan rates per tracked team.
    - First blood from players.firstblood_claimed (more reliable than objectives).
    - First tower from earliest building_kill.
    - First Roshan from first CHAT_MESSAGE_ROSHAN_KILL (team 2=radiant, 3=dire).
    """
    # first blood: players table
    fb = (
        players.filter(pl.col("firstblood_claimed") == 1)
        .select("match_id", "is_radiant")
        .group_by("match_id")
        .agg(pl.first("is_radiant").alias("fb_is_radiant"))
    )

    # first tower: building_kill earliest per match
    towers = objectives.filter(pl.col("type") == "building_kill").with_columns(
        pl.when(pl.col("key").str.contains("goodguys")).then(True)
        .when(pl.col("key").str.contains("badguys")).then(False)
        .otherwise(None)
        .alias("building_is_radiant")
    )
    first_tower = (
        towers.sort(["match_id", "time"])
        .group_by("match_id")
        .agg(pl.first("building_is_radiant").alias("building_is_radiant"))
    )

    # first roshan via CHAT_MESSAGE_ROSHAN_KILL team=2 (Radiant) / 3 (Dire)
    rosh = (
        objectives.filter(pl.col("type") == "CHAT_MESSAGE_ROSHAN_KILL")
        .sort(["match_id", "time"])
        .group_by("match_id")
        .agg(pl.first("team").alias("roshan_team"))
    )

    m = matches.select(
        "match_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
    ).join(fb, on="match_id", how="left").join(first_tower, on="match_id", how="left").join(rosh, on="match_id", how="left")

    rows = []
    for side in ("radiant", "dire"):
        side_bool = side == "radiant"
        team_col = f"{side}_team_id"
        team_rows = (
            m.with_columns(
                [
                    (pl.col(team_col)).alias("team_id"),
                    pl.lit(side_bool).alias("team_is_radiant"),
                ]
            )
            .filter(pl.col("team_id").is_in(tracked_ids))
        )
        for r in team_rows.iter_rows(named=True):
            team_id = r["team_id"]
            fb_hit = r["fb_is_radiant"] == side_bool if r.get("fb_is_radiant") is not None else None
            ft_hit = None
            if r.get("building_is_radiant") is not None:
                ft_hit = r["building_is_radiant"] != side_bool  # first tower belonged to opponent
            fr_hit = None
            if r.get("roshan_team") is not None:
                # roshan_team 2 ~ Radiant, 3 ~ Dire
                fr_hit = (r["roshan_team"] == 2 and side_bool) or (r["roshan_team"] == 3 and not side_bool)
            rows.append(
                {
                    "team_id": team_id,
                    "team_is_radiant": side_bool,
                    "first_blood": fb_hit,
                    "first_tower": ft_hit,
                    "first_roshan": fr_hit,
                }
            )
    df = pl.DataFrame(rows, strict=False)
    agg = (
        df.group_by(["team_id", "team_is_radiant"])
        .agg(
            pl.col("first_blood").mean().alias("first_blood_rate"),
            pl.col("first_tower").mean().alias("first_tower_rate"),
            pl.col("first_roshan").mean().alias("first_roshan_rate"),
            pl.sum("first_blood").alias("first_blood_count"),
            pl.sum("first_tower").alias("first_tower_count"),
            pl.sum("first_roshan").alias("first_roshan_count"),
            pl.len().alias("matches"),
        )
        .sort(["team_id", "team_is_radiant"])
    )
    return agg


def _objectives_for_match(objectives: pl.DataFrame, match_id: int) -> List[Dict[str, object]]:
    return objectives.filter(pl.col("match_id") == match_id).sort("time").to_dicts()


def compute_roshan_metrics(matches: pl.DataFrame, objectives: pl.DataFrame, tracked_ids: List[int]) -> pl.DataFrame:
    """
    Compute Roshan/Aegis metrics:
    - roshan_kills, aegis_claims, first_roshan, steals
    """
    # Pre-split objectives by match_id for faster access
    raw_groups = objectives.partition_by("match_id", as_dict=True, maintain_order=True)
    obj_groups = {}
    for key, df in raw_groups.items():
        k = key[0] if isinstance(key, tuple) else key
        try:
            k = int(k)
        except Exception:  # noqa: BLE001
            continue
        obj_groups[k] = df

    rows = []
    for row in matches.iter_rows(named=True):
        match_id = row.get("match_id")
        radiant_team_id = row.get("radiant_team_id")
        dire_team_id = row.get("dire_team_id")
        if match_id not in obj_groups:
            continue
        obj_list = obj_groups[match_id].to_dicts()

        # Roshan kills
        rosh_kills = [o for o in obj_list if o.get("type") == "CHAT_MESSAGE_ROSHAN_KILL"]
        rosh_sides = []
        for o in rosh_kills:
            team_val = o.get("team")
            if team_val == 2:
                rosh_sides.append("radiant")
            elif team_val == 3:
                rosh_sides.append("dire")

        # Aegis claims
        aegis_claims = [o for o in obj_list if o.get("type") == "CHAT_MESSAGE_AEGIS"]
        aegis_sides = []
        for o in aegis_claims:
            slot = o.get("slot")
            pslot = o.get("player_slot")
            if slot is None and pslot is None:
                continue
            is_radiant = False
            if slot is not None:
                is_radiant = slot < 5
            elif pslot is not None:
                is_radiant = pslot < 128
            aegis_sides.append("radiant" if is_radiant else "dire")

        # Pair roshan kill with next aegis for steal detection
        steals_rad = 0
        steals_dire = 0
        if rosh_kills and aegis_claims:
            rosh_iter = iter(rosh_kills)
            aeg_iter = iter(aegis_claims)
            current_aeg = next(aeg_iter, None)
            for rk in rosh_kills:
                while current_aeg is not None and current_aeg.get("time", 0) < rk.get("time", 0):
                    current_aeg = next(aeg_iter, None)
                if current_aeg is None:
                    break
                rk_side = "radiant" if rk.get("team") == 2 else "dire"
                aeg_side = "radiant" if (current_aeg.get("slot", 10) < 5 or (current_aeg.get("player_slot", 200) < 128)) else "dire"
                if rk_side != aeg_side:
                    if aeg_side == "radiant":
                        steals_rad += 1
                    else:
                        steals_dire += 1
                current_aeg = next(aeg_iter, None)

        first_rosh_side = None
        if rosh_sides:
            first_rosh_side = rosh_sides[0]

        for team_id, side in ((radiant_team_id, "radiant"), (dire_team_id, "dire")):
            if team_id not in tracked_ids:
                continue
            team_is_radiant = side == "radiant"
            rk = sum(1 for s in rosh_sides if (s == "radiant") == team_is_radiant)
            ac = sum(1 for s in aegis_sides if (s == "radiant") == team_is_radiant)
            first_rosh = None
            if first_rosh_side is not None:
                first_rosh = (first_rosh_side == "radiant") == team_is_radiant
            rows.append(
                {
                    "team_id": team_id,
                    "match_id": match_id,
                    "roshan_kills": rk,
                    "aegis_claims": ac,
                    "first_roshan": first_rosh,
                    "steals": steals_rad if team_is_radiant else steals_dire,
                    "steals_any": (steals_rad if team_is_radiant else steals_dire) > 0,
                }
            )

    if not rows:
        return pl.DataFrame(
            [], schema={"team_id": pl.Int64, "matches": pl.Int64, "roshan_kills_avg": pl.Float64, "aegis_claims_avg": pl.Float64, "first_roshan_rate": pl.Float64, "steals_total": pl.Int64, "steals_rate": pl.Float64}
        )
    df = pl.DataFrame(rows, strict=False)
    agg = (
        df.group_by("team_id")
        .agg(
            pl.len().alias("matches"),
            pl.col("roshan_kills").mean().alias("roshan_kills_avg"),
            pl.col("aegis_claims").mean().alias("aegis_claims_avg"),
            pl.col("first_roshan").mean().alias("first_roshan_rate"),
            pl.col("steals").sum().alias("steals_total"),
            pl.col("steals_any").mean().alias("steals_rate"),
        )
        .sort("team_id")
    )
    return agg


def _parse_adv(val: object) -> List[float]:
    if val is None:
        return []
    if isinstance(val, list):
        return [float(x) if x is not None else 0.0 for x in val]
    if isinstance(val, str):
        try:
            arr = json.loads(val)
            if isinstance(arr, list):
                return [float(x) if x is not None else 0.0 for x in arr]
        except json.JSONDecodeError:
            return []
    return []


def compute_adv_buckets(
    matches: pl.DataFrame,
    raw_map: Dict[int, dict],
    tracked_ids: List[int],
    key: str,
    minutes: Sequence[int] = (5, 10, 12, 15, 20),
) -> pl.DataFrame:
    """Compute winrate by bucket for a given advantage key (radiant_gold_adv or radiant_xp_adv)."""
    rows = []
    buckets = [-10_000, -5_000, -1_000, 0, 1_000, 5_000, 10_000, 999_999]

    def bucketize(val: float) -> str:
        for i in range(len(buckets) - 1):
            if buckets[i] <= val < buckets[i + 1]:
                return f"[{buckets[i]/1000:.0f}k,{buckets[i+1]/1000:.0f}k)"
        return f">={buckets[-2]/1000:.0f}k"

    for row in matches.iter_rows(named=True):
        match_id = row.get("match_id")
        rad_id = row.get("radiant_team_id")
        dire_id = row.get("dire_team_id")
        radiant_win = row.get("radiant_win")
        if radiant_win is None or match_id not in raw_map:
            continue
        raw_match = raw_map[match_id]
        adv_array = _parse_adv(raw_match.get(key))
        for team_id, is_rad in ((rad_id, True), (dire_id, False)):
            if team_id not in tracked_ids:
                continue
            for minute in minutes:
                idx = min(minute, len(adv_array) - 1) if adv_array else None
                if idx is None or idx < 0:
                    continue
                adv = adv_array[idx] if is_rad else -adv_array[idx]
                bucket = bucketize(adv)
                team_win = radiant_win if is_rad else (1 - int(radiant_win))
                rows.append(
                    {
                        "team_id": team_id,
                        "minute": minute,
                        "adv": adv,
                        "bucket": bucket,
                        "team_win": team_win,
                    }
                )

    df = pl.DataFrame(rows, strict=False)
    if df.is_empty():
        return pl.DataFrame(
            [],
            schema={"team_id": pl.Int64, "minute": pl.Int64, "bucket": pl.Utf8, "winrate": pl.Float64, "matches": pl.Int64, "adv_avg": pl.Float64},
        )
    agg = (
        df.group_by(["team_id", "minute", "bucket"])
        .agg(
            pl.col("team_win").mean().alias("winrate"),
            pl.len().alias("matches"),
            pl.col("adv").mean().alias("adv_avg"),
        )
        .sort(["team_id", "minute", "bucket"])
    )
    return agg


def compute_series_maps(matches: pl.DataFrame, tracked_ids: List[int]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Reconstruct series map order and per-team map winrates.
    Returns (series_maps, team_map_stats)
    """
    series_matches = matches.filter(pl.col("series_id").is_not_null())
    if series_matches.is_empty():
        empty_series = pl.DataFrame(
            [],
            schema={
                "series_id": pl.Int64,
                "series_type": pl.Int64,
                "map_num": pl.Int64,
                "match_id": pl.Int64,
                "start_time": pl.Int64,
                "radiant_team_id": pl.Int64,
                "dire_team_id": pl.Int64,
                "radiant_win": pl.Boolean,
            },
        )
        empty_team = pl.DataFrame(
            [],
            schema={
                "team_id": pl.Int64,
                "map_num": pl.Int64,
                "winrate": pl.Float64,
                "maps_played": pl.Int64,
                "series_type_sample": pl.Int64,
            },
        )
        return empty_series, empty_team

    max_maps = {0: 1, 1: 3, 2: 5, 3: 2}  # BO1, BO3, BO5, BO2

    series_maps = (
        series_matches.sort(["series_id", "start_time", "match_id"])
        .with_columns(
            [
                pl.col("match_id").cum_count().over("series_id").alias("map_num"),
                pl.col("series_type").map_elements(lambda x: max_maps.get(x)).alias("max_maps"),
            ]
        )
        .filter(
            pl.when(pl.col("max_maps").is_not_null())
            .then(pl.col("map_num") <= pl.col("max_maps"))
            .otherwise(True)
        )
    )
    series_maps = series_maps.select(
        "series_id",
        "series_type",
        "map_num",
        "match_id",
        "start_time",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
    )

    rows = []
    for row in series_maps.iter_rows(named=True):
        for team_id, is_rad in ((row["radiant_team_id"], True), (row["dire_team_id"], False)):
            if team_id not in tracked_ids:
                continue
            team_win = row["radiant_win"] if is_rad else (1 - int(row["radiant_win"]))
            rows.append(
                {
                    "team_id": team_id,
                    "map_num": row["map_num"],
                    "team_win": team_win,
                    "series_type": row.get("series_type"),
                }
            )
    team_stats = (
        pl.DataFrame(rows, strict=False)
        .group_by(["team_id", "map_num", "series_type"])
        .agg(
            pl.col("team_win").mean().alias("winrate"),
            pl.len().alias("maps_played"),
        )
        .sort(["team_id", "series_type", "map_num"])
    )
    return series_maps, team_stats


def main():
    parser = argparse.ArgumentParser(description="Precompute metrics (Elo, firsts) for tracked teams.")
    parser.add_argument("--processed", default="data/processed", help="Path to processed parquet dir.")
    parser.add_argument("--teams", default="data/teams_to_look.csv", help="CSV listing tracked teams (TeamID).")
    parser.add_argument("--out", default="data/metrics", help="Output directory for metrics.")
    parser.add_argument("--raw", default="data/raw/data_v2.json", help="Raw JSON file for gold/xp adv.")
    parser.add_argument("--k", type=float, default=24.0, help="Elo K factor.")
    parser.add_argument("--base-elo", type=float, default=1500.0, help="Base Elo for unseen teams.")
    parser.add_argument("--side-adv", type=float, default=25.0, help="Radiant side advantage in Elo calc.")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = read_processed_tables(processed_dir)
    matches_raw = tables["matches"]
    objectives = tables["objectives"]

    team_ids = load_lookup(Path(args.teams))
    teams_dict = build_team_dictionary(matches_raw)
    tracked_names = teams_dict.filter(pl.col("team_id").is_in(team_ids))[["team_id", "name"]]

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    raw_data = json.loads(raw_path.read_text())
    raw_map = {item["json"]["match_id"]: item["json"] for item in raw_data}

    elo_hist, elo_latest = compute_elo(
        matches_raw.select(
            ["match_id", "start_time", "radiant_team_id", "dire_team_id", "radiant_win"]
        ),
        team_ids=team_ids,
        k=args.k,
        base_elo=args.base_elo,
        side_adv=args.side_adv,
    )
    firsts = compute_firsts(matches_raw, objectives, tables["players"], tracked_ids=team_ids)
    roshan = compute_roshan_metrics(matches_raw, objectives, tracked_ids=team_ids)
    gold_buckets = compute_adv_buckets(matches_raw, raw_map=raw_map, tracked_ids=team_ids, key="radiant_gold_adv", minutes=(5, 10, 12, 15, 20))
    xp_buckets = compute_adv_buckets(matches_raw, raw_map=raw_map, tracked_ids=team_ids, key="radiant_xp_adv", minutes=(5, 10, 12, 15, 20))
    series_maps, series_team_stats = compute_series_maps(matches_raw, tracked_ids=team_ids)

    elo_hist.write_parquet(out_dir / "elo_timeseries.parquet")
    elo_latest.write_parquet(out_dir / "elo_latest.parquet")
    firsts.write_parquet(out_dir / "firsts.parquet")
    tracked_names.write_parquet(out_dir / "tracked_teams.parquet")
    roshan.write_parquet(out_dir / "roshan.parquet")
    gold_buckets.write_parquet(out_dir / "gold_buckets.parquet")
    xp_buckets.write_parquet(out_dir / "xp_buckets.parquet")
    series_maps.write_parquet(out_dir / "series_maps.parquet")
    series_team_stats.write_parquet(out_dir / "series_team_stats.parquet")

    print("Metrics written:")
    print(f"- Elo history: {out_dir / 'elo_timeseries.parquet'} ({elo_hist.shape})")
    print(f"- Elo latest: {out_dir / 'elo_latest.parquet'} ({elo_latest.shape})")
    print(f"- Firsts: {out_dir / 'firsts.parquet'} ({firsts.shape})")
    print(f"- Tracked teams: {out_dir / 'tracked_teams.parquet'} ({tracked_names.shape})")
    print(f"- Roshan: {out_dir / 'roshan.parquet'} ({roshan.shape})")
    print(f"- Gold buckets: {out_dir / 'gold_buckets.parquet'} ({gold_buckets.shape})")
    print(f"- XP buckets: {out_dir / 'xp_buckets.parquet'} ({xp_buckets.shape})")
    print(f"- Series maps: {out_dir / 'series_maps.parquet'} ({series_maps.shape})")
    print(f"- Series team stats: {out_dir / 'series_team_stats.parquet'} ({series_team_stats.shape})")


if __name__ == "__main__":
    main()
