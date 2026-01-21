from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import polars as pl

from src.dota_data import (
    read_processed_tables,
    build_team_dictionary,
)

SERIES_OVERRIDE_PATH = Path("data/mappings/series_overrides.csv")

ADV_BUCKET_MINUTES = (5, 10, 12, 15, 20)
ADV_BUCKETS = [-10_000, -5_000, -1_000, 0, 1_000, 5_000, 10_000, 999_999]
TIER_WEIGHTS = {"major": 1.25, "qualifier": 1.05, "regular": 1.0}
LOCATION_WEIGHTS = {"lan": 1.05, "online": 1.0}
PATCH_INIT_WEIGHT = 0.75


def _k_factor(games: int) -> float:
    """Tiered K-factor depending on experience."""
    if games < 30:
        return 40.0
    if games < 80:
        return 25.0
    return 15.0


def _match_weight(tier: str | None, location: str | None) -> float:
    tier_w = TIER_WEIGHTS.get(str(tier or "").lower(), 1.0)
    loc_w = LOCATION_WEIGHTS.get(str(location or "").lower(), 1.0)
    return tier_w * loc_w


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


def compute_elo(
    matches: pl.DataFrame,
    team_ids: List[int],
    base_elo: float = 1500.0,
    side_adv: float = 20.0,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Compute Elo for tracked teams (global + patch).
    - Only tracked teams are emitted in history/latest, but opponents can be untracked.
    - Weighted by tournament tier/location; side_adv applies to Radiant.
    - Patch Elo is initialized as a blend of current global Elo and base_elo.
    """
    tracked = set(team_ids)
    state: Dict[int, Dict[str, Dict]] = {}
    history_rows: List[Dict[str, object]] = []

    def get_team_state(tid: int) -> Dict[str, Dict]:
        if tid not in state:
            state[tid] = {"global": {"elo": base_elo, "games": 0}, "patches": {}}
        return state[tid]

    # Sort matches by time to avoid leakage
    needed_cols = [
        "match_id",
        "start_time",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
        "patch",
        "tournament_tier",
        "tournament_location",
    ]
    matches_sorted = matches.select([c for c in needed_cols if c in matches.columns]).sort("start_time")

    for row in matches_sorted.iter_rows(named=True):
        rid = row.get("radiant_team_id")
        did = row.get("dire_team_id")
        if rid is None or did is None:
            continue
        # ensure ints
        try:
            rid = int(rid)
            did = int(did)
        except Exception:
            continue
        rad_win = row.get("radiant_win")
        if rad_win is None:
            continue
        rad_win = bool(rad_win)

        tier = row.get("tournament_tier") or "regular"
        location = row.get("tournament_location") or "online"
        weight = _match_weight(tier, location)
        patch_val = row.get("patch")
        try:
            patch_int = int(patch_val) if patch_val is not None else None
        except Exception:
            patch_int = None

        # Pull states
        r_state = get_team_state(rid)
        d_state = get_team_state(did)

        # Global K (shared for the match)
        k_match_global = weight * (_k_factor(r_state["global"]["games"]) + _k_factor(d_state["global"]["games"])) / 2

        # Expected scores with side advantage (Radiant gets +side_adv)
        expected_rad = 1 / (1 + 10 ** (((d_state["global"]["elo"] - r_state["global"]["elo"]) - side_adv) / 400))
        delta_rad = k_match_global * ((1 if rad_win else 0) - expected_rad)

        # Patch states
        r_patch = d_patch = None
        if patch_int is not None:
            if patch_int not in r_state["patches"]:
                r_state["patches"][patch_int] = {
                    "elo": PATCH_INIT_WEIGHT * r_state["global"]["elo"] + (1 - PATCH_INIT_WEIGHT) * base_elo,
                    "games": 0,
                }
            if patch_int not in d_state["patches"]:
                d_state["patches"][patch_int] = {
                    "elo": PATCH_INIT_WEIGHT * d_state["global"]["elo"] + (1 - PATCH_INIT_WEIGHT) * base_elo,
                    "games": 0,
                }
            r_patch = r_state["patches"][patch_int]
            d_patch = d_state["patches"][patch_int]
            k_match_patch = weight * (_k_factor(r_patch["games"]) + _k_factor(d_patch["games"])) / 2
            expected_rad_patch = 1 / (1 + 10 ** (((d_patch["elo"] - r_patch["elo"]) - side_adv) / 400))
            delta_rad_patch = k_match_patch * ((1 if rad_win else 0) - expected_rad_patch)
        else:
            expected_rad_patch = None
            delta_rad_patch = None

        # Update global
        r_state["global"]["elo"] += delta_rad
        d_state["global"]["elo"] -= delta_rad
        r_state["global"]["games"] += 1
        d_state["global"]["games"] += 1

        # Update patch
        if r_patch is not None and d_patch is not None and delta_rad_patch is not None:
            r_patch["elo"] += delta_rad_patch
            d_patch["elo"] -= delta_rad_patch
            r_patch["games"] += 1
            d_patch["games"] += 1

        # History rows for tracked teams
        for team_id, opp_id, team_is_radiant, team_win in [
            (rid, did, True, 1 if rad_win else 0),
            (did, rid, False, 0 if rad_win else 1),
        ]:
            if team_id not in tracked:
                continue
            t_state = r_state if team_is_radiant else d_state
            tp_state = r_patch if team_is_radiant else d_patch
            history_rows.append(
                {
                    "match_id": row.get("match_id"),
                    "start_time": row.get("start_time"),
                    "start_dt": datetime.fromtimestamp(row.get("start_time", 0) or 0, tz=timezone.utc),
                    "team_id": team_id,
                    "opponent_id": opp_id,
                    "team_is_radiant": bool(team_is_radiant),
                    "team_win": team_win,
                    "rating_pre": (t_state["global"]["elo"] - delta_rad) if team_is_radiant else (t_state["global"]["elo"] + delta_rad),
                    "rating_post": t_state["global"]["elo"],
                    "rating_pre_patch": (tp_state["elo"] - (delta_rad_patch if team_is_radiant else -delta_rad_patch))
                    if tp_state is not None and delta_rad_patch is not None
                    else None,
                    "rating_post_patch": tp_state["elo"] if tp_state is not None else None,
                    "expected": expected_rad if team_is_radiant else (1 - expected_rad),
                    "expected_patch": expected_rad_patch if team_is_radiant else (1 - expected_rad_patch) if expected_rad_patch is not None else None,
                    "tournament_tier": tier,
                    "tournament_location": location,
                    "weight": weight,
                    "patch": patch_int,
                }
            )

    hist_df = pl.DataFrame(history_rows, strict=False)

    # Latest global Elo
    latest_rows = [{"team_id": tid, "elo": state.get(tid, {}).get("global", {}).get("elo", base_elo)} for tid in tracked]
    latest_df = pl.DataFrame(latest_rows, strict=False).sort("elo", descending=True)

    # Latest patch Elo
    patch_rows = []
    for tid in tracked:
        patches = state.get(tid, {}).get("patches", {})
        for p_val, p_state in patches.items():
            patch_rows.append({"team_id": tid, "patch": p_val, "elo": p_state.get("elo")})
    patch_latest = pl.DataFrame(patch_rows, strict=False)
    if not patch_latest.is_empty():
        patch_latest = patch_latest.sort(["patch", "elo"], descending=[False, True])

    return hist_df, latest_df, patch_latest


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
            (pl.col("first_blood").fill_null(False).cast(pl.Int64).sum() / pl.len()).alias("first_blood_rate"),
            (pl.col("first_tower").fill_null(False).cast(pl.Int64).sum() / pl.len()).alias("first_tower_rate"),
            (pl.col("first_roshan").fill_null(False).cast(pl.Int64).sum() / pl.len()).alias("first_roshan_rate"),
            pl.col("first_blood").fill_null(False).cast(pl.Int64).sum().alias("first_blood_count"),
            pl.col("first_tower").fill_null(False).cast(pl.Int64).sum().alias("first_tower_count"),
            pl.col("first_roshan").fill_null(False).cast(pl.Int64).sum().alias("first_roshan_count"),
            pl.len().alias("matches"),
        )
        .sort(["team_id", "team_is_radiant"])
    )
    return agg


def _fb_map(players: pl.DataFrame) -> Dict[int, Optional[bool]]:
    fb = (
        players.filter(pl.col("firstblood_claimed") == 1)
        .select("match_id", "is_radiant")
        .group_by("match_id")
        .agg(pl.first("is_radiant").alias("fb_is_radiant"))
    )
    return {int(r["match_id"]): r["fb_is_radiant"] for r in fb.iter_rows(named=True)}


def _ft_map(objectives: pl.DataFrame) -> Dict[int, Optional[bool]]:
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
    return {int(r["match_id"]): r["building_is_radiant"] for r in first_tower.iter_rows(named=True)}


def _roshan_maps(objectives: pl.DataFrame) -> Tuple[Dict[int, Optional[str]], Dict[int, Tuple[int, int]]]:
    """Return (first_roshan_side, steals_rad/dire counts) per match."""
    obj_groups = objectives.partition_by("match_id", as_dict=True, maintain_order=True)
    first_rosh_map: Dict[int, Optional[str]] = {}
    steals_map: Dict[int, Tuple[int, int]] = {}
    for key, df in obj_groups.items():
        mid = key[0] if isinstance(key, tuple) else key
        try:
            mid = int(mid)
        except Exception:  # noqa: BLE001
            continue
        obj_list = df.to_dicts()
        rosh_kills = [o for o in obj_list if o.get("type") == "CHAT_MESSAGE_ROSHAN_KILL"]
        rosh_sides = []
        for o in rosh_kills:
            team_val = o.get("team")
            if team_val == 2:
                rosh_sides.append("radiant")
            elif team_val == 3:
                rosh_sides.append("dire")
        first_rosh_map[mid] = rosh_sides[0] if rosh_sides else None

        aegis_claims = [o for o in obj_list if o.get("type") == "CHAT_MESSAGE_AEGIS"]
        steals_rad = steals_dire = 0
        if rosh_kills and aegis_claims:
            aeg_iter = iter(sorted(aegis_claims, key=lambda x: x.get("time", 0)))
            current_aeg = next(aeg_iter, None)
            for rk in sorted(rosh_kills, key=lambda x: x.get("time", 0)):
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
        steals_map[mid] = (steals_rad, steals_dire)
    return first_rosh_map, steals_map


def compute_pick_outcomes(
    matches: pl.DataFrame,
    objectives: pl.DataFrame,
    players: pl.DataFrame,
    raw_map: Dict[int, dict],
    tracked_ids: List[int],
) -> pl.DataFrame:
    """
    Compute per-team outcomes split by pick order/side.
    Output rows per team/label: overall, radiant first/last pick, dire first/last pick.
    """
    fb_map = _fb_map(players)
    ft_map = _ft_map(objectives)
    first_rosh_map, steals_map = _roshan_maps(objectives)

    rows = []
    for row in matches.iter_rows(named=True):
        mid = row.get("match_id")
        rad_id = row.get("radiant_team_id")
        dire_id = row.get("dire_team_id")
        radiant_win = row.get("radiant_win")
        if mid is None or rad_id is None or dire_id is None or radiant_win is None:
            continue
        raw = raw_map.get(mid, {})
        pb = raw.get("picks_bans") or []
        picks = [x for x in pb if x.get("is_pick")]
        first_pick_team = last_pick_team = None
        if picks:
            first = min(picks, key=lambda x: x.get("order", 0))
            last = max(picks, key=lambda x: x.get("order", 0))
            first_pick_team = rad_id if first.get("team") == 0 else dire_id
            last_pick_team = rad_id if last.get("team") == 0 else dire_id

        fb_is_rad = fb_map.get(mid)
        ft_building_is_rad = ft_map.get(mid)
        fr_side = first_rosh_map.get(mid)
        steals_rad, steals_dire = steals_map.get(mid, (None, None))

        for team_id, team_is_radiant in ((rad_id, True), (dire_id, False)):
            if team_id not in tracked_ids:
                continue
            win = radiant_win if team_is_radiant else (1 - int(radiant_win))
            fb_hit = fb_is_rad == team_is_radiant if fb_is_rad is not None else False
            ft_hit = (ft_building_is_rad is not None and ft_building_is_rad != team_is_radiant)
            ft_hit = ft_hit if ft_building_is_rad is not None else False
            fr_hit = (fr_side == "radiant") == team_is_radiant if fr_side is not None else False
            combo_for = bool(fb_hit and ft_hit and fr_hit and win)
            combo_against = bool((not fb_hit) and (not ft_hit) and (not fr_hit) and (not win))

            steal_for = (steals_rad if team_is_radiant else steals_dire) > 0 if steals_rad is not None and steals_dire is not None else False
            steal_against = (steals_dire if team_is_radiant else steals_rad) > 0 if steals_rad is not None and steals_dire is not None else False

            rows.append(
                {
                    "team_id": team_id,
                    "team_is_radiant": team_is_radiant,
                    "is_first_pick": first_pick_team == team_id if first_pick_team is not None else None,
                    "is_last_pick": last_pick_team == team_id if last_pick_team is not None else None,
                    "win": win,
                    "first_blood": fb_hit,
                    "first_tower": ft_hit,
                    "first_roshan": fr_hit,
                    "combo_for": combo_for,
                    "combo_against": combo_against,
                    "aegis_steal_for": steal_for,
                    "aegis_steal_against": steal_against,
                }
            )

    df = pl.DataFrame(rows, strict=False)
    out_rows = []
    labels = [
        ("overall", None),
        ("radiant_first_pick", (pl.col("team_is_radiant") & pl.col("is_first_pick"))),
        ("radiant_last_pick", (pl.col("team_is_radiant") & pl.col("is_last_pick"))),
        ("dire_first_pick", (~pl.col("team_is_radiant") & pl.col("is_first_pick"))),
        ("dire_last_pick", (~pl.col("team_is_radiant") & pl.col("is_last_pick"))),
    ]

    for team_id in tracked_ids:
        df_team = df.filter(pl.col("team_id") == team_id)
        if df_team.is_empty():
            continue
        for label, mask in labels:
            sub = df_team if mask is None else df_team.filter(mask)
            if sub.is_empty():
                continue
            out_rows.append(
                {
                    "team_id": team_id,
                    "label": label,
                    "matches": sub.height,
                    "winrate": sub["win"].mean(),
                    "first_blood_rate": sub["first_blood"].mean(),
                    "first_blood_count": sub["first_blood"].cast(pl.Int64, strict=False).fill_null(0).sum(),
                    "first_tower_rate": sub["first_tower"].mean(),
                    "first_tower_count": sub["first_tower"].cast(pl.Int64, strict=False).fill_null(0).sum(),
                    "first_roshan_rate": sub["first_roshan"].mean(),
                    "first_roshan_count": sub["first_roshan"].cast(pl.Int64, strict=False).fill_null(0).sum(),
                    "combo_for_rate": sub["combo_for"].mean(),
                    "combo_against_rate": sub["combo_against"].mean(),
                    "aegis_steal_rate": sub["aegis_steal_for"].mean(),
                    "aegis_steal_against_rate": sub["aegis_steal_against"].mean(),
                }
            )

    return pl.DataFrame(out_rows, strict=False)


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


def _bucketize_adv(val: float) -> str:
    for i in range(len(ADV_BUCKETS) - 1):
        if ADV_BUCKETS[i] <= val < ADV_BUCKETS[i + 1]:
            return f"[{ADV_BUCKETS[i]/1000:.0f}k,{ADV_BUCKETS[i+1]/1000:.0f}k)"
    return f">={ADV_BUCKETS[-2]/1000:.0f}k"


def compute_draft_meta(matches: pl.DataFrame, raw_map: Dict[int, dict]) -> pl.DataFrame:
    """Return first/last pick team per match so the app never touches raw picks."""
    rows = []
    for row in matches.iter_rows(named=True):
        mid = row.get("match_id")
        raw = raw_map.get(mid, {})
        pb = raw.get("picks_bans") or []
        picks = [x for x in pb if x.get("is_pick")]
        first_pick_team = last_pick_team = None
        if picks:
            first = min(picks, key=lambda x: x.get("order", 0))
            last = max(picks, key=lambda x: x.get("order", 0))
            first_pick_team = row.get("radiant_team_id") if first.get("team") == 0 else row.get("dire_team_id")
            last_pick_team = row.get("radiant_team_id") if last.get("team") == 0 else row.get("dire_team_id")
        rows.append(
            {
                "match_id": mid,
                "first_pick_team_id": first_pick_team,
                "last_pick_team_id": last_pick_team,
            }
        )
    return pl.DataFrame(rows, strict=False)


def compute_adv_snapshots(
    matches: pl.DataFrame,
    raw_map: Dict[int, dict],
    tracked_ids: List[int],
    minutes: Sequence[int] = ADV_BUCKET_MINUTES,
) -> pl.DataFrame:
    """
    Store per-team advantage snapshots (gold/xp) at selected minutes for quick lookup in Streamlit.
    This replaces on-the-fly raw JSON parsing in the dashboard.
    """
    rows = []
    for row in matches.iter_rows(named=True):
        match_id = row.get("match_id")
        rad_id = row.get("radiant_team_id")
        dire_id = row.get("dire_team_id")
        raw = raw_map.get(match_id)
        if raw is None or rad_id is None or dire_id is None:
            continue
        gold_adv = _parse_adv(raw.get("radiant_gold_adv"))
        xp_adv = _parse_adv(raw.get("radiant_xp_adv"))
        for team_id, opp_id, is_rad in (
            (rad_id, dire_id, True),
            (dire_id, rad_id, False),
        ):
            if team_id not in tracked_ids:
                continue
            for minute in minutes:
                idx_gold = min(minute, len(gold_adv) - 1) if gold_adv else None
                idx_xp = min(minute, len(xp_adv) - 1) if xp_adv else None
                if idx_gold is None and idx_xp is None:
                    continue
                gold_val = gold_adv[idx_gold] if idx_gold is not None else None
                xp_val = xp_adv[idx_xp] if idx_xp is not None else None
                gold_team_adv = gold_val if is_rad else (-gold_val if gold_val is not None else None)
                xp_team_adv = xp_val if is_rad else (-xp_val if xp_val is not None else None)
                rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team_id,
                        "opponent_id": opp_id,
                        "team_is_radiant": is_rad,
                        "minute": minute,
                        "gold_adv": gold_team_adv,
                        "gold_bucket": _bucketize_adv(gold_team_adv) if gold_team_adv is not None else None,
                        "xp_adv": xp_team_adv,
                        "xp_bucket": _bucketize_adv(xp_team_adv) if xp_team_adv is not None else None,
                        "start_time": row.get("start_time"),
                    }
                )
    return pl.DataFrame(rows, strict=False)


def compute_adv_buckets(
    matches: pl.DataFrame,
    raw_map: Dict[int, dict],
    tracked_ids: List[int],
    key: str,
    minutes: Sequence[int] = ADV_BUCKET_MINUTES,
) -> pl.DataFrame:
    """Compute winrate by bucket for a given advantage key (radiant_gold_adv or radiant_xp_adv)."""
    rows = []

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
                bucket = _bucketize_adv(adv)
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
                "leagueid": pl.Int64,
                "series_type": pl.Int64,
                "bo_type": pl.Int64,
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
                "bo_type": pl.Int64,
            },
        )
        return empty_series, empty_team

    max_maps_by_bo = {1: 1, 2: 2, 3: 3, 5: 5}
    required_wins_by_bo = {1: 1, 2: 1, 3: 2, 5: 3}

    series_rows: List[Dict[str, Any]] = []
    team_rows: List[Dict[str, Any]] = []

    def first_non_null(df: pl.DataFrame, col: str) -> Optional[int]:
        if col not in df.columns:
            return None
        vals = [v for v in df[col].to_list() if v is not None]
        return int(vals[0]) if vals else None

    for (_sid, _lid), df in series_matches.group_by(["series_id", "leagueid"], maintain_order=True):
        df_sorted = df.sort(["start_time", "match_id"])
        bo_val = first_non_null(df_sorted, "bo_type")
        series_type_val = first_non_null(df_sorted, "series_type")
        if bo_val not in max_maps_by_bo:
            bo_val = series_type_val if series_type_val in max_maps_by_bo else None
        if bo_val is None:
            # Can't reason about this series BO reliably
            continue

        teams: set[int] = set()
        for rid, did in zip(df_sorted["radiant_team_id"], df_sorted["dire_team_id"]):
            if rid is not None:
                teams.add(int(rid))
            if did is not None:
                teams.add(int(did))
        if len(teams) != 2:
            # Ignore malformed series with more/less than two teams
            continue

        if df_sorted.height > max_maps_by_bo.get(bo_val, df_sorted.height):
            # Skip series that exceed allowed map count for their BO
            continue

        team_list = list(teams)
        win_count = {team_list[0]: 0, team_list[1]: 0}
        for rw, rid, did in zip(df_sorted["radiant_win"], df_sorted["radiant_team_id"], df_sorted["dire_team_id"]):
            winner = int(rid) if rw else int(did)
            win_count[winner] = win_count.get(winner, 0) + 1

        needed = required_wins_by_bo.get(bo_val, 1)
        if max(win_count.values()) < needed:
            # Incomplete series (no one reached required wins)
            continue
        if win_count[team_list[0]] == win_count[team_list[1]]:
            # Drawn series -> disregard for BO stats
            continue

        # Persist ordered maps
        for idx, row in enumerate(df_sorted.iter_rows(named=True), start=1):
            series_rows.append(
                {
                    "series_id": row["series_id"],
                    "leagueid": row["leagueid"],
                    "series_type": series_type_val,
                    "bo_type": bo_val,
                    "map_num": idx,
                    "match_id": row["match_id"],
                    "start_time": row["start_time"],
                    "radiant_team_id": row["radiant_team_id"],
                    "dire_team_id": row["dire_team_id"],
                    "radiant_win": row["radiant_win"],
                }
            )

            map_winner = row["radiant_team_id"] if row["radiant_win"] else row["dire_team_id"]
            for team_id in (row["radiant_team_id"], row["dire_team_id"]):
                if team_id is None or team_id not in tracked_ids:
                    continue
                team_rows.append(
                    {
                        "team_id": int(team_id),
                        "map_num": idx,
                        "team_win": 1 if map_winner == team_id else 0,
                        "bo_type": bo_val,
                    }
                )

    if series_rows:
        series_maps = pl.DataFrame(series_rows)
    else:
        series_maps = pl.DataFrame(
            [],
            schema={
                "series_id": pl.Int64,
                "leagueid": pl.Int64,
                "series_type": pl.Int64,
                "bo_type": pl.Int64,
                "map_num": pl.Int64,
                "match_id": pl.Int64,
                "start_time": pl.Int64,
                "radiant_team_id": pl.Int64,
                "dire_team_id": pl.Int64,
                "radiant_win": pl.Boolean,
            },
        )

    if team_rows:
        team_stats = (
            pl.DataFrame(team_rows, strict=False)
            .group_by(["team_id", "map_num", "bo_type"])
            .agg(
                pl.col("team_win").mean().alias("winrate"),
                pl.len().alias("maps_played"),
            )
            .sort(["team_id", "bo_type", "map_num"])
        )
    else:
        team_stats = pl.DataFrame(
            [],
            schema={
                "team_id": pl.Int64,
                "map_num": pl.Int64,
                "winrate": pl.Float64,
                "maps_played": pl.Int64,
                "bo_type": pl.Int64,
            },
        )

    return series_maps, team_stats


def apply_series_overrides(matches: pl.DataFrame, overrides_path: Path = SERIES_OVERRIDE_PATH) -> pl.DataFrame:
    """
    Apply manual overrides for specific (leagueid, series_id), e.g. finals played in BO5 or LAN.
    Override columns supported: override_bo_type, override_series_type (legacy), override_tournament_tier, override_tournament_location.
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
    # Ensure expected columns exist
    base = matches
    for col in ["bo_type", "series_type_raw", "series_type"]:
        if col not in base.columns:
            base = base.with_columns(pl.lit(None).alias(col))
    joined = base.join(ov, on=["leagueid", "series_id"], how="left")
    # Drop existing series_type to avoid duplicate names when overriding
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


def _iter_chunk_files(root: Path, pattern: str = "matches_chunk*.json") -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.glob(f"**/{pattern}") if p.is_file()])


def _raw_map_from_loaded(obj: Any, match_ids: set[int]) -> Dict[int, dict]:
    raw_map: Dict[int, dict] = {}
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return raw_map

    for item in obj:
        if not isinstance(item, dict):
            continue
        match = item.get("json") if "json" in item else item
        if not isinstance(match, dict):
            continue
        mid = match.get("match_id")
        if not isinstance(mid, int):
            continue
        if match_ids and mid not in match_ids:
            continue
        raw_map[mid] = match
    return raw_map


def load_raw_map(raw_sources: Sequence[Path], match_ids: set[int]) -> Dict[int, dict]:
    """
    Load match raw payloads keyed by match_id from:
    - combined JSON files (list of wrapped matches), and/or
    - directories containing chunk files (matches_chunk*.json).
    """
    raw_map: Dict[int, dict] = {}
    for src in raw_sources:
        if not src.exists():
            continue
        if src.is_dir():
            for fp in _iter_chunk_files(src):
                try:
                    with fp.open("r", encoding="utf-8") as f:
                        loaded = json.load(f)
                except Exception:
                    continue
                raw_map.update(_raw_map_from_loaded(loaded, match_ids))
            continue

        if src.is_file():
            try:
                with src.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except Exception:
                continue
            raw_map.update(_raw_map_from_loaded(loaded, match_ids))
    return raw_map


def main():
    parser = argparse.ArgumentParser(description="Precompute metrics (Elo, firsts) for tracked teams.")
    parser.add_argument("--processed", default="data/processed", help="Path to processed parquet dir.")
    parser.add_argument("--teams", default="data/teams_to_look.csv", help="CSV listing tracked teams (TeamID).")
    parser.add_argument("--out", default="data/metrics", help="Output directory for metrics.")
    parser.add_argument(
        "--raw",
        action="append",
        default=None,
        help="Raw source (file or directory). Repeatable. If omitted, defaults to data/raw/data_v2.json and data/raw/updates (if present).",
    )
    parser.add_argument("--base-elo", type=float, default=1500.0, help="Base Elo for unseen teams.")
    parser.add_argument("--side-adv", type=float, default=20.0, help="Radiant side advantage in Elo calc.")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = read_processed_tables(processed_dir)
    matches_raw = apply_series_overrides(tables["matches"], overrides_path=SERIES_OVERRIDE_PATH)
    objectives = tables["objectives"]

    teams_csv = pl.read_csv(args.teams)
    # normalize column names (strip spaces)
    teams_csv = teams_csv.rename({c: c.strip() for c in teams_csv.columns})
    team_ids = teams_csv["TeamID"].to_list()
    teams_dict = build_team_dictionary(matches_raw)
    tracked_names = (
        teams_dict.filter(pl.col("team_id").is_in(team_ids))
        .group_by("team_id")
        .agg(pl.col("name").drop_nulls().first().alias("name"))
    )
    # prefer canonical name from CSV
    tracked_names = teams_csv.select(pl.col("TeamID").alias("team_id"), pl.col("TeamName").alias("csv_name")).join(
        tracked_names, on="team_id", how="left"
    )
    tracked_names = tracked_names.with_columns(pl.coalesce(pl.col("csv_name"), pl.col("name")).alias("name")).select(["team_id", "name"])

    match_ids_needed: set[int] = set()
    if "match_id" in matches_raw.columns:
        match_ids_needed = set(
            int(v)
            for v in matches_raw.select(pl.col("match_id").cast(pl.Int64, strict=False))
            .drop_nulls()
            .get_column("match_id")
            .to_list()
            if v is not None
        )

    raw_args = args.raw if args.raw else ["data/raw/data_v2.json", "data/raw/updates"]
    raw_sources = [Path(p) for p in raw_args if p]
    existing_sources = [p for p in raw_sources if p.exists()]
    if not existing_sources:
        raise FileNotFoundError(f"No valid --raw sources found. Tried: {[str(p) for p in raw_sources]}")
    raw_map = load_raw_map(existing_sources, match_ids=match_ids_needed)
    if match_ids_needed and len(raw_map) < max(1, int(0.9 * len(match_ids_needed))):
        missing = len(match_ids_needed) - len(raw_map)
        print(f"[warn] raw_map coverage: {len(raw_map)}/{len(match_ids_needed)} (missing {missing})")
    draft_meta = compute_draft_meta(matches_raw, raw_map)

    elo_hist, elo_latest, elo_patch_latest = compute_elo(
        matches_raw,
        team_ids=team_ids,
        base_elo=args.base_elo,
        side_adv=args.side_adv,
    )
    # Add leaderboard rank (1 = highest elo)
    elo_latest = elo_latest.with_columns(pl.col("elo").rank(method="dense", descending=True).alias("elo_rank"))
    if elo_patch_latest is not None and not elo_patch_latest.is_empty():
        elo_patch_latest = elo_patch_latest.with_columns(
            pl.col("elo").rank(method="dense", descending=True).over("patch").alias("elo_rank")
        )
    firsts = compute_firsts(matches_raw, objectives, tables["players"], tracked_ids=team_ids)
    roshan = compute_roshan_metrics(matches_raw, objectives, tracked_ids=team_ids)
    gold_buckets = compute_adv_buckets(matches_raw, raw_map=raw_map, tracked_ids=team_ids, key="radiant_gold_adv", minutes=ADV_BUCKET_MINUTES)
    xp_buckets = compute_adv_buckets(matches_raw, raw_map=raw_map, tracked_ids=team_ids, key="radiant_xp_adv", minutes=ADV_BUCKET_MINUTES)
    adv_snapshots = compute_adv_snapshots(matches_raw, raw_map=raw_map, tracked_ids=team_ids, minutes=ADV_BUCKET_MINUTES)
    series_maps, series_team_stats = compute_series_maps(matches_raw, tracked_ids=team_ids)
    pick_outcomes = compute_pick_outcomes(matches_raw, objectives, tables["players"], raw_map=raw_map, tracked_ids=team_ids)

    elo_hist.write_parquet(out_dir / "elo_timeseries.parquet")
    elo_latest.write_parquet(out_dir / "elo_latest.parquet")
    if elo_patch_latest is not None and not elo_patch_latest.is_empty():
        elo_patch_latest.write_parquet(out_dir / "elo_patch_latest.parquet")
    firsts.write_parquet(out_dir / "firsts.parquet")
    tracked_names.write_parquet(out_dir / "tracked_teams.parquet")
    roshan.write_parquet(out_dir / "roshan.parquet")
    gold_buckets.write_parquet(out_dir / "gold_buckets.parquet")
    xp_buckets.write_parquet(out_dir / "xp_buckets.parquet")
    series_maps.write_parquet(out_dir / "series_maps.parquet")
    series_team_stats.write_parquet(out_dir / "series_team_stats.parquet")
    pick_outcomes.write_parquet(out_dir / "pick_outcomes.parquet")
    draft_meta.write_parquet(out_dir / "draft_meta.parquet")
    adv_snapshots.write_parquet(out_dir / "adv_snapshots.parquet")

    print("Metrics written:")
    print(f"- Elo history: {out_dir / 'elo_timeseries.parquet'} ({elo_hist.shape})")
    print(f"- Elo latest: {out_dir / 'elo_latest.parquet'} ({elo_latest.shape})")
    if elo_patch_latest is not None and not elo_patch_latest.is_empty():
        print(f"- Elo patch latest: {out_dir / 'elo_patch_latest.parquet'} ({elo_patch_latest.shape})")
    print(f"- Firsts: {out_dir / 'firsts.parquet'} ({firsts.shape})")
    print(f"- Tracked teams: {out_dir / 'tracked_teams.parquet'} ({tracked_names.shape})")
    print(f"- Roshan: {out_dir / 'roshan.parquet'} ({roshan.shape})")
    print(f"- Gold buckets: {out_dir / 'gold_buckets.parquet'} ({gold_buckets.shape})")
    print(f"- XP buckets: {out_dir / 'xp_buckets.parquet'} ({xp_buckets.shape})")
    print(f"- Series maps: {out_dir / 'series_maps.parquet'} ({series_maps.shape})")
    print(f"- Series team stats: {out_dir / 'series_team_stats.parquet'} ({series_team_stats.shape})")
    print(f"- Pick outcomes: {out_dir / 'pick_outcomes.parquet'} ({pick_outcomes.shape})")
    print(f"- Draft meta: {out_dir / 'draft_meta.parquet'} ({draft_meta.shape})")
    print(f"- Advantage snapshots: {out_dir / 'adv_snapshots.parquet'} ({adv_snapshots.shape})")


if __name__ == "__main__":
    main()
