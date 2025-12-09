from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import pandas as pd
import polars as pl
import streamlit as st
import numpy as np
import altair as alt


def _find_root() -> Path:
    cand = Path.cwd()
    for c in [cand, *cand.parents]:
        if (c / "src").exists() and (c / "data").exists():
            return c
    return cand


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dota_data import (  # noqa: E402
    read_processed_tables,
    build_team_dictionary,
)

RAW_PATH_DEFAULT = Path("data/raw/data_v2.json")
PROCESSED_DIR_DEFAULT = Path("data/processed")
METRICS_DIR_DEFAULT = Path("data/metrics")
TEAMS_CSV_DEFAULT = Path("data/teams_to_look.csv")


@st.cache_data(show_spinner=False)
def load_tables(raw_path: Path, processed_dir: Path):
    tables = read_processed_tables(processed_dir)
    return tables


@st.cache_data(show_spinner=False)
def load_raw_map(raw_path: Path):
    data = json.loads(raw_path.read_text())
    return {item["json"]["match_id"]: item["json"] for item in data}


@st.cache_data(show_spinner=False)
def load_metrics(metrics_dir: Path):
    metrics = {}
    files = {
        "elo_hist": metrics_dir / "elo_timeseries.parquet",
        "elo_latest": metrics_dir / "elo_latest.parquet",
        "firsts": metrics_dir / "firsts.parquet",
        "roshan": metrics_dir / "roshan.parquet",
        "gold_buckets": metrics_dir / "gold_buckets.parquet",
        "xp_buckets": metrics_dir / "xp_buckets.parquet",
        "series_maps": metrics_dir / "series_maps.parquet",
        "series_team": metrics_dir / "series_team_stats.parquet",
        "tracked_teams": metrics_dir / "tracked_teams.parquet",
        "pick_outcomes": metrics_dir / "pick_outcomes.parquet",
    }
    for key, path in files.items():
        if path.exists():
            metrics[key] = pl.read_parquet(path)
    return metrics


def load_team_options(teams_dict: pl.DataFrame, tracked_names: Optional[pl.DataFrame], teams_csv: Path) -> pl.DataFrame:
    # Base lookup: unique team_id -> first non-null name/logo
    base = (
        teams_dict.group_by("team_id")
        .agg(
            pl.col("name").drop_nulls().first().alias("name"),
            pl.col("logo_url").drop_nulls().first().alias("logo_url"),
        )
        .filter(pl.col("team_id").is_not_null())
    )
    if tracked_names is not None and not tracked_names.is_empty():
        # Merge names from tracked, keep logos from base when available
        tracked = tracked_names
        if "logo_url" not in tracked.columns:
            tracked = tracked.with_columns(pl.lit(None).alias("logo_url"))
        merged = tracked.join(base, on="team_id", how="left", suffix="_base")
        merged = merged.with_columns(
            [
                pl.coalesce(pl.col("name"), pl.col("name_base")).alias("name"),
                pl.coalesce(pl.col("logo_url"), pl.col("logo_url_base")).alias("logo_url"),
            ]
        ).select(["team_id", "name", "logo_url"])
        return merged.sort("name")
    if teams_csv.exists():
        lookup = pl.read_csv(teams_csv)
        lookup_ids = lookup["TeamID"].to_list()
        return base.filter(pl.col("team_id").is_in(lookup_ids)).sort("name")
    return base.sort("name")


def firsts_for_team(firsts: pl.DataFrame, team_id: int) -> pl.DataFrame:
    if firsts is None or firsts.is_empty():
        return pl.DataFrame([])
    return firsts.filter(pl.col("team_id") == team_id)


def roshan_for_team(roshan: pl.DataFrame, team_id: int) -> pl.DataFrame:
    if roshan is None or roshan.is_empty():
        return pl.DataFrame([])
    return roshan.filter(pl.col("team_id") == team_id)


def buckets_for_team(df: pl.DataFrame, team_id: int, minute: int) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame([])
    return df.filter((pl.col("team_id") == team_id) & (pl.col("minute") == minute))

def bucketize_value(val: float) -> str:
    buckets = [-10_000, -5_000, -1_000, 0, 1_000, 5_000, 10_000, 999_999]
    for i in range(len(buckets) - 1):
        if buckets[i] <= val < buckets[i + 1]:
            return f"[{buckets[i]/1000:.0f}k,{buckets[i+1]/1000:.0f}k)"
    return f">={buckets[-2]/1000:.0f}k"


def recent_matches_for_bucket(team_id: int, team_name: str, minute: int, bucket: str, key: str, matches_raw: pl.DataFrame, raw_map: dict, limit: int = 10):
    """Return recent matches for a team where its advantage at minute falls in bucket."""
    records = []
    for r in matches_raw.iter_rows(named=True):
        if r.get("radiant_team_id") not in (team_id, r.get("dire_team_id")) and r.get("dire_team_id") != team_id:
            continue
        mid = r.get("match_id")
        if mid not in raw_map:
            continue
        raw = raw_map[mid]
        adv_arr = raw.get(key) or []
        try:
            adv_arr = [float(x) for x in adv_arr]
        except Exception:
            continue
        if not adv_arr:
            continue
        idx = min(minute, len(adv_arr) - 1)
        adv = adv_arr[idx]
        team_is_rad = r.get("radiant_team_id") == team_id
        adv = adv if team_is_rad else -adv
        b = bucketize_value(adv)
        if b != bucket:
            continue
        opp_id = r.get("dire_team_id") if team_is_rad else r.get("radiant_team_id")
        opp_name = r.get("dire_name") if team_is_rad else r.get("radiant_name")
        if not opp_name:
            opp_name = str(opp_id)
        dt = datetime.fromtimestamp(r.get("start_time", 0))
        label = f"{team_name} VS {opp_name} ({dt:%d/%m/%Y})"
        records.append({"match_id": mid, "label": label, "start_time": r.get("start_time", 0)})
    records = sorted(records, key=lambda x: x["start_time"], reverse=True)
    return records[:limit]


BUCKET_ORDER = [
    "[-10k,-5k)",
    "[-5k,-1k)",
    "[-1k,0k)",
    "[0k,1k)",
    "[1k,5k)",
    "[5k,10k)",
    ">=10k",
]


def series_stats_for_team(series_team: pl.DataFrame, team_id: int) -> pl.DataFrame:
    if series_team is None or series_team.is_empty():
        return pl.DataFrame([])
    return series_team.filter(pl.col("team_id") == team_id)


def pick_outcomes_for_team(df: pl.DataFrame, team_id: int) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame([])
    return df.filter(pl.col("team_id") == team_id)


def elo_history_for_team(elo_hist: pl.DataFrame, team_id: int) -> pd.DataFrame:
    if elo_hist is None or elo_hist.is_empty():
        return pd.DataFrame()
    df = elo_hist.filter(pl.col("team_id") == team_id).sort("start_time").to_pandas()
    if not df.empty:
        df["start_dt"] = pd.to_datetime(df["start_time"], unit="s")
    return df


def combined_firsts_win(matches: pl.DataFrame, objectives: pl.DataFrame, team_id: int) -> Optional[float]:
    """Rate of matches where team gets first blood + first tower + first Roshan AND wins (common history)."""
    rel = matches.filter((pl.col("radiant_team_id") == team_id) | (pl.col("dire_team_id") == team_id))
    if rel.is_empty():
        return None
    df = rel.select(
        "match_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
    ).to_pandas()
    df["team_is_radiant"] = df["radiant_team_id"] == team_id
    df["team_win"] = df.apply(
        lambda r: r["radiant_win"] if r["team_is_radiant"] else (1 - int(r["radiant_win"])),
        axis=1,
    )
    obj = objectives.filter(pl.col("match_id").is_in(df["match_id"].tolist())).to_pandas().sort_values("time")
    flags = []
    for _, r in df.iterrows():
        mid = r["match_id"]
        team_is_rad = r["team_is_radiant"]
        team_win = r["team_win"]
        objs = obj[obj["match_id"] == mid]
        # FB
        fb_hit = None
        fb_grp = objs[objs["type"] == "CHAT_MESSAGE_FIRSTBLOOD"]
        if not fb_grp.empty:
            fb_row = fb_grp.iloc[0]
            if pd.notnull(fb_row.get("player_slot")):
                fb_hit = (fb_row["player_slot"] < 128) == team_is_rad
            elif pd.notnull(fb_row.get("slot")):
                fb_hit = (fb_row["slot"] < 5) == team_is_rad
        # FT
        ft_hit = None
        ft_grp = objs[objs["type"] == "building_kill"]
        if not ft_grp.empty:
            ft_row = ft_grp.iloc[0]
            key = str(ft_row.get("key", ""))
            building_is_rad = True if "goodguys" in key else False if "badguys" in key else None
            if building_is_rad is not None:
                ft_hit = (building_is_rad != team_is_rad)
        # FR
        fr_hit = None
        fr_grp = objs[objs["type"] == "CHAT_MESSAGE_ROSHAN_KILL"]
        if not fr_grp.empty:
            fr_row = fr_grp.iloc[0]
            team_val = fr_row.get("team")
            if team_val in [2, 3]:
                rosh_rad = team_val == 2
                fr_hit = (rosh_rad == team_is_rad)
        if fb_hit is not None and ft_hit is not None and fr_hit is not None:
            flags.append(fb_hit and ft_hit and fr_hit and bool(team_win))
    if not flags:
        return None
    return float(np.mean(flags))


def compute_h2h_metrics(team_a_id: int, team_b_id: int, matches: pl.DataFrame, objectives: pl.DataFrame, players: pl.DataFrame, raw_map: Dict[int, dict], last_n: int = 10) -> Optional[dict]:
    pair = matches.filter(
        ((pl.col("radiant_team_id") == team_a_id) & (pl.col("dire_team_id") == team_b_id))
        | ((pl.col("radiant_team_id") == team_b_id) & (pl.col("dire_team_id") == team_a_id))
    )
    if pair.is_empty():
        return None

    match_ids = pair["match_id"].to_list()
    obj = objectives.filter(pl.col("match_id").is_in(match_ids))
    fb_map = (
        players.filter((pl.col("match_id").is_in(match_ids)) & (pl.col("firstblood_claimed") == 1))
        .select("match_id", "is_radiant")
        .group_by("match_id")
        .agg(pl.first("is_radiant").alias("fb_is_radiant"))
    )
    fb_map = {int(r["match_id"]): r["fb_is_radiant"] for r in fb_map.iter_rows(named=True)}

    towers = obj.filter(pl.col("type") == "building_kill").with_columns(
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
    ft_map = {int(r["match_id"]): r["building_is_radiant"] for r in first_tower.iter_rows(named=True)}

    # Roshan and steals per match
    first_rosh_map: Dict[int, Optional[str]] = {}
    steals_map: Dict[int, Tuple[int, int]] = {}
    obj_groups = obj.partition_by("match_id", as_dict=True, maintain_order=True)
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

    rows = []
    pair_sorted = pair.sort("start_time")
    for r in pair_sorted.iter_rows(named=True):
        mid = r.get("match_id")
        rad_id = r.get("radiant_team_id")
        dire_id = r.get("dire_team_id")
        radiant_win = r.get("radiant_win")
        if None in (mid, rad_id, dire_id, radiant_win):
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
            win = radiant_win if team_is_radiant else (1 - int(radiant_win))
            fb_hit = fb_is_rad == team_is_radiant if fb_is_rad is not None else None
            ft_hit = (ft_building_is_rad is not None and ft_building_is_rad != team_is_radiant)
            ft_hit = ft_hit if ft_building_is_rad is not None else None
            if fr_side is None:
                fr_hit = None
            else:
                fr_hit = (fr_side == "radiant") == team_is_radiant
            combo_for = combo_against = None
            if fb_hit is not None and ft_hit is not None and fr_hit is not None:
                combo_for = bool(fb_hit and ft_hit and fr_hit and win)
                combo_against = bool((not fb_hit) and (not ft_hit) and (not fr_hit) and (not win))

            steal_for = None
            steal_against = None
            if steals_rad is not None and steals_dire is not None:
                steal_for = (steals_rad if team_is_radiant else steals_dire) > 0
                steal_against = (steals_dire if team_is_radiant else steals_rad) > 0

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
                    "match_id": mid,
                    "start_time": r.get("start_time"),
                }
            )

    df = pl.DataFrame(rows, strict=False)
    labels = [
        ("overall", None),
        ("radiant_first_pick", (pl.col("team_is_radiant") & pl.col("is_first_pick"))),
        ("radiant_last_pick", (pl.col("team_is_radiant") & pl.col("is_last_pick"))),
        ("dire_first_pick", (~pl.col("team_is_radiant") & pl.col("is_first_pick"))),
        ("dire_last_pick", (~pl.col("team_is_radiant") & pl.col("is_last_pick"))),
    ]

    def agg_for_team(tid: int):
        out_rows = []
        df_team = df.filter(pl.col("team_id") == tid)
        if df_team.is_empty():
            return pl.DataFrame([])
        for label, mask in labels:
            sub = df_team if mask is None else df_team.filter(mask)
            if sub.is_empty():
                continue
            out_rows.append(
                {
                    "label": label,
                    "matches": sub.height,
                    "winrate": sub["win"].mean(),
                    "first_blood_rate": sub["first_blood"].mean(),
                    "first_tower_rate": sub["first_tower"].mean(),
                    "first_roshan_rate": sub["first_roshan"].mean(),
                    "combo_for_rate": sub["combo_for"].mean(),
                    "combo_against_rate": sub["combo_against"].mean(),
                    "aegis_steal_rate": sub["aegis_steal_for"].mean(),
                    "aegis_steal_against_rate": sub["aegis_steal_against"].mean(),
                }
            )
        return pl.DataFrame(out_rows, strict=False)

    # timeline last n
    pair_pd = pair_sorted.select(["match_id", "start_time", "radiant_team_id", "dire_team_id", "radiant_win"]).to_pandas()
    pair_pd["start_dt"] = pd.to_datetime(pair_pd["start_time"], unit="s")
    pair_pd = pair_pd.sort_values("start_time", ascending=True).tail(last_n)
    timeline_rows = []
    for _, r in pair_pd.iterrows():
        for tid, is_rad in [(r["radiant_team_id"], True), (r["dire_team_id"], False)]:
            win = r["radiant_win"] if is_rad else (1 - int(r["radiant_win"]))
            timeline_rows.append(
                {
                    "team_id": tid,
                    "match_id": r["match_id"],
                    "start_dt": r["start_dt"],
                    "result": "Win" if win else "Loss",
                    "win": win,
                }
            )
    timeline = pd.DataFrame(timeline_rows)

    return {
        "matches": len(match_ids),
        "team_a": agg_for_team(team_a_id),
        "team_b": agg_for_team(team_b_id),
        "timeline": timeline,
    }


def team_block(
    title: str,
    team_id: Optional[int],
    team_name: str,
    metrics: dict,
    logo_url: Optional[str] = None,
    matches_count: Optional[int] = None,
    matches_raw: Optional[pl.DataFrame] = None,
    objectives: Optional[pl.DataFrame] = None,
    raw_map: Optional[dict] = None,
):
    suffix = f" ({matches_count} games)" if matches_count is not None else ""
    label = f"{team_name}{suffix}" if team_name else f"{title}{suffix}"
    if logo_url:
        col_logo, col_label = st.columns([1, 5])
        with col_logo:
            st.image(logo_url)
        with col_label:
            st.subheader(label)
    else:
        st.subheader(label)
    if team_id is None:
        st.info("Select a team.")
        return

    # Pick/side outcomes table
    pick_df = pick_outcomes_for_team(metrics.get("pick_outcomes"), team_id)
    st.subheader("Outcomes by side & pick order")
    if not pick_df.is_empty():
        label_map = {
            "overall": "Overall",
            "radiant_first_pick": "Radiant first pick",
            "radiant_last_pick": "Radiant last pick",
            "dire_first_pick": "Dire first pick",
            "dire_last_pick": "Dire last pick",
        }
        order = ["overall", "radiant_first_pick", "radiant_last_pick", "dire_first_pick", "dire_last_pick"]
        pick_pd = pick_df.to_pandas()
        pick_pd = pick_pd[pick_pd["label"].isin(order)]
        pick_pd["label"] = pick_pd["label"].map(label_map)

        def pct(val):
            return f"{val*100:.3f}%" if pd.notnull(val) else "N/A"

        rows = []
        for lbl in order:
            sub = pick_pd[pick_pd["label"] == label_map.get(lbl, lbl)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "Context": r["label"],
                    "First blood %": pct(r["first_blood_rate"]),
                    "First tower %": pct(r["first_tower_rate"]),
                    "First Roshan %": pct(r["first_roshan_rate"]),
                    "Winrate %": pct(r["winrate"]),
                    "FB+FT+FR & win %": pct(r["combo_for_rate"]),
                    "FB+FT+FR against %": pct(r["combo_against_rate"]),
                    "Aegis steal %": pct(r["aegis_steal_rate"]),
                    "Aegis stolen %": pct(r["aegis_steal_against_rate"]),
                    "Games": int(r["matches"]),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No pick/side outcomes for this team.")
    else:
        st.info("No pick/side outcomes for this team.")

    # Gold/XP buckets
    gold_df = metrics.get("gold_buckets")
    xp_df = metrics.get("xp_buckets")
    if gold_df is not None and not gold_df.is_empty():
        minutes = sorted(gold_df["minute"].unique())
        default_min = 10 if 10 in minutes else (minutes[0] if minutes else 10)
        min_choice = st.selectbox(
            "Minute (gold/xp buckets)",
            minutes,
            index=minutes.index(default_min) if minutes else 0,
            key=f"gx_minute_{team_id}_{title}",
        )
        gold_team = buckets_for_team(gold_df, team_id, min_choice)
        if not gold_team.is_empty():
            df_gold = gold_team.to_pandas()
            gold_plot = df_gold.groupby("bucket", as_index=False)["winrate"].mean()
            # order buckets
            cats = [b for b in BUCKET_ORDER if b in gold_plot["bucket"].unique()]
            gold_plot["bucket"] = pd.Categorical(gold_plot["bucket"], categories=cats, ordered=True)
            gold_plot = gold_plot.sort_values("bucket")
            st.subheader("Gold advantage buckets")
            chart = (
                alt.Chart(gold_plot)
                .mark_bar()
                .encode(
                    x=alt.X("bucket:N", sort=cats, axis=alt.Axis(labelAngle=0, labelFontSize=18, titleFontSize=18)),
                    y=alt.Y("winrate:Q", title="Winrate", axis=alt.Axis(labelFontSize=18, titleFontSize=18)),
                    tooltip=["bucket", alt.Tooltip("winrate:Q", format=".3f")],
                )
            )
            text = chart.mark_text(dy=-8, color="black", size=22).encode(text=alt.Text("winrate:Q", format=".1%"))
            st.altair_chart(chart + text, use_container_width=True)
            with st.expander("Full table (gold buckets)", expanded=False):
                st.dataframe(df_gold)


                if matches_raw is not None and raw_map is not None:
                    bucket_options = ["(none)"] + [b for b in cats if b in df_gold["bucket"].unique()]
                    if len(bucket_options) > 1:
                        bucket_choice = st.selectbox("Bucket range (gold)", bucket_options, index=0, key=f"gold_bucket_choice_{team_id}_{title}")
                        if bucket_choice != "(none)":
                            recent = recent_matches_for_bucket(team_id, team_name, min_choice, bucket_choice, "radiant_gold_adv", matches_raw, raw_map, limit=10)
                            if recent:
                                st.markdown("Recent matches in this bucket:", unsafe_allow_html=True)
                                for r in recent:
                                    st.markdown(
                                        f"<div><a href='https://www.dotabuff.com/matches/{r['match_id']}' target='_blank'>{r['label']}</a></div>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info("No matches in this bucket at this minute.")
        else:
            st.info("No gold buckets for this team/minute.")
        if xp_df is not None and not buckets_for_team(xp_df, team_id, min_choice).is_empty():
            xp_team = buckets_for_team(xp_df, team_id, min_choice)
            df_xp = xp_team.to_pandas()
            xp_plot = df_xp.groupby("bucket", as_index=False)["winrate"].mean()
            cats_xp = [b for b in BUCKET_ORDER if b in xp_plot["bucket"].unique()]
            xp_plot["bucket"] = pd.Categorical(xp_plot["bucket"], categories=cats_xp, ordered=True)
            xp_plot = xp_plot.sort_values("bucket")
            st.subheader("XP advantage buckets")
            chart_xp = (
                alt.Chart(xp_plot)
                .mark_bar()
                .encode(
                    x=alt.X("bucket:N", sort=cats_xp, axis=alt.Axis(labelAngle=0, labelFontSize=18, titleFontSize=18)),
                    y=alt.Y("winrate:Q", title="Winrate", axis=alt.Axis(labelFontSize=18, titleFontSize=18)),
                    tooltip=["bucket", alt.Tooltip("winrate:Q", format=".3f")],
                )
            )
            text_xp = chart_xp.mark_text(dy=-8, color="black", size=22).encode(text=alt.Text("winrate:Q", format=".1%"))
            st.altair_chart(chart_xp + text_xp, use_container_width=True)
            with st.expander("Full table (xp buckets)", expanded=False):
                st.dataframe(df_xp)
                if matches_raw is not None and raw_map is not None:
                    bucket_options_xp = ["(none)"] + [b for b in cats_xp if b in df_xp["bucket"].unique()]
                    if len(bucket_options_xp) > 1:
                        bucket_choice_xp = st.selectbox("Bucket range (xp)", bucket_options_xp, index=0, key=f"xp_bucket_choice_{team_id}_{title}")
                        if bucket_choice_xp != "(none)":
                            recent_xp = recent_matches_for_bucket(team_id, team_name, min_choice, bucket_choice_xp, "radiant_xp_adv", matches_raw, raw_map, limit=10)
                            if recent_xp:
                                st.markdown("Recent matches in this bucket:", unsafe_allow_html=True)
                                for r in recent_xp:
                                    st.markdown(
                                        f"<div><a href='https://www.dotabuff.com/matches/{r['match_id']}' target='_blank'>{r['label']}</a></div>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info("No matches in this bucket at this minute.")
    else:
        st.info("No gold/xp metrics (run make precompute).")

    # Series stats by BO type
    series_team = metrics.get("series_team")
    if series_team is not None and not series_team.is_empty():
        st.subheader("Winrate by map_num and BO type")
        df_team = series_stats_for_team(series_team, team_id).to_pandas()
        if not df_team.empty:
            # Map series_type to label
            bo_map = {0: "BO1", 1: "BO3", 2: "BO5", 3: "BO2"}
            df_team["bo_type"] = df_team["series_type"].map(bo_map).fillna("other")
            bo_options = sorted(df_team["bo_type"].unique())
            default_idx = bo_options.index("BO3") if "BO3" in bo_options else 0
            bo_choice = st.selectbox("BO type", bo_options, index=default_idx, key=f"bo_type_{team_id}")
            df_sel = df_team[df_team["bo_type"] == bo_choice]

            if not df_sel.empty:
                df_sel = df_sel.copy()
                df_sel["map_label"] = df_sel["map_num"].apply(lambda x: f"Map {x}")
                chart_series = (
                    alt.Chart(df_sel)
                    .mark_bar()
                    .encode(
                        x=alt.X("map_label:N", axis=alt.Axis(title="Map num", labelFontSize=18, titleFontSize=18, labelAngle=0)),
                        y=alt.Y("winrate:Q", axis=alt.Axis(title="Winrate", labelFontSize=18, titleFontSize=18)),
                        tooltip=["map_num", "map_label", alt.Tooltip("winrate:Q", format=".3f"), "maps_played"],
                    )
                )
                text_series = chart_series.mark_text(dy=-8, color="black", size=22).encode(text=alt.Text("winrate:Q", format=".1%"))
                st.altair_chart(chart_series + text_series, use_container_width=True)

                with st.expander("Full table (winrate by map_num)", expanded=False):
                    st.dataframe(df_sel[["map_num", "winrate", "maps_played"]])

               
        else:
            st.info("No series data for this team.")
    else:
        st.info("No series metrics (run make precompute).")

    # Elo history
    elo_hist = metrics.get("elo_hist")
    elo_latest = metrics.get("elo_latest")
    if elo_hist is not None and not elo_hist.is_empty():
        st.subheader("Elo history")
        current_elo = None
        current_rank = None
        if elo_latest is not None and not elo_latest.is_empty():
            cur = elo_latest.filter(pl.col("team_id") == team_id)
            if not cur.is_empty():
                current_elo = cur["elo"][0]
                if "elo_rank" in cur.columns:
                    current_rank = cur["elo_rank"][0]
        if current_elo is not None:
            rank_txt = f" (rank #{int(current_rank)})" if current_rank is not None else ""
            st.markdown(f"<h2 style='margin-top:-10px;'>Elo: {current_elo:.1f}{rank_txt}</h2>", unsafe_allow_html=True)
        df_elo = elo_history_for_team(elo_hist, team_id)
        if not df_elo.empty:
            st.line_chart(df_elo.set_index("start_dt")["rating_post"])
        else:
            st.info("No Elo history for this team.")
    else:
        st.info("No Elo metrics.")


def main():
    st.set_page_config(page_title="Dota Data Dashboard v2", layout="wide")
    st.title("Dota Data Dashboard v2")
    st.caption("Side-by-side comparison using precomputed metrics.")

    raw_path = ROOT / RAW_PATH_DEFAULT
    processed_dir = ROOT / PROCESSED_DIR_DEFAULT
    metrics_dir = ROOT / METRICS_DIR_DEFAULT

    tables = load_tables(raw_path, processed_dir)
    raw_map = load_raw_map(raw_path)
    metrics = load_metrics(metrics_dir)

    matches_raw = tables["matches"]
    objectives = tables["objectives"]

    teams_dict = build_team_dictionary(matches_raw)
    # matches count per team
    matches_counts = (
        matches_raw.select(
            pl.concat_list([pl.col("radiant_team_id"), pl.col("dire_team_id")]).alias("team_ids")
        )
        .explode("team_ids")
        .group_by("team_ids")
        .agg(pl.len().alias("matches"))
        .rename({"team_ids": "team_id"})
    )
    tracked_names = metrics.get("tracked_teams")
    team_names_df = load_team_options(teams_dict, tracked_names, ROOT / TEAMS_CSV_DEFAULT)
    team_options = team_names_df["name"].to_list()

    st.sidebar.header("Filters")
    team_a = st.sidebar.selectbox("Team A", team_options, index=0 if team_options else None, key="team_a_new")
    team_a_row = team_names_df.filter(pl.col("name") == team_a)
    team_a_id = team_a_row["team_id"][0] if team_options else None
    team_a_logo = team_a_row["logo_url"][0] if "logo_url" in team_a_row.columns and team_a_row.height else None
    matches_a = (
        matches_counts.filter(pl.col("team_id") == team_a_id)["matches"][0]
        if team_a_id is not None and not matches_counts.filter(pl.col("team_id") == team_a_id).is_empty()
        else None
    )

    team_b_options = ["None"] + team_options
    team_b = st.sidebar.selectbox("Team B", team_b_options, index=0, key="team_b_new")
    team_b_id = None
    matches_b = None
    team_b_logo = None
    if team_b != "None":
        team_b_row = team_names_df.filter(pl.col("name") == team_b)
        team_b_id = team_b_row["team_id"][0]
        team_b_logo = team_b_row["logo_url"][0] if "logo_url" in team_b_row.columns and team_b_row.height else None
        matches_b = (
            matches_counts.filter(pl.col("team_id") == team_b_id)["matches"][0]
            if team_b_id is not None and not matches_counts.filter(pl.col("team_id") == team_b_id).is_empty()
            else None
        )

    if team_b_id is None:
        team_block(
            f"{team_a}",
            team_a_id,
            team_a,
            metrics,
            logo_url=team_a_logo,
            matches_count=matches_a,
            matches_raw=matches_raw,
            objectives=objectives,
            raw_map=raw_map,
        )
    else:
        col_left, col_right = st.columns([1, 1], gap="small")
        with col_left:
            team_block(
                f"{team_a}",
                team_a_id,
                team_a,
                metrics,
                logo_url=team_a_logo,
                matches_count=matches_a,
                matches_raw=matches_raw,
                objectives=objectives,
                raw_map=raw_map,
            )
        # Ligne verticale fine entre les colonnes
        st.markdown(
            """
            <div style="position: absolute; top: 120px; bottom: 20px; left: 50%; width: 1px; background: #000;"></div>
            """,
            unsafe_allow_html=True,
        )
        with col_right:
            team_block(
                f"{team_b}",
                team_b_id,
                team_b,
                metrics,
                logo_url=team_b_logo,
                matches_count=matches_b,
                matches_raw=matches_raw,
                objectives=objectives,
                raw_map=raw_map,
            )

        # Head-to-head section
        st.header(f"Head-to-head: {team_a} vs {team_b}")
        h2h = compute_h2h_metrics(team_a_id, team_b_id, matches_raw, objectives, tables["players"], raw_map, last_n=20)
        if h2h is None:
            st.info("No head-to-head games between these teams in the dataset.")
        else:
            st.metric("Matches", h2h["matches"])

            def render_h2h_table(df_pl: pl.DataFrame, team_label: str):
                if df_pl is None or df_pl.is_empty():
                    st.info(f"No H2H data for {team_label}.")
                    return
                label_map = {
                    "overall": "Overall",
                    "radiant_first_pick": "Radiant first pick",
                    "radiant_last_pick": "Radiant last pick",
                    "dire_first_pick": "Dire first pick",
                    "dire_last_pick": "Dire last pick",
                }
                order = ["overall", "radiant_first_pick", "radiant_last_pick", "dire_first_pick", "dire_last_pick"]
                df = df_pl.to_pandas()
                df = df[df["label"].isin(order)]
                df["label"] = df["label"].map(label_map)
                def pct(val):
                    return f"{val*100:.3f}%" if pd.notnull(val) else "N/A"
                rows = []
                for lbl in order:
                    sub = df[df["label"] == label_map.get(lbl, lbl)]
                    if sub.empty:
                        continue
                    r = sub.iloc[0]
                    rows.append(
                        {
                            "Context": r["label"],
                            "First blood %": pct(r["first_blood_rate"]),
                            "First tower %": pct(r["first_tower_rate"]),
                            "First Roshan %": pct(r["first_roshan_rate"]),
                            "Winrate %": pct(r["winrate"]),
                            "FB+FT+FR & win %": pct(r["combo_for_rate"]),
                            "FB+FT+FR against %": pct(r["combo_against_rate"]),
                            "Aegis steal %": pct(r["aegis_steal_rate"]),
                            "Aegis stolen %": pct(r["aegis_steal_against_rate"]),
                            "Games": int(r["matches"]),
                        }
                    )
                st.subheader(f"{team_label} (H2H)")
                st.dataframe(pd.DataFrame(rows))

            colA, colB = st.columns(2)
            with colA:
                render_h2h_table(h2h["team_a"], team_a)
            with colB:
                render_h2h_table(h2h["team_b"], team_b)

            # Timeline of last matches
            timeline = h2h.get("timeline")
            if timeline is not None and not timeline.empty:
                st.subheader("Recent H2H results (chronological)")
                # Build tidy data for both teams
                tl = timeline.copy()
                team_map = {team_a_id: team_a, team_b_id: team_b}
                tl["team"] = tl["team_id"].map(team_map)
                # create a repeating order index per match
                n_matches = len(tl) // 2 if len(tl) else 0
                tl = tl.sort_values("start_dt")
                match_orders = []
                current = 1
                last_mid = None
                for _, row in tl.iterrows():
                    if row["match_id"] != last_mid:
                        match_orders.append(current)
                        last_mid = row["match_id"]
                        current += 1
                    else:
                        match_orders.append(current - 1)
                tl["match_order"] = match_orders
                chart_tl = (
                    alt.Chart(tl)
                    .mark_square(size=200)
                    .encode(
                        x=alt.X("match_order:O", axis=alt.Axis(title="Match (oldest→latest)", labelAngle=0, labelFontSize=12, titleFontSize=12)),
                        y=alt.Y("team:N", axis=alt.Axis(title="", labelFontSize=12)),
                        color=alt.Color("result:N", scale=alt.Scale(domain=["Win", "Loss"], range=["#2ca02c", "#d62728"])),
                        tooltip=["team", "result", "match_id", alt.Tooltip("start_dt:T", title="Date")],
                    )
                )
                st.altair_chart(chart_tl, use_container_width=True)
            else:
                st.info("No H2H timeline data.")


if __name__ == "__main__":
    main()
