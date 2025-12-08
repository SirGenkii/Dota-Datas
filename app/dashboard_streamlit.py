from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import polars as pl
import streamlit as st
import numpy as np


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


def series_stats_for_team(series_team: pl.DataFrame, team_id: int) -> pl.DataFrame:
    if series_team is None or series_team.is_empty():
        return pl.DataFrame([])
    return series_team.filter(pl.col("team_id") == team_id)


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


def compute_h2h_metrics(team_a_id: int, team_b_id: int, matches: pl.DataFrame, objectives: pl.DataFrame) -> Optional[dict]:
    pair = matches.filter(
        ((pl.col("radiant_team_id") == team_a_id) & (pl.col("dire_team_id") == team_b_id))
        | ((pl.col("radiant_team_id") == team_b_id) & (pl.col("dire_team_id") == team_a_id))
    )
    if pair.is_empty():
        return None
    # Select only needed columns to avoid pyarrow dtype issues
    pair_pd = pair.select(
        "match_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
        "start_time",
    ).to_pandas()
    match_ids = pair_pd["match_id"].tolist()
    obj = objectives.filter(pl.col("match_id").is_in(match_ids)).to_pandas().sort_values("time")

    def side_bool(is_radiant: bool, team_id: int, match_row: dict) -> Optional[bool]:
        if is_radiant:
            return match_row["radiant_team_id"] == team_id
        return match_row["dire_team_id"] == team_id

    def per_team_metrics(team_id: int):
        df = pair_pd.copy()
        df["team_is_radiant"] = df["radiant_team_id"] == team_id
        df["team_win"] = df.apply(
            lambda r: r["radiant_win"] if r["team_is_radiant"] else (1 - int(r["radiant_win"])),
            axis=1,
        )
        if df.empty:
            return {}
        team_is_rad_map = dict(zip(df["match_id"], df["team_is_radiant"]))

        # Precompute per-match stats
        stats = []
        fb_all = obj[obj["type"] == "CHAT_MESSAGE_FIRSTBLOOD"]
        towers = obj[obj["type"] == "building_kill"]
        rosh = obj[obj["type"] == "CHAT_MESSAGE_ROSHAN_KILL"]
        for _, r in df.iterrows():
            mid = r["match_id"]
            team_is_rad = r["team_is_radiant"]
            win = r["team_win"]

            # first blood
            fb_hit = None
            grp = fb_all[fb_all["match_id"] == mid]
            if not grp.empty:
                row_fb = grp.iloc[0]
                if pd.notnull(row_fb.get("player_slot")):
                    fb_hit = (row_fb["player_slot"] < 128) == team_is_rad
                elif pd.notnull(row_fb.get("slot")):
                    fb_hit = (row_fb["slot"] < 5) == team_is_rad

            # first tower
            ft_hit = None
            grp = towers[towers["match_id"] == mid]
            if not grp.empty:
                row_ft = grp.iloc[0]
                key = str(row_ft.get("key", ""))
                if "goodguys" in key:
                    building_is_rad = True
                elif "badguys" in key:
                    building_is_rad = False
                else:
                    building_is_rad = None
                if building_is_rad is not None:
                    ft_hit = (building_is_rad != team_is_rad)

            # first roshan
            fr_hit = None
            grp = rosh[rosh["match_id"] == mid]
            if not grp.empty:
                row_rosh = grp.iloc[0]
                team_val = row_rosh.get("team")
                if team_val in [2, 3]:
                    rosh_rad = team_val == 2
                    fr_hit = (rosh_rad == team_is_rad)

            # roshan kills count for team
            rk_count = 0
            if not grp.empty:
                for _, row_rk in grp.iterrows():
                    team_val = row_rk.get("team")
                    if team_val in [2, 3]:
                        rk_is_rad = team_val == 2
                        if rk_is_rad == team_is_rad:
                            rk_count += 1

            stats.append(
                {
                    "team_is_radiant": team_is_rad,
                    "win": win,
                    "first_blood": fb_hit,
                    "first_tower": ft_hit,
                    "first_roshan": fr_hit,
                    "roshan_kills": rk_count,
                }
            )

        if not stats:
            return {}
        df_stats = pd.DataFrame(stats)
        overall = {
            "winrate": df_stats["win"].mean(),
            "first_blood": df_stats["first_blood"].mean() if df_stats["first_blood"].notna().any() else None,
            "first_tower": df_stats["first_tower"].mean() if df_stats["first_tower"].notna().any() else None,
            "first_roshan": df_stats["first_roshan"].mean() if df_stats["first_roshan"].notna().any() else None,
            "roshan_kills_avg": df_stats["roshan_kills"].mean(),
        }
        by_side = {}
        for side_bool, label in [(True, "Radiant"), (False, "Dire")]:
            sub = df_stats[df_stats["team_is_radiant"] == side_bool]
            if sub.empty:
                by_side[label] = None
                continue
            by_side[label] = {
                "matches": len(sub),
                "winrate": sub["win"].mean(),
                "first_blood": sub["first_blood"].mean() if sub["first_blood"].notna().any() else None,
                "first_tower": sub["first_tower"].mean() if sub["first_tower"].notna().any() else None,
                "first_roshan": sub["first_roshan"].mean() if sub["first_roshan"].notna().any() else None,
                "roshan_kills_avg": sub["roshan_kills"].mean(),
            }

        overall["by_side"] = by_side
        return overall

    return {
        "matches": len(match_ids),
        "team_a": per_team_metrics(team_a_id),
        "team_b": per_team_metrics(team_b_id),
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
):
    suffix = f" ({matches_count} games)" if matches_count is not None else ""
    label = f"{team_name}{suffix}" if team_name else f"{title}{suffix}"
    if logo_url:
        col_logo, col_label = st.columns([1, 5])
        with col_logo:
            st.image(logo_url, width=64)
        with col_label:
            st.subheader(label)
    else:
        st.subheader(label)
    if team_id is None:
        st.info("Select a team.")
        return

    # Top stats: firsts + roshan
    firsts = metrics.get("firsts")
    roshan = metrics.get("roshan")

    first_team = firsts_for_team(firsts, team_id)
    roshan_team = roshan_for_team(roshan, team_id)

    winrate = None
    if matches_raw is not None:
        sub = matches_raw.filter((pl.col("radiant_team_id") == team_id) | (pl.col("dire_team_id") == team_id))
        if not sub.is_empty():
            tmp = sub.with_columns(
                pl.when(pl.col("radiant_team_id") == team_id)
                .then(pl.col("radiant_win").cast(pl.Float64))
                .otherwise(1 - pl.col("radiant_win").cast(pl.Float64))
                .alias("team_win")
            )
            winrate = tmp["team_win"].mean()

    # Aggregate firsts
    if not first_team.is_empty():
        fb = first_team["first_blood_rate"].mean()
        ft = first_team["first_tower_rate"].mean()
        fr = first_team["first_roshan_rate"].mean()
    else:
        fb = ft = fr = None

    # Roshan aggregates
    rk = roshan_team["roshan_kills_avg"].item() if not roshan_team.is_empty() else None
    ac = roshan_team["aegis_claims_avg"].item() if not roshan_team.is_empty() else None
    steals = roshan_team["steals_total"].item() if not roshan_team.is_empty() else None

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Winrate", f"{winrate*100:.1f}%" if winrate is not None else "N/A")
    c1.metric("First blood %", f"{fb*100:.1f}%" if fb is not None else "N/A")
    c2.metric("First tower %", f"{ft*100:.1f}%" if ft is not None else "N/A")
    c3.metric("First Roshan %", f"{fr*100:.1f}%" if fr is not None else "N/A")
    c4, c5, c6 = st.columns(3)
    c4.metric("Roshan kills avg", f"{rk:.2f}" if rk is not None else "N/A")
    c5.metric("Aegis claims avg", f"{ac:.2f}" if ac is not None else "N/A")
    steals_rate = roshan_team["steals_rate"].item() if not roshan_team.is_empty() and "steals_rate" in roshan_team.columns else None
    c6.metric("Aegis steals total", steals if steals is not None else "N/A")
    if steals_rate is not None:
        st.caption(f"Aegis steals (>=1 per match): {steals_rate*100:.1f}%")

    if matches_raw is not None and objectives is not None:
        combo = combined_firsts_win(matches_raw, objectives, team_id)
        st.metric("First blood+tower+roshan & win %", f"{combo*100:.1f}%" if combo is not None else "N/A")

    with st.expander("Firsts by side", expanded=False):
        if not first_team.is_empty():
            col_rad, col_dire = st.columns(2)
            for side_bool, label, col in [(True, "Radiant", col_rad), (False, "Dire", col_dire)]:
                sub = first_team.filter(pl.col("team_is_radiant") == side_bool)
                if sub.is_empty():
                    continue
                row = sub.row(0, named=True)
                fb_val = f"{row['first_blood_rate']*100:.1f}%" if row["first_blood_rate"] is not None else "N/A"
                ft_val = f"{row['first_tower_rate']*100:.1f}%" if row["first_tower_rate"] is not None else "N/A"
                fr_val = f"{row['first_roshan_rate']*100:.1f}%" if row["first_roshan_rate"] is not None else "N/A"
                with col:
                    st.markdown(f"**{label}** ({row.get('matches', 'N/A')} games)")
                    st.metric("First blood %", fb_val)
                    st.metric("First tower %", ft_val)
                    st.metric("First Roshan %", fr_val)
        else:
            st.info("No firsts data for this team.")

    # Gold/XP buckets
    gold_df = metrics.get("gold_buckets")
    xp_df = metrics.get("xp_buckets")
    if gold_df is not None and not gold_df.is_empty():
        minutes = sorted(gold_df["minute"].unique())
        default_min = minutes[0] if minutes else 10
        min_choice = st.selectbox(
            "Minute (gold/xp buckets)",
            minutes,
            index=minutes.index(default_min) if minutes else 0,
            key=f"gx_minute_{team_id}",
        )
        gold_team = buckets_for_team(gold_df, team_id, min_choice)
        if not gold_team.is_empty():
            df_gold = gold_team.to_pandas()
            gold_plot = df_gold.groupby("bucket")["winrate"].mean().reset_index().set_index("bucket")
            st.subheader("Gold advantage buckets")
            st.bar_chart(gold_plot, height=250)
            with st.expander("Full table (gold buckets)", expanded=False):
                st.dataframe(df_gold)
        else:
            st.info("No gold buckets for this team/minute.")
        if xp_df is not None and not buckets_for_team(xp_df, team_id, min_choice).is_empty():
            xp_team = buckets_for_team(xp_df, team_id, min_choice)
            df_xp = xp_team.to_pandas()
            xp_plot = df_xp.groupby("bucket")["winrate"].mean().reset_index().set_index("bucket")
            st.subheader("XP advantage buckets")
            st.bar_chart(xp_plot, height=250)
            with st.expander("Full table (xp buckets)", expanded=False):
                st.dataframe(df_xp)
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
            bo_choice = st.selectbox("BO type", sorted(df_team["bo_type"].unique()), index=0, key=f"bo_type_{team_id}")
            df_sel = df_team[df_team["bo_type"] == bo_choice]
            st.dataframe(df_sel[["map_num", "winrate", "maps_played"]])
            if not df_sel.empty:
                st.bar_chart(df_sel.set_index("map_num")["winrate"])
        else:
            st.info("No series data for this team.")
    else:
        st.info("No series metrics (run make precompute).")

    # Elo history
    elo_hist = metrics.get("elo_hist")
    if elo_hist is not None and not elo_hist.is_empty():
        st.subheader("Elo history")
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
            )

        # Head-to-head section
        st.header(f"Head-to-head: {team_a} vs {team_b}")
        h2h = compute_h2h_metrics(team_a_id, team_b_id, matches_raw, objectives)
        if h2h is None:
            st.info("No head-to-head games between these teams in the dataset.")
        else:
            st.metric("Matches", h2h["matches"])
            colA, colB = st.columns(2)
            for col, label, data in [
                (colA, team_a, h2h["team_a"]),
                (colB, team_b, h2h["team_b"]),
            ]:
                with col:
                    col.metric("Winrate", f"{data.get('winrate', 0)*100:.1f}%" if data.get("winrate") is not None else "N/A")
                    col.metric("First blood %", f"{data.get('first_blood', 0)*100:.1f}%" if data.get("first_blood") is not None else "N/A")
                    col.metric("First tower %", f"{data.get('first_tower', 0)*100:.1f}%" if data.get("first_tower") is not None else "N/A")
                    col.metric("First Roshan %", f"{data.get('first_roshan', 0)*100:.1f}%" if data.get("first_roshan") is not None else "N/A")
                    col.metric("Roshan kills avg", f"{data.get('roshan_kills_avg', 0):.2f}" if data.get("roshan_kills_avg") is not None else "N/A")

            # By side table
            with st.expander("H2H by side (A vs B)", expanded=False):
                rows = []
                for label, data in [(team_a, h2h["team_a"]), (team_b, h2h["team_b"])]:
                    by_side = data.get("by_side") or {}
                    for side_label in ["Radiant", "Dire"]:
                        side_data = by_side.get(side_label)
                        if side_data is None:
                            continue
                        rows.append(
                            {
                                "team": label,
                                "side": side_label,
                                "matches": side_data["matches"],
                                "winrate": side_data["winrate"],
                                "first_blood": side_data["first_blood"],
                                "first_tower": side_data["first_tower"],
                                "first_roshan": side_data["first_roshan"],
                                "roshan_kills_avg": side_data["roshan_kills_avg"],
                            }
                        )
                if rows:
                    df_side = pd.DataFrame(rows)
                    df_side["winrate"] = (df_side["winrate"] * 100).round(1)
                    df_side["first_blood"] = (df_side["first_blood"] * 100).round(1)
                    df_side["first_tower"] = (df_side["first_tower"] * 100).round(1)
                    df_side["first_roshan"] = (df_side["first_roshan"] * 100).round(1)
                    st.dataframe(df_side)
                else:
                    st.info("No side-level data for this matchup.")


if __name__ == "__main__":
    main()
