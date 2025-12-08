from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import polars as pl
import streamlit as st


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


def team_block(
    title: str,
    team_id: Optional[int],
    team_name: str,
    metrics: dict,
    logo_url: Optional[str] = None,
    matches_count: Optional[int] = None,
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
        st.info("Sélectionnez une équipe.")
        return

    # Top stats: firsts + roshan
    firsts = metrics.get("firsts")
    roshan = metrics.get("roshan")

    first_team = firsts_for_team(firsts, team_id)
    roshan_team = roshan_for_team(roshan, team_id)

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

    c1, c2, c3 = st.columns(3)
    c1.metric("First blood %", f"{fb*100:.1f}%" if fb is not None else "N/A")
    c2.metric("First tower %", f"{ft*100:.1f}%" if ft is not None else "N/A")
    c3.metric("First Roshan %", f"{fr*100:.1f}%" if fr is not None else "N/A")
    c4, c5, c6 = st.columns(3)
    c4.metric("Roshan kills avg", f"{rk:.2f}" if rk is not None else "N/A")
    c5.metric("Aegis claims avg", f"{ac:.2f}" if ac is not None else "N/A")
    steals_rate = roshan_team["steals_rate"].item() if not roshan_team.is_empty() and "steals_rate" in roshan_team.columns else None
    c6.metric("Aegis steals % (>=1)", f"{steals_rate*100:.1f}%" if steals_rate is not None else "N/A")

    with st.expander("Firsts by side (table + chart)", expanded=False):
        if not first_team.is_empty():
            df = first_team.to_pandas()
            st.dataframe(df)
            # bar chart by side
            if not df.empty:
                df_plot = df.copy()
                df_plot["side"] = df_plot["team_is_radiant"].map({True: "Radiant", False: "Dire"})
                st.bar_chart(df_plot.set_index("side")[["first_blood_rate", "first_tower_rate", "first_roshan_rate"]])
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

    st.sidebar.header("Filtres")
    team_a = st.sidebar.selectbox("Équipe A", team_options, index=0 if team_options else None, key="team_a_new")
    team_a_row = team_names_df.filter(pl.col("name") == team_a)
    team_a_id = team_a_row["team_id"][0] if team_options else None
    team_a_logo = team_a_row["logo_url"][0] if "logo_url" in team_a_row.columns and team_a_row.height else None
    matches_a = (
        matches_counts.filter(pl.col("team_id") == team_a_id)["matches"][0]
        if team_a_id is not None and not matches_counts.filter(pl.col("team_id") == team_a_id).is_empty()
        else None
    )

    team_b_options = ["None"] + team_options
    team_b = st.sidebar.selectbox("Équipe B", team_b_options, index=0, key="team_b_new")
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
        team_block(f"{team_a}", team_a_id, team_a, metrics, logo_url=team_a_logo, matches_count=matches_a)
    else:
        col_left, col_right = st.columns([1, 1], gap="small")
        with col_left:
            team_block(f"{team_a}", team_a_id, team_a, metrics, logo_url=team_a_logo, matches_count=matches_a)
        # Ligne verticale fine entre les colonnes
        st.markdown(
            """
            <div style="position: absolute; top: 120px; bottom: 20px; left: 50%; width: 1px; background: #000;"></div>
            """,
            unsafe_allow_html=True,
        )
        with col_right:
            team_block(f"{team_b}", team_b_id, team_b, metrics, logo_url=team_b_logo, matches_count=matches_b)


if __name__ == "__main__":
    main()
