from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import pandas as pd
import streamlit as st


def _configure_path() -> Path:
    """Ensure project root is on sys.path so imports work when run from anywhere."""
    this_file = Path(__file__).resolve()
    candidates = [
        Path(os.getcwd()).resolve(),
        this_file.parent.parent.resolve(),  # app/ -> project root
        this_file.parent.parent.parent.resolve(),
    ]
    for root in candidates:
        if (root / "src").exists() and str(root) not in sys.path:
            sys.path.append(str(root))
            return root
    # Fallback: use cwd
    root = Path(os.getcwd()).resolve()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


ROOT = _configure_path()

from src.dota_data import (  # noqa: E402
    read_processed_tables,
    write_parquet_tables,
    match_header,
    objectives_timeline,
    build_team_dictionary,
    build_player_dictionary,
    build_hero_counts,
    load_hero_dictionary,
)


RAW_PATH_DEFAULT = Path("data/raw/data.json")
PROCESSED_DIR_DEFAULT = Path("data/processed")
METRICS_DIR_DEFAULT = Path("data/metrics")


@st.cache_data(show_spinner=False)
def load_tables(raw_path: Path, processed_dir: Path):
    if not (processed_dir / "matches.parquet").exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
        write_parquet_tables(raw_path, processed_dir)
    tables = read_processed_tables(processed_dir)
    return tables


@st.cache_data(show_spinner=False)
def load_raw_map(raw_path: Path):
    data = json.loads(raw_path.read_text())
    return {item["json"]["match_id"]: item["json"] for item in data}


@st.cache_data(show_spinner=False)
def load_metrics(metrics_dir: Path):
    """Load precomputed metrics if available."""
    metrics = {}
    files = {
        "elo_hist": metrics_dir / "elo_timeseries.parquet",
        "elo_latest": metrics_dir / "elo_latest.parquet",
        "firsts": metrics_dir / "firsts.parquet",
        "roshan": metrics_dir / "roshan.parquet",
        "gold_xp": metrics_dir / "gold_xp_buckets.parquet",
        "series_maps": metrics_dir / "series_maps.parquet",
        "series_team": metrics_dir / "series_team_stats.parquet",
        "tracked_teams": metrics_dir / "tracked_teams.parquet",
    }
    for key, path in files.items():
        if path.exists():
            metrics[key] = pl.read_parquet(path)
    return metrics


def first_blood_rate(players: pl.DataFrame) -> float:
    fb = players.filter(pl.col("firstblood_claimed") == 1).select("match_id", "is_radiant").unique()
    if len(fb) == 0:
        return 0.0
    return fb["is_radiant"].mean()


def first_tower_rate(objectives: pl.DataFrame) -> Optional[float]:
    if "type" not in objectives.columns or "team" not in objectives.columns:
        return None
    # Fallback to regex matching case-insensitive manually.
    towers = objectives.filter(pl.col("type").str.contains("(?i)building", literal=False))
    if len(towers) == 0:
        return None
    towers_sorted = towers.sort(["match_id", "time"])
    first_towers = towers_sorted.group_by("match_id").agg(
        pl.col("team").first().alias("team"),
    )
    # Heuristic: team value (2/3) depends on source; treat lowest value as radiant indicator
    min_team = first_towers["team"].min()
    radiant_rate = first_towers.filter(pl.col("team") == min_team).height / first_towers.height
    return radiant_rate


def _friendly_building_name(key: Optional[str]) -> Optional[str]:
    if not key or not isinstance(key, str):
        return None
    name = key
    for prefix in ["npc_dota_", "dota_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("goodguys", "radiant").replace("badguys", "dire")
    name = name.replace("_", " ").strip()
    return name


def main():
    st.set_page_config(page_title="Dota Data Dashboard", layout="wide")
    st.title("Dota Data Dashboard")
    st.caption("Team-centric view + head-to-head + match detail (first blood / first tower / winrate / gold-xp).")

    root = Path.cwd()
    raw_candidates = [
        root / "data/raw/data.json",
        root.parent / "data/raw/data.json",
    ]
    raw_path = next((p for p in raw_candidates if p.exists()), RAW_PATH_DEFAULT)
    processed_dir = raw_path.parent.parent / "processed"

    metrics_dir = root / METRICS_DIR_DEFAULT
    tables = load_tables(raw_path, processed_dir)
    raw_map = load_raw_map(raw_path)
    metrics = load_metrics(metrics_dir)

    matches_raw = tables["matches"]
    matches = match_header(matches_raw)
    players = tables["players"]
    objectives = tables["objectives"]

    teams_dict = build_team_dictionary(matches_raw)
    tracked_names = None
    if "tracked_teams" in metrics:
        tracked_names = metrics["tracked_teams"]
    players_dict = build_player_dictionary(players)
    hero_counts = build_hero_counts(players)
    try:
        heroes_dict = load_hero_dictionary()
    except FileNotFoundError:
        heroes_dict = None

    # Sidebar: team A and optional team B (H2H)
    # Restreindre les équipes aux 24 référencées dans data/teams_to_look.csv si dispo
    teams_csv = Path(ROOT) / "data/teams_to_look.csv"
    if tracked_names is not None and not tracked_names.is_empty():
        team_names = tracked_names.select(pl.col("team_id").alias("team_id"), pl.col("name")).drop_nulls().sort("name")
    elif teams_csv.exists():
        lookup = pl.read_csv(teams_csv)
        lookup_ids = lookup["TeamID"].to_list()
        team_names = (
            teams_dict.select(["team_id", "name"])
            .drop_nulls()
            .filter(pl.col("team_id").is_in(lookup_ids))
            .sort("name")
        )
    else:
        team_names = teams_dict.select(["team_id", "name"]).drop_nulls().sort("name")
    team_options = team_names["name"].to_list()

    # Precompute H2H pairs for filtering
    pairs = (
        matches_raw.select(
            pl.col("radiant_team_id").alias("a"),
            pl.col("dire_team_id").alias("b"),
        )
        .with_columns(
            [
                pl.when(pl.col("a") < pl.col("b")).then(pl.col("a")).otherwise(pl.col("b")).alias("t1"),
                pl.when(pl.col("a") < pl.col("b")).then(pl.col("b")).otherwise(pl.col("a")).alias("t2"),
            ]
        )
        .group_by(["t1", "t2"])
        .agg(pl.len().alias("matches"))
    )

    default_a = st.session_state.get("team_a_choice", team_options[0] if team_options else None)
    team_a = st.sidebar.selectbox(
        "Équipe A",
        team_options,
        index=team_options.index(default_a) if team_options and default_a in team_options else 0,
        key="team_a_choice",
    )
    team_a_id = team_names.filter(pl.col("name") == team_a)["team_id"][0] if team_options else None

    filter_h2h = st.sidebar.checkbox("Limit Team B to opponents already played", value=True)
    team_b_options = ["None"]
    if team_a_id is not None and filter_h2h:
        # collect opponents with at least one match
        opp_ids = []
        for row in pairs.iter_rows(named=True):
            if row["t1"] == team_a_id:
                opp_ids.append(row["t2"])
            elif row["t2"] == team_a_id:
                opp_ids.append(row["t1"])
        opp_ids = list(dict.fromkeys(opp_ids))  # unique preserve order
        opp_names = team_names.filter(pl.col("team_id").is_in(opp_ids))["name"].to_list()
        team_b_options += opp_names
    else:
        team_b_options += team_options

    default_b = st.session_state.get("team_b_choice", "None")
    team_b = st.sidebar.selectbox(
        "Équipe B (H2H)",
        team_b_options,
        index=team_b_options.index(default_b) if default_b in team_b_options else 0,
        key="team_b_choice",
    )
    team_b_id = None
    if team_b != "None":
        team_b_id = team_names.filter(pl.col("name") == team_b)["team_id"][0]

    # Team A datasets
    team_a_matches_full = matches_raw.filter(
        (pl.col("radiant_team_id") == team_a_id) | (pl.col("dire_team_id") == team_a_id)
    )
    team_a_matches = match_header(team_a_matches_full)
    allowed_ids = team_a_matches["match_id"].to_list()
    team_a_players = players.filter(pl.col("match_id").is_in(allowed_ids))
    team_a_objectives = objectives.filter(pl.col("match_id").is_in(allowed_ids))

    st.header(f"Team A: {team_a}")
    # Determine side per match for team A
    team_match_info = team_a_matches_full.select(
        "match_id",
        (pl.col("radiant_team_id") == team_a_id).alias("team_is_radiant"),
        "radiant_win",
        "duration",
        "radiant_score",
        "dire_score",
    )
    # Team-centric winrate
    team_win = team_match_info.with_columns(
        pl.when(pl.col("team_is_radiant")).then(pl.col("radiant_win")).otherwise(1 - pl.col("radiant_win")).alias("team_win")
    )
    winrate_a = team_win["team_win"].mean() if team_win.height else 0
    avg_dur = team_win["duration"].mean() / 60 if team_win.height else 0

    # Team-centric first blood: only players on same side as team A
    team_players = (
        team_match_info.select("match_id", "team_is_radiant")
        .join(players, on="match_id", how="inner")
        .filter(pl.col("is_radiant") == pl.col("team_is_radiant"))
    )
    fb_a = (
        team_players.group_by("match_id")
        .agg((pl.col("firstblood_claimed") == 1).any().cast(pl.Float64))
        .select(pl.col("firstblood_claimed").mean())
        .item(0, 0)
        if team_players.height
        else 0
    )

    # Split metrics by side (radiant/dire)
    team_win_r = team_win.filter(pl.col("team_is_radiant"))
    team_win_d = team_win.filter(~pl.col("team_is_radiant"))
    win_r = team_win_r["team_win"].mean() if team_win_r.height else 0
    win_d = team_win_d["team_win"].mean() if team_win_d.height else 0

    fb_r = (
        team_players.filter(pl.col("team_is_radiant"))
        .group_by("match_id")
        .agg((pl.col("firstblood_claimed") == 1).any().cast(pl.Float64))
        .select(pl.col("firstblood_claimed").mean())
        .item(0, 0)
        if team_players.filter(pl.col("team_is_radiant")).height
        else 0
    )
    fb_d = (
        team_players.filter(~pl.col("team_is_radiant"))
        .group_by("match_id")
        .agg((pl.col("firstblood_claimed") == 1).any().cast(pl.Float64))
        .select(pl.col("firstblood_claimed").mean())
        .item(0, 0)
        if team_players.filter(~pl.col("team_is_radiant")).height
        else 0
    )

    # Team-centric first tower using building owners from key (goodguys=Radiant, badguys=Dire)
    ft_a = None
    ft_r = None
    ft_d = None
    towers = (
        objectives.filter(pl.col("type") == "building_kill")
        .with_columns(
            pl.when(pl.col("key").str.contains("goodguys")).then(True)
            .when(pl.col("key").str.contains("badguys")).then(False)
            .otherwise(None)
            .alias("building_is_radiant")
        )
    )
    if towers.height:
        first_towers = towers.sort(["match_id", "time"]).group_by("match_id").agg(
            pl.first("building_is_radiant").alias("building_is_radiant"),
            pl.first("time").alias("first_tower_time"),
        )
        ft_join = team_match_info.join(first_towers, on="match_id", how="left")
        ft_all = ft_join.with_columns(
            pl.when(pl.col("building_is_radiant").is_null())
            .then(None)
            .when(pl.col("team_is_radiant"))
            .then(~pl.col("building_is_radiant"))
            .otherwise(pl.col("building_is_radiant"))
            .alias("team_first_tower")
        )
        ft_a = ft_all["team_first_tower"].mean()
        ft_r_df = ft_all.filter(pl.col("team_is_radiant"))
        ft_d_df = ft_all.filter(~pl.col("team_is_radiant"))
        ft_r = ft_r_df["team_first_tower"].mean() if ft_r_df.height else None
        ft_d = ft_d_df["team_first_tower"].mean() if ft_d_df.height else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", team_a_matches.height)
    col2.metric("Team winrate", f"{winrate_a*100:.1f}%")
    col3.metric("First blood share", f"{fb_a*100:.1f}%")
    col4.metric("First tower share (heuristic)", f"{ft_a*100:.1f}%" if ft_a is not None else "N/A")
    st.metric("Average duration (min)", f"{avg_dur:.1f}")

    st.subheader("By side")
    side_col1, side_col2 = st.columns(2)
    with side_col1:
        st.write("As Radiant")
        st.progress(win_r, text=f"Winrate {win_r*100:.1f}%" if team_win_r.height else "No radiant games")
        st.progress(fb_r, text=f"First blood {fb_r*100:.1f}%" if team_win_r.height else "No radiant games")
        st.progress(ft_r if ft_r is not None else 0, text=f"First tower {ft_r*100:.1f}%"
                   if ft_r is not None else "First tower N/A")
    with side_col2:
        st.write("As Dire")
        st.progress(win_d, text=f"Winrate {win_d*100:.1f}%" if team_win_d.height else "No dire games")
        st.progress(fb_d, text=f"First blood {fb_d*100:.1f}%" if team_win_d.height else "No dire games")
        st.progress(ft_d if ft_d is not None else 0, text=f"First tower {ft_d*100:.1f}%"
                   if ft_d is not None else "First tower N/A")

    st.subheader("Recent matches (Team A)")
    st.dataframe(team_a_matches.sort("start_time", descending=True).head(15).to_pandas())

    # Head-to-head
    st.header("Head-to-head")
    if team_b_id is not None:
        h2h_matches_full = team_a_matches_full.filter(
            (pl.col("radiant_team_id") == team_b_id) | (pl.col("dire_team_id") == team_b_id)
        )
        h2h_matches = match_header(h2h_matches_full)
        # Team-centric winrate for H2H subset
        h2h_info = team_match_info.filter(pl.col("match_id").is_in(h2h_matches["match_id"]))
        h2h_winrate = 0
        if h2h_info.height:
            h2h_winrate = (
                h2h_info.with_columns(
                    pl.when(pl.col("team_is_radiant")).then(pl.col("radiant_win")).otherwise(1 - pl.col("radiant_win")).alias("team_win")
                )["team_win"]
                .mean()
            )
        colh1, colh2 = st.columns(2)
        colh1.metric("H2H matches", h2h_matches.height)
        colh2.metric("H2H winrate (Team A)", f"{h2h_winrate*100:.1f}%")
        st.dataframe(h2h_matches.sort("start_time", descending=True).to_pandas())
    else:
        st.info("Select Team B to see head-to-head.")

    # Pré-calculs metrics (Roshan/Aegis, firsts, gold/xp buckets, séries, Elo)
    st.header("Pré-calculs v2 (team A)")
    if metrics:
        team_names_df = team_names.rename({"name": "team_name"})
        tabs = st.tabs(["Roshan/Aegis", "Firsts", "Gold/XP buckets", "Series", "Elo"])

        with tabs[0]:
            if "roshan" in metrics:
                rosh = metrics["roshan"].join(team_names_df, on="team_id", how="left")
                st.subheader("Roshan/Aegis global (tracked teams)")
                st.dataframe(rosh.sort("roshan_kills_avg", descending=True).to_pandas())
                team_rosh = rosh.filter(pl.col("team_id") == team_a_id)
                st.subheader(f"Team {team_a} Roshan/Aegis")
                st.dataframe(team_rosh.to_pandas())
                if "firsts" in metrics:
                    firsts_team = metrics["firsts"].filter(pl.col("team_id") == team_a_id)
                    st.write("First roshan rate par side (from firsts):")
                    st.dataframe(firsts_team.select(["team_is_radiant", "first_roshan_rate", "matches"]).to_pandas())
            else:
                st.info("Roshan metrics manquants (lance `make precompute`).")

        with tabs[1]:
            if "firsts" in metrics:
                firsts_all = metrics["firsts"].join(team_names_df, on="team_id", how="left")
                st.subheader("Firsts (first blood/tower/roshan) par équipe/côté")
                st.dataframe(firsts_all.to_pandas())
                team_firsts = firsts_all.filter(pl.col("team_id") == team_a_id)
                st.subheader(f"Team {team_a} (split Radiant/Dire)")
                st.dataframe(team_firsts.to_pandas())
            else:
                st.info("Firsts metrics manquants.")

        with tabs[2]:
            if "gold_xp" in metrics:
                gx = metrics["gold_xp"]
                minutes = sorted(gx["minute"].unique())
                min_choice = st.select_slider("Minute", options=minutes, value=minutes[0] if minutes else 10)
                gx_team = gx.filter((pl.col("team_id") == team_a_id) & (pl.col("minute") == min_choice))
                st.subheader(f"Gold buckets (min {min_choice}) — {team_a}")
                st.dataframe(gx_team.to_pandas())
                if not gx_team.is_empty():
                    st.bar_chart(gx_team.select(["bucket_gold", "winrate"]).to_pandas(), x="bucket_gold", y="winrate")
            else:
                st.info("Gold/XP metrics manquants.")

        with tabs[3]:
            if "series_team" in metrics:
                series_team = metrics["series_team"].join(team_names_df, on="team_id", how="left")
                st.subheader(f"Winrate par map_num — {team_a}")
                st.dataframe(series_team.filter(pl.col("team_id") == team_a_id).to_pandas())
                st.subheader("Distribution globale")
                st.dataframe(series_team.to_pandas())
            else:
                st.info("Séries metrics manquants.")

        with tabs[4]:
            if "elo_latest" in metrics and "elo_hist" in metrics:
                elo_latest = metrics["elo_latest"].join(team_names_df, on="team_id", how="left")
                st.subheader("Classement Elo (tracked teams)")
                st.dataframe(elo_latest.sort("elo", descending=True).to_pandas())

                elo_hist = metrics["elo_hist"].filter(pl.col("team_id") == team_a_id).sort("start_time")
                df_elo = elo_hist.to_pandas()
                if not df_elo.empty:
                    df_elo["start_dt"] = pd.to_datetime(df_elo["start_time"], unit="s")
                    st.subheader(f"Elo history — {team_a}")
                    st.line_chart(df_elo.set_index("start_dt")["rating_post"])
            else:
                st.info("Elo metrics manquants.")
    else:
        st.info("Metrics pré-calculées non trouvées (data/metrics). Lance `make precompute`.")

    # Match detail picker (within team A matches)
    st.subheader("Match detail (Team A scope)")
    match_ids = team_a_matches.select("match_id")["match_id"].to_list()
    if match_ids:
        chosen_match = st.selectbox("Match id", match_ids)
        obj_df = objectives_timeline(objectives, chosen_match).to_pandas()
        # Map team codes to labels (when present)
        team_label = {2: "radiant", 3: "dire"}
        if "team" in obj_df.columns:
            obj_df["team_label"] = obj_df["team"].map(team_label)
        else:
            obj_df["team_label"] = None
        obj_df["building_name"] = obj_df["key"].apply(_friendly_building_name) if "key" in obj_df.columns else None
        obj_df["time_min"] = obj_df["time"] / 60
        keep_cols = [
            "time",
            "time_min",
            "type",
            "team_label",
            "value",
            "slot",
            "player_slot",
            "unit",
            "building_name",
            "key",
            "killer",
        ]
        table_cols = [c for c in keep_cols if c in obj_df.columns]
        col_t1, col_t2 = st.columns([2, 1])
        col_t1.write("Objectives / events")
        col_t1.dataframe(obj_df[table_cols].sort_values("time").head(200))
        col_t2.write("Events distribution")
        if not obj_df.empty:
            col_t2.bar_chart(obj_df["type"].value_counts())
            st.vega_lite_chart(
                obj_df,
                {
                    "mark": {"type": "circle", "tooltip": True, "size": 60},
                    "encoding": {
                        "x": {"field": "time_min", "type": "quantitative", "title": "Time (min)"},
                        "y": {"field": "type", "type": "nominal"},
                        "color": {"field": "team_label", "type": "nominal", "legend": {"title": "side"}},
                        "tooltip": [
                            {"field": "time_min", "type": "quantitative", "title": "Time (min)", "format": ".1f"},
                            {"field": "type", "type": "nominal"},
                            {"field": "team_label", "type": "nominal"},
                            {"field": "building_name", "type": "nominal", "title": "building"},
                            {"field": "unit", "type": "nominal", "title": "unit"},
                            {"field": "key", "type": "nominal", "title": "key"},
                        ],
                    },
                },
                use_container_width=True,
            )
        if chosen_match in raw_map:
            raw_match = raw_map[chosen_match]
            st.markdown(
                f"**{raw_match.get('radiant_name','Radiant')} vs {raw_match.get('dire_name','Dire')}** — "
                f"Score {raw_match.get('radiant_score')} - {raw_match.get('dire_score')}"
            )
            gold = raw_match.get("radiant_gold_adv")
            xp = raw_match.get("radiant_xp_adv")
            if gold:
                st.markdown("**Radiant gold advantage**")
                st.line_chart(pd.DataFrame({"gold_adv": gold}))
            if xp:
                st.markdown("**Radiant XP advantage**")
                st.line_chart(pd.DataFrame({"xp_adv": xp}))
    else:
        st.info("No matches for this team.")

    st.divider()
    st.header("Global analysis (all games)")

    # Correlation matrix on numeric match-level fields (raw) with safe casting
    num_cols = [c for c, dtype in matches_raw.schema.items() if dtype.is_numeric()]
    if num_cols:
        st.subheader("Match-level correlations")
        # Avoid wide int64 that break pyarrow conversion
        corr_df = (
            matches_raw.select(num_cols)
            .with_columns([pl.col(c).cast(pl.Float64) for c in num_cols])
            .to_pandas()
            .corr()
        )
        st.dataframe(corr_df)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_df, cmap="RdBu", center=0, annot=False, ax=ax)
            ax.set_title("Match-level correlations")
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Download correlation heatmap (PNG)",
                data=buf.getvalue(),
                file_name="correlation_heatmap.png",
                mime="image/png",
            )
            plt.close(fig)
        except Exception as e:
            st.info(f"Heatmap not available: {e}")
    else:
        st.info("No numeric columns for correlation.")

    # Objectives timing overview
    st.subheader("Objective timing overview")
    obj_df_all = objectives.with_columns((pl.col("time") / 60).alias("time_min")).to_pandas()
    obj_df_all["building_name"] = obj_df_all["key"].apply(_friendly_building_name) if "key" in obj_df_all.columns else None
    type_options = obj_df_all["type"].dropna().unique().tolist()
    default_types = [t for t in ["building_kill", "CHAT_MESSAGE_ROSHAN_KILL", "CHAT_MESSAGE_AEGIS"] if t in type_options]
    selected_types = st.multiselect("Objective types", type_options, default=default_types)
    filtered_obj = obj_df_all if not selected_types else obj_df_all[obj_df_all["type"].isin(selected_types)]
    if not filtered_obj.empty:
        st.write("Scatter of objectives over time (all matches)")
        st.vega_lite_chart(
            filtered_obj,
            {
                "mark": {"type": "circle", "tooltip": True, "size": 40},
                "encoding": {
                    "x": {"field": "time_min", "type": "quantitative", "title": "Time (min)"},
                    "y": {"field": "type", "type": "nominal"},
                    "color": {"field": "type", "type": "nominal"},
                    "tooltip": [
                        {"field": "time_min", "type": "quantitative", "format": ".1f"},
                        {"field": "type", "type": "nominal"},
                        {"field": "building_name", "type": "nominal", "title": "building"},
                        {"field": "unit", "type": "nominal", "title": "unit"},
                        {"field": "key", "type": "nominal", "title": "key"},
                    ],
                },
            },
            use_container_width=True,
        )
        st.write("Objectives density by time bin (5 min)")
        filtered_obj["time_bin"] = (filtered_obj["time_min"] // 5 * 5).astype(int)
        density = filtered_obj.groupby(["type", "time_bin"]).size().reset_index(name="count")
        st.vega_lite_chart(
            density,
            {
                "mark": "bar",
                "encoding": {
                    "x": {"field": "time_bin", "type": "quantitative", "bin": False, "title": "Time bin (min)"},
                    "y": {"field": "count", "type": "quantitative"},
                    "color": {"field": "type", "type": "nominal"},
                },
            },
            use_container_width=True,
        )
    else:
        st.info("No objectives for selected filters.")


if __name__ == "__main__":
    main()
