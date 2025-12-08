"""Utilities for loading and transforming Dota match data."""

from .io import (
    load_raw_matches,
    matches_table,
    players_table,
    objectives_table,
    teamfights_table,
    write_parquet_tables,
    summarize_raw,
)
from .transforms import read_processed_tables, lane_phase_features, match_header
from .stats import kill_participation, vision_stats, team_scores
from .viz import objectives_timeline, teamfights_for_match
from .metadata import (
    build_team_dictionary,
    build_player_dictionary,
    build_hero_counts,
    load_hero_dictionary,
)
from .api import (
    load_api_key,
    build_session,
    load_team_list,
    fetch_team_matches,
    annotate_matches_with_team,
    filter_matches_since,
    unique_match_ids,
    fetch_match_detail,
    fetch_match_details,
    wrap_raw_match,
    write_json,
    fetch_matches_chunked,
)

__all__ = [
    "load_raw_matches",
    "matches_table",
    "players_table",
    "objectives_table",
    "teamfights_table",
    "write_parquet_tables",
    "summarize_raw",
    "read_processed_tables",
    "lane_phase_features",
    "match_header",
    "kill_participation",
    "vision_stats",
    "team_scores",
    "objectives_timeline",
    "teamfights_for_match",
    "build_team_dictionary",
    "build_player_dictionary",
    "build_hero_counts",
    "load_hero_dictionary",
    "load_api_key",
    "build_session",
    "load_team_list",
    "fetch_team_matches",
    "annotate_matches_with_team",
    "filter_matches_since",
    "unique_match_ids",
    "fetch_match_detail",
    "fetch_match_details",
    "wrap_raw_match",
    "write_json",
    "fetch_matches_chunked",
]
