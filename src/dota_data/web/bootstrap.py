from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .models import Extra, Match, ObjectiveRow, PlayerRow, Series, Team, TeamAlias, TeamfightRow


@dataclass(frozen=True)
class BootstrapPaths:
    processed_dir: Path
    teams_csv: Path
    aliases_csv: Path


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in row.items()}


def _iter_parquet_rows(path: Path, *, batch_size: int = 10_000) -> Iterable[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def _safe_batch_rows(num_cols: int, requested: int) -> int:
    # Postgres has a hard limit of ~65535 bind parameters per statement.
    # Keep some margin and compute a conservative max rows-per-insert.
    return max(1, min(int(requested), 60_000 // max(1, int(num_cols))))


def upsert_teams_and_aliases(db: Session, paths: BootstrapPaths) -> None:
    teams_df = pd.read_csv(paths.teams_csv)
    aliases_df = pd.read_csv(paths.aliases_csv)

    # Ensure all team ids referenced by aliases exist (even if we don't have metadata for them yet).
    all_team_ids: set[int] = set()
    for tid in teams_df.get("TeamID", []):
        if pd.isna(tid):
            continue
        all_team_ids.add(int(tid))
    for col in ("alias_team_id", "canonical_team_id"):
        if col not in aliases_df.columns:
            continue
        for tid in aliases_df[col]:
            if pd.isna(tid):
                continue
            all_team_ids.add(int(tid))

    if all_team_ids:
        base_rows = [{"team_id": int(tid), "name": None, "is_tracked": False} for tid in sorted(all_team_ids)]
        stmt = insert(Team).values(base_rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=[Team.team_id])
        db.execute(stmt)

    teams_rows = []
    for r in teams_df.to_dict(orient="records"):
        tid = r.get("TeamID")
        if pd.isna(tid):
            continue
        teams_rows.append(
            {
                "team_id": int(tid),
                "name": None if pd.isna(r.get("TeamName")) else str(r.get("TeamName")),
                "is_tracked": False if pd.isna(r.get("is_tracked")) else bool(r.get("is_tracked")),
            }
        )
    if teams_rows:
        stmt = insert(Team).values(teams_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Team.team_id],
            set_={
                "name": stmt.excluded.name,
                "is_tracked": stmt.excluded.is_tracked,
            },
        )
        db.execute(stmt)

    aliases_rows = []
    for r in aliases_df.to_dict(orient="records"):
        a = r.get("alias_team_id")
        c = r.get("canonical_team_id")
        if pd.isna(a) or pd.isna(c):
            continue
        aliases_rows.append({"alias_team_id": int(a), "canonical_team_id": int(c)})
    if aliases_rows:
        stmt = insert(TeamAlias).values(aliases_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[TeamAlias.alias_team_id],
            set_={"canonical_team_id": stmt.excluded.canonical_team_id},
        )
        db.execute(stmt)


def bootstrap_processed_tables(
    db: Session,
    processed_dir: Path,
    *,
    batch_size: int = 10_000,
    include_big: bool = True,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    matches_path = processed_dir / "matches.parquet"
    if matches_path.exists():
        flush_n = _safe_batch_rows(8, batch_size)
        rows = []
        for row in _iter_parquet_rows(matches_path, batch_size=batch_size):
            mid = row.get("match_id")
            if mid is None:
                continue
            rows.append(
                {
                    "match_id": int(mid),
                    "start_time": int(row.get("start_time")) if isinstance(row.get("start_time"), int) else None,
                    "leagueid": int(row.get("leagueid")) if isinstance(row.get("leagueid"), int) else None,
                    "series_id": int(row.get("series_id")) if isinstance(row.get("series_id"), int) else None,
                    "radiant_team_id": int(row.get("radiant_team_id")) if isinstance(row.get("radiant_team_id"), int) else None,
                    "dire_team_id": int(row.get("dire_team_id")) if isinstance(row.get("dire_team_id"), int) else None,
                    "radiant_win": bool(row.get("radiant_win")) if row.get("radiant_win") is not None else None,
                    "payload": _row_payload(row),
                }
            )
            if len(rows) >= flush_n:
                stmt = insert(Match).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Match.match_id],
                    set_={
                        "start_time": stmt.excluded.start_time,
                        "leagueid": stmt.excluded.leagueid,
                        "series_id": stmt.excluded.series_id,
                        "radiant_team_id": stmt.excluded.radiant_team_id,
                        "dire_team_id": stmt.excluded.dire_team_id,
                        "radiant_win": stmt.excluded.radiant_win,
                        "payload": stmt.excluded.payload,
                    },
                )
                db.execute(stmt)
                counts["matches"] = counts.get("matches", 0) + len(rows)
                rows = []
        if rows:
            stmt = insert(Match).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Match.match_id],
                set_={
                    "start_time": stmt.excluded.start_time,
                    "leagueid": stmt.excluded.leagueid,
                    "series_id": stmt.excluded.series_id,
                    "radiant_team_id": stmt.excluded.radiant_team_id,
                    "dire_team_id": stmt.excluded.dire_team_id,
                    "radiant_win": stmt.excluded.radiant_win,
                    "payload": stmt.excluded.payload,
                },
            )
            db.execute(stmt)
            counts["matches"] = counts.get("matches", 0) + len(rows)

    series_path = processed_dir / "series.parquet"
    if series_path.exists():
        flush_n = _safe_batch_rows(12, batch_size)
        rows = []
        for row in _iter_parquet_rows(series_path, batch_size=batch_size):
            leagueid = row.get("leagueid")
            series_id = row.get("series_id")
            if leagueid is None or series_id is None:
                continue
            rows.append(
                {
                    "leagueid": int(leagueid),
                    "series_id": int(series_id),
                    "start_time_min": int(row.get("start_time_min")) if isinstance(row.get("start_time_min"), int) else None,
                    "start_time_max": int(row.get("start_time_max")) if isinstance(row.get("start_time_max"), int) else None,
                    "team_a": int(row.get("team_a")) if isinstance(row.get("team_a"), int) else None,
                    "team_b": int(row.get("team_b")) if isinstance(row.get("team_b"), int) else None,
                    "score_team_a": int(row.get("score_team_a")) if isinstance(row.get("score_team_a"), int) else None,
                    "score_team_b": int(row.get("score_team_b")) if isinstance(row.get("score_team_b"), int) else None,
                    "winner_team_id": int(row.get("winner_team_id")) if isinstance(row.get("winner_team_id"), int) else None,
                    "bo_type": int(row.get("bo_type")) if isinstance(row.get("bo_type"), int) else None,
                    "series_type_raw": int(row.get("series_type_raw")) if isinstance(row.get("series_type_raw"), int) else None,
                    "payload": _row_payload(row),
                }
            )
            if len(rows) >= flush_n:
                stmt = insert(Series).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Series.leagueid, Series.series_id],
                    set_={
                        "start_time_min": stmt.excluded.start_time_min,
                        "start_time_max": stmt.excluded.start_time_max,
                        "team_a": stmt.excluded.team_a,
                        "team_b": stmt.excluded.team_b,
                        "score_team_a": stmt.excluded.score_team_a,
                        "score_team_b": stmt.excluded.score_team_b,
                        "winner_team_id": stmt.excluded.winner_team_id,
                        "bo_type": stmt.excluded.bo_type,
                        "series_type_raw": stmt.excluded.series_type_raw,
                        "payload": stmt.excluded.payload,
                    },
                )
                db.execute(stmt)
                counts["series"] = counts.get("series", 0) + len(rows)
                rows = []
        if rows:
            stmt = insert(Series).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Series.leagueid, Series.series_id],
                set_={
                    "start_time_min": stmt.excluded.start_time_min,
                    "start_time_max": stmt.excluded.start_time_max,
                    "team_a": stmt.excluded.team_a,
                    "team_b": stmt.excluded.team_b,
                    "score_team_a": stmt.excluded.score_team_a,
                    "score_team_b": stmt.excluded.score_team_b,
                    "winner_team_id": stmt.excluded.winner_team_id,
                    "bo_type": stmt.excluded.bo_type,
                    "series_type_raw": stmt.excluded.series_type_raw,
                    "payload": stmt.excluded.payload,
                },
            )
            db.execute(stmt)
            counts["series"] = counts.get("series", 0) + len(rows)

    extras_path = processed_dir / "extras.parquet"
    if extras_path.exists():
        flush_n = _safe_batch_rows(4, batch_size)
        rows = []
        for row in _iter_parquet_rows(extras_path, batch_size=batch_size):
            mid = row.get("match_id")
            if mid is None:
                continue
            rows.append(
                {
                    "match_id": int(mid),
                    "radiant_gold_adv": _jsonable(row.get("radiant_gold_adv")),
                    "radiant_xp_adv": _jsonable(row.get("radiant_xp_adv")),
                    "picks_bans": _jsonable(row.get("picks_bans")),
                }
            )
            if len(rows) >= flush_n:
                stmt = insert(Extra).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Extra.match_id],
                    set_={
                        "radiant_gold_adv": stmt.excluded.radiant_gold_adv,
                        "radiant_xp_adv": stmt.excluded.radiant_xp_adv,
                        "picks_bans": stmt.excluded.picks_bans,
                    },
                )
                db.execute(stmt)
                counts["extras"] = counts.get("extras", 0) + len(rows)
                rows = []
        if rows:
            stmt = insert(Extra).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Extra.match_id],
                set_={
                    "radiant_gold_adv": stmt.excluded.radiant_gold_adv,
                    "radiant_xp_adv": stmt.excluded.radiant_xp_adv,
                    "picks_bans": stmt.excluded.picks_bans,
                },
            )
            db.execute(stmt)
            counts["extras"] = counts.get("extras", 0) + len(rows)

    if include_big:
        players_path = processed_dir / "players.parquet"
    else:
        players_path = Path("__skip__")
    if players_path.exists():
        flush_n = _safe_batch_rows(5, batch_size)
        inserted = 0
        rows = []
        for row in _iter_parquet_rows(players_path, batch_size=batch_size):
            rows.append(
                {
                    "match_id": int(row.get("match_id")) if isinstance(row.get("match_id"), int) else None,
                    "player_slot": int(row.get("player_slot")) if isinstance(row.get("player_slot"), int) else None,
                    "account_id": int(row.get("account_id")) if isinstance(row.get("account_id"), int) else None,
                    "hero_id": int(row.get("hero_id")) if isinstance(row.get("hero_id"), int) else None,
                    "payload": _row_payload(row),
                }
            )
            if len(rows) >= flush_n:
                db.execute(insert(PlayerRow).values(rows))
                inserted += len(rows)
                rows = []
        if rows:
            db.execute(insert(PlayerRow).values(rows))
            inserted += len(rows)
        counts["players"] = inserted

    if include_big:
        objectives_path = processed_dir / "objectives.parquet"
    else:
        objectives_path = Path("__skip__")
    if objectives_path.exists():
        flush_n = _safe_batch_rows(4, batch_size)
        inserted = 0
        rows = []
        for row in _iter_parquet_rows(objectives_path, batch_size=batch_size):
            rows.append(
                {
                    "match_id": int(row.get("match_id")) if isinstance(row.get("match_id"), int) else None,
                    "time": int(row.get("time")) if isinstance(row.get("time"), int) else None,
                    "objective_type": None if row.get("type") is None else str(row.get("type")),
                    "payload": _row_payload(row),
                }
            )
            if len(rows) >= flush_n:
                db.execute(insert(ObjectiveRow).values(rows))
                inserted += len(rows)
                rows = []
        if rows:
            db.execute(insert(ObjectiveRow).values(rows))
            inserted += len(rows)
        counts["objectives"] = inserted

    if include_big:
        teamfights_path = processed_dir / "teamfights.parquet"
    else:
        teamfights_path = Path("__skip__")
    if teamfights_path.exists():
        flush_n = _safe_batch_rows(3, batch_size)
        inserted = 0
        rows = []
        for row in _iter_parquet_rows(teamfights_path, batch_size=batch_size):
            rows.append(
                {
                    "match_id": int(row.get("match_id")) if isinstance(row.get("match_id"), int) else None,
                    "teamfight_index": int(row.get("teamfight_index")) if isinstance(row.get("teamfight_index"), int) else None,
                    "payload": _row_payload(row),
                }
            )
            if len(rows) >= flush_n:
                db.execute(insert(TeamfightRow).values(rows))
                inserted += len(rows)
                rows = []
        if rows:
            db.execute(insert(TeamfightRow).values(rows))
            inserted += len(rows)
        counts["teamfights"] = inserted

    return counts
