from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import polars as pl
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.dota_data import io as io_mod
from src.dota_data.api import BASE_URL, build_session, wrap_raw_match

from .models import Extra, FetchEvent, Match, MatchRaw, ObjectiveRow, PlayerRow, Run, Series, Team, TeamAlias, TeamfightRow


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _existing_match_ids(db: Session, match_ids: list[int]) -> set[int]:
    if not match_ids:
        return set()
    existing: set[int] = set()
    chunk = 5000
    for i in range(0, len(match_ids), chunk):
        rows = db.query(Match.match_id).filter(Match.match_id.in_(match_ids[i : i + chunk])).all()
        existing.update(int(r[0]) for r in rows)
    return existing


def _alias_map(db: Session) -> dict[int, int]:
    rows = db.query(TeamAlias.alias_team_id, TeamAlias.canonical_team_id).all()
    out: dict[int, int] = {}
    for a, c in rows:
        if a is None or c is None:
            continue
        ai = int(a)
        ci = int(c)
        if ai != ci:
            out[ai] = ci
    return out


def _max_start_time(db: Session) -> int:
    value = db.query(Match.start_time).order_by(Match.start_time.desc()).limit(1).scalar()
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


@dataclass(frozen=True)
class ScrapeConfig:
    since_start_time: int
    limit: int = 100
    max_pages: int = 3
    max_new: int | None = None
    sleep_pages: float = 0.2
    sleep_match_detail: float = 1.0
    timeout_team_matches: int = 60
    timeout_match_detail: int = 90
    retries: int = 2
    backoff: float = 1.5


def _record_fetch(
    db: Session,
    *,
    run_id: Any,
    endpoint: str,
    requested_at: datetime,
    duration_ms: int,
    status_code: int | None,
    error: str | None,
    meta: dict[str, Any] | None,
) -> None:
    db.add(
        FetchEvent(
            run_id=run_id,
            endpoint=endpoint,
            requested_at=requested_at,
            duration_ms=duration_ms,
            status_code=status_code,
            error=error,
            meta=meta,
        )
    )


def _get_json(session: Any, url: str, *, timeout: int) -> tuple[Any, int | None, int, str | None]:
    started = time.time()
    status_code = None
    try:
        resp = session.get(url, timeout=timeout)
        status_code = int(resp.status_code)
        resp.raise_for_status()
        return resp.json(), status_code, int((time.time() - started) * 1000), None
    except Exception as exc:  # noqa: BLE001
        return None, status_code, int((time.time() - started) * 1000), repr(exc)


def _get_json_with_retries(
    db: Session,
    *,
    run_id: Any,
    session: Any,
    endpoint: str,
    timeout: int,
    retries: int,
    backoff: float,
    meta: dict[str, Any] | None = None,
) -> Any:
    url = f"{BASE_URL}{endpoint}"
    last_data: Any = None
    last_error: str | None = None
    attempts = max(1, int(retries) + 1)
    for i in range(attempts):
        attempt = i + 1
        requested_at = _now()
        data, status_code, duration_ms, error = _get_json(session, url, timeout=timeout)
        _record_fetch(
            db,
            run_id=run_id,
            endpoint=endpoint,
            requested_at=requested_at,
            duration_ms=duration_ms,
            status_code=status_code,
            error=error,
            meta={**(meta or {}), "attempt": attempt},
        )
        if error is None:
            return data
        last_data = data
        last_error = error
        if i < int(retries):
            time.sleep(max(0.0, float(backoff)) * (2 ** i))
    # give up
    return last_data if last_error is None else None


def discover_new_match_ids(
    db: Session,
    *,
    session: Any,
    run_id: Any,
    team_ids: list[int],
    cfg: ScrapeConfig,
) -> list[int]:
    candidates: dict[int, int] = {}  # match_id -> start_time
    for team_id in team_ids:
        for page in range(int(cfg.max_pages)):
            offset = page * int(cfg.limit)
            endpoint = f"/teams/{team_id}/matches?limit={int(cfg.limit)}&offset={int(offset)}"
            data = _get_json_with_retries(
                db,
                run_id=run_id,
                session=session,
                endpoint=endpoint,
                timeout=int(cfg.timeout_team_matches),
                retries=int(cfg.retries),
                backoff=float(cfg.backoff),
                meta={"team_id": team_id, "limit": int(cfg.limit), "offset": int(offset)},
            )
            if not isinstance(data, list):
                break

            page_times: list[int] = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                mid = row.get("match_id")
                st = row.get("start_time")
                if isinstance(st, int):
                    page_times.append(st)
                if not isinstance(mid, int) or not isinstance(st, int):
                    continue
                if st <= int(cfg.since_start_time):
                    continue
                candidates[mid] = max(candidates.get(mid, 0), st)

            if cfg.since_start_time and page_times and min(page_times) <= int(cfg.since_start_time):
                break
            if cfg.sleep_pages and cfg.sleep_pages > 0:
                time.sleep(float(cfg.sleep_pages))

    if not candidates:
        return []

    ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    match_ids = [mid for mid, _ in ordered]

    # Enforce max_new (newest first).
    if cfg.max_new is not None:
        match_ids = match_ids[: int(cfg.max_new)]

    # Final dedupe against DB.
    existing = _existing_match_ids(db, match_ids)
    match_ids = [mid for mid in match_ids if mid not in existing]
    return match_ids


def _df_to_dicts(df: pl.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.is_empty():
        return []
    return [dict(r) for r in df.to_dicts()]


def _jsonable(v: Any) -> Any:
    try:
        import json

        json.dumps(v)
        return v
    except Exception:
        return str(v)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in row.items()}


def upsert_parsed_tables(db: Session, wrapped: list[dict[str, Any]], *, alias_map: dict[int, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not wrapped:
        return counts

    matches_df = io_mod.matches_table(wrapped, alias_map=alias_map)
    series_df = io_mod.series_summary_table(matches_df)
    players_df = io_mod.players_table(wrapped)
    objectives_df = io_mod.objectives_table(wrapped)
    teamfights_df = io_mod.teamfights_table(wrapped)
    extras_df = io_mod.extras_table(wrapped)

    match_rows = []
    for row in _df_to_dicts(matches_df):
        mid = row.get("match_id")
        if mid is None:
            continue
        match_rows.append(
            {
                "match_id": int(mid),
                "start_time": int(row.get("start_time")) if isinstance(row.get("start_time"), int) else None,
                "leagueid": int(row.get("leagueid")) if isinstance(row.get("leagueid"), int) else None,
                "series_id": int(row.get("series_id")) if isinstance(row.get("series_id"), int) else None,
                "radiant_team_id": int(row.get("radiant_team_id")) if isinstance(row.get("radiant_team_id"), int) else None,
                "dire_team_id": int(row.get("dire_team_id")) if isinstance(row.get("dire_team_id"), int) else None,
                "radiant_win": bool(row.get("radiant_win")) if row.get("radiant_win") is not None else None,
                "payload": _payload(row),
            }
        )
    if match_rows:
        stmt = insert(Match).values(match_rows)
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
        counts["matches"] = len(match_rows)

    series_rows = []
    for row in _df_to_dicts(series_df):
        leagueid = row.get("leagueid")
        sid = row.get("series_id")
        if leagueid is None or sid is None:
            continue
        series_rows.append(
            {
                "leagueid": int(leagueid),
                "series_id": int(sid),
                "start_time_min": int(row.get("start_time_min")) if isinstance(row.get("start_time_min"), int) else None,
                "start_time_max": int(row.get("start_time_max")) if isinstance(row.get("start_time_max"), int) else None,
                "team_a": int(row.get("team_a")) if isinstance(row.get("team_a"), int) else None,
                "team_b": int(row.get("team_b")) if isinstance(row.get("team_b"), int) else None,
                "score_team_a": int(row.get("score_team_a")) if isinstance(row.get("score_team_a"), int) else None,
                "score_team_b": int(row.get("score_team_b")) if isinstance(row.get("score_team_b"), int) else None,
                "winner_team_id": int(row.get("winner_team_id")) if isinstance(row.get("winner_team_id"), int) else None,
                "bo_type": int(row.get("bo_type")) if isinstance(row.get("bo_type"), int) else None,
                "series_type_raw": int(row.get("series_type_raw")) if isinstance(row.get("series_type_raw"), int) else None,
                "payload": _payload(row),
            }
        )
    if series_rows:
        stmt = insert(Series).values(series_rows)
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
        counts["series"] = len(series_rows)

    extra_rows = []
    for row in _df_to_dicts(extras_df):
        mid = row.get("match_id")
        if mid is None:
            continue
        extra_rows.append(
            {
                "match_id": int(mid),
                "radiant_gold_adv": _jsonable(row.get("radiant_gold_adv")),
                "radiant_xp_adv": _jsonable(row.get("radiant_xp_adv")),
                "picks_bans": _jsonable(row.get("picks_bans")),
            }
        )
    if extra_rows:
        stmt = insert(Extra).values(extra_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Extra.match_id],
            set_={
                "radiant_gold_adv": stmt.excluded.radiant_gold_adv,
                "radiant_xp_adv": stmt.excluded.radiant_xp_adv,
                "picks_bans": stmt.excluded.picks_bans,
            },
        )
        db.execute(stmt)
        counts["extras"] = len(extra_rows)

    match_ids = [int(r.get("match_id")) for r in match_rows if r.get("match_id") is not None]
    if match_ids:
        db.query(PlayerRow).filter(PlayerRow.match_id.in_(match_ids)).delete(synchronize_session=False)
        db.query(ObjectiveRow).filter(ObjectiveRow.match_id.in_(match_ids)).delete(synchronize_session=False)
        db.query(TeamfightRow).filter(TeamfightRow.match_id.in_(match_ids)).delete(synchronize_session=False)

    player_rows = []
    for row in _df_to_dicts(players_df):
        player_rows.append(
            {
                "match_id": int(row.get("match_id")) if isinstance(row.get("match_id"), int) else None,
                "player_slot": int(row.get("player_slot")) if isinstance(row.get("player_slot"), int) else None,
                "account_id": int(row.get("account_id")) if isinstance(row.get("account_id"), int) else None,
                "hero_id": int(row.get("hero_id")) if isinstance(row.get("hero_id"), int) else None,
                "payload": _payload(row),
            }
        )
    if player_rows:
        db.execute(insert(PlayerRow).values(player_rows))
        counts["players"] = len(player_rows)

    objective_rows = []
    for row in _df_to_dicts(objectives_df):
        objective_rows.append(
            {
                "match_id": int(row.get("match_id")) if isinstance(row.get("match_id"), int) else None,
                "time": int(row.get("time")) if isinstance(row.get("time"), int) else None,
                "objective_type": None if row.get("type") is None else str(row.get("type")),
                "payload": _payload(row),
            }
        )
    if objective_rows:
        db.execute(insert(ObjectiveRow).values(objective_rows))
        counts["objectives"] = len(objective_rows)

    tf_rows = []
    for row in _df_to_dicts(teamfights_df):
        tf_rows.append(
            {
                "match_id": int(row.get("match_id")) if isinstance(row.get("match_id"), int) else None,
                "teamfight_index": int(row.get("teamfight_index")) if isinstance(row.get("teamfight_index"), int) else None,
                "payload": _payload(row),
            }
        )
    if tf_rows:
        db.execute(insert(TeamfightRow).values(tf_rows))
        counts["teamfights"] = len(tf_rows)

    return counts


def scrape_into_db(db: Session, *, payload: dict[str, Any] | None, job_log: list[str]) -> dict[str, Any]:
    payload = payload or {}

    since_ts = payload.get("since_start_time")
    if since_ts is None:
        since_ts = _max_start_time(db)
    try:
        since_ts_int = int(since_ts)
    except Exception:
        since_ts_int = _max_start_time(db)

    cfg = ScrapeConfig(
        since_start_time=since_ts_int,
        limit=int(payload.get("limit", 100)),
        max_pages=int(payload.get("max_pages", 3)),
        max_new=(int(payload["max_new"]) if payload.get("max_new") is not None else None),
        sleep_pages=float(payload.get("sleep_pages", 0.2)),
        sleep_match_detail=float(payload.get("sleep_match_detail", 1.0)),
        timeout_team_matches=int(payload.get("timeout_team_matches", 60)),
        timeout_match_detail=int(payload.get("timeout_match_detail", 90)),
        retries=int(payload.get("retries", 2)),
        backoff=float(payload.get("backoff", 1.5)),
    )

    tracked_teams = db.query(Team).filter(Team.is_tracked.is_(True)).order_by(Team.team_id.asc()).all()
    team_ids = [int(t.team_id) for t in tracked_teams]
    requested_team_ids = payload.get("team_ids")
    if isinstance(requested_team_ids, list) and requested_team_ids:
        wanted = {int(x) for x in requested_team_ids if isinstance(x, (int, str)) and str(x).isdigit()}
        team_ids = [tid for tid in team_ids if tid in wanted]
    if not team_ids:
        return {"ok": False, "error": "no tracked teams in DB"}

    run = Run(run_type="scrape", status="running", valid=True, config={"payload": payload, "cfg": cfg.__dict__})
    db.add(run)
    db.commit()
    db.refresh(run)

    api_key = os.getenv("OPENDOTA_KEY") or None
    session = build_session(api_key=api_key)

    job_log.append(f"[{_now().isoformat()}] scrape since_start_time={cfg.since_start_time} tracked_teams={len(team_ids)}")

    match_ids = discover_new_match_ids(db, session=session, run_id=run.id, team_ids=team_ids, cfg=cfg)
    db.commit()
    job_log.append(f"[{_now().isoformat()}] discovered new matches={len(match_ids)}")

    if not match_ids:
        run.status = "done"
        run.stats = {"new_matches": 0}
        run.finished_at = _now()
        db.add(run)
        db.commit()
        return {"ok": True, "run_id": str(run.id), "new_matches": 0}

    alias_map = _alias_map(db)

    fetched = 0
    parsed_counts: dict[str, int] = {}
    errors: list[dict[str, Any]] = []

    chunk_size = int(payload.get("chunk_size", 20))
    for i in range(0, len(match_ids), chunk_size):
        chunk_ids = match_ids[i : i + chunk_size]
        raw_payloads: list[dict[str, Any]] = []
        for mid in chunk_ids:
            endpoint = f"/matches/{mid}"
            data = _get_json_with_retries(
                db,
                run_id=run.id,
                session=session,
                endpoint=endpoint,
                timeout=int(cfg.timeout_match_detail),
                retries=int(cfg.retries),
                backoff=float(cfg.backoff),
                meta={"match_id": mid},
            )
            if not isinstance(data, dict):
                errors.append({"match_id": mid, "error": "fetch failed"})
            else:
                raw_payloads.append(data)
            fetched += 1
            if cfg.sleep_match_detail and cfg.sleep_match_detail > 0:
                time.sleep(float(cfg.sleep_match_detail))

        # Persist raw
        for match in raw_payloads:
            mid = match.get("match_id")
            if isinstance(mid, int):
                stmt = insert(MatchRaw).values({"match_id": int(mid), "payload": match, "fetched_at": _now()})
                stmt = stmt.on_conflict_do_update(
                    index_elements=[MatchRaw.match_id],
                    set_={"payload": stmt.excluded.payload, "fetched_at": stmt.excluded.fetched_at},
                )
                db.execute(stmt)

        # Parse + upsert tables
        wrapped = [wrap_raw_match(m) for m in raw_payloads]
        counts = upsert_parsed_tables(db, wrapped, alias_map=alias_map)
        for k, v in counts.items():
            parsed_counts[k] = parsed_counts.get(k, 0) + int(v)
        db.commit()

        job_log.append(f"[{_now().isoformat()}] chunk {i//chunk_size+1}: fetched={len(chunk_ids)} parsed={counts}")

    run.status = "done"
    run.valid = len(errors) == 0
    run.stats = {"requested": len(match_ids), "fetched": fetched, "parsed": parsed_counts, "errors": len(errors)}
    if errors:
        run.error = str(errors[:5])
    run.finished_at = _now()
    db.add(run)
    db.commit()

    return {
        "ok": True,
        "run_id": str(run.id),
        "requested": len(match_ids),
        "fetched": fetched,
        "parsed": parsed_counts,
        "errors": errors,
    }
