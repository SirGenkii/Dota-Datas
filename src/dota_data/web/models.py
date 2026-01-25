from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(Text, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    logo_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_tracked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TeamAlias(Base):
    __tablename__ = "team_aliases"

    alias_team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    canonical_team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.team_id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    canonical_team: Mapped[Team] = relationship("Team")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_type: Mapped[str] = mapped_column(Text, index=True)  # scrape | precompute | bootstrap
    status: Mapped[str] = mapped_column(Text, index=True)  # created|running|done|failed
    valid: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class FetchEvent(Base):
    __tablename__ = "fetch_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    endpoint: Mapped[str] = mapped_column(Text, index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type: Mapped[str] = mapped_column(Text, index=True)  # scrape | precompute | bootstrap
    status: Mapped[str] = mapped_column(Text, index=True)  # queued|running|done|failed
    rq_job_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    requested_by_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    logs: Mapped[str | None] = mapped_column(Text, nullable=True)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    requested_by: Mapped[User | None] = relationship("User")


class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    start_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    leagueid: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    series_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    radiant_team_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    dire_team_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    radiant_win: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class MatchRaw(Base):
    __tablename__ = "match_raw"

    match_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSONB)


class Series(Base):
    __tablename__ = "series"

    leagueid: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    series_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    start_time_min: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    start_time_max: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    team_a: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    team_b: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    score_team_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_team_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    winner_team_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    bo_type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    series_type_raw: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (UniqueConstraint("leagueid", "series_id", name="uq_series_league_series"),)


class Extra(Base):
    __tablename__ = "extras"

    match_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    radiant_gold_adv: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    radiant_xp_adv: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    picks_bans: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)


class PlayerRow(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    player_slot: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    account_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    hero_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class ObjectiveRow(Base):
    __tablename__ = "objectives"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    time: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    objective_type: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class TeamfightRow(Base):
    __tablename__ = "teamfights"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    teamfight_index: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
