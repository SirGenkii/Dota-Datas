from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from sqlalchemy import text
from sqlalchemy.orm import Session

from .auth import authenticate_user, create_access_token, get_current_user, require_admin
from .db import get_db
from .models import Job, Team, User
from .settings import REDIS_URL


app = FastAPI(title="dota-data API")


def _rq() -> Queue:
    redis = Redis.from_url(REDIS_URL)
    return Queue("default", connection=redis)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    is_admin: bool
    is_active: bool


class CreateUserIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    is_admin: bool = False


class JobOut(BaseModel):
    id: uuid.UUID
    job_type: str
    status: str
    rq_job_id: str | None
    payload: dict[str, Any] | None
    result: dict[str, Any] | None = None
    error: str | None
    logs: str | None = None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RunOut(BaseModel):
    id: uuid.UUID
    run_type: str
    status: str
    valid: bool
    stats: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class CreateJobIn(BaseModel):
    payload: dict[str, Any] | None = None


@app.get("/health")
def health(db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    db.execute(text("SELECT 1"))
    redis = Redis.from_url(REDIS_URL)
    redis.ping()
    return {"ok": True}


@app.post("/auth/token", response_model=TokenOut)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenOut:
    user = authenticate_user(db, form.username, form.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    user.last_login_at = datetime.now(tz=timezone.utc)
    db.add(user)
    db.commit()
    token = create_access_token(subject=user.username)
    return TokenOut(access_token=token)


@app.get("/auth/me", response_model=UserOut)
def me(user: Annotated[User, Depends(get_current_user)]) -> UserOut:
    return UserOut(id=user.id, username=user.username, is_admin=user.is_admin, is_active=user.is_active)


@app.get("/admin/users", response_model=list[UserOut])
def list_users(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[UserOut]:
    users = db.query(User).order_by(User.created_at.desc()).limit(500).all()
    return [UserOut(id=u.id, username=u.username, is_admin=u.is_admin, is_active=u.is_active) for u in users]


@app.post("/admin/users", response_model=UserOut)
def create_user(
    data: CreateUserIn,
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> UserOut:
    from .auth import get_password_hash  # local import to avoid cycles

    if db.query(User).filter(User.username == data.username).first() is not None:
        raise HTTPException(status_code=400, detail="username already exists")
    u = User(username=data.username, password_hash=get_password_hash(data.password), is_admin=bool(data.is_admin))
    db.add(u)
    db.commit()
    db.refresh(u)
    return UserOut(id=u.id, username=u.username, is_admin=u.is_admin, is_active=u.is_active)


@app.get("/teams")
def list_teams(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    tracked_only: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    q = db.query(Team)
    if tracked_only:
        q = q.filter(Team.is_tracked.is_(True))
    teams = q.order_by(Team.is_tracked.desc(), Team.team_id.asc()).limit(min(1000, max(1, limit))).all()
    return {
        "requested_by": user.username,
        "items": [{"team_id": t.team_id, "name": t.name, "is_tracked": t.is_tracked} for t in teams],
    }


def _enqueue_job(db: Session, *, job_type: str, requested_by: User, payload: dict[str, Any] | None) -> Job:
    job = Job(job_type=job_type, status="queued", requested_by_user_id=requested_by.id, payload=payload)
    db.add(job)
    db.commit()
    db.refresh(job)

    rq = _rq()
    # NOTE: Queue.enqueue has a reserved kwarg `job_id` (RQ's job id). We set it to our DB job id.
    rq_job = rq.enqueue(
        "src.dota_data.web.worker.execute_job",
        job_type=job_type,
        db_job_id=str(job.id),
        job_id=str(job.id),
    )
    job.rq_job_id = rq_job.id
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@app.post("/jobs/scrape", response_model=JobOut)
def create_scrape_job(
    data: CreateJobIn,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> JobOut:
    job = _enqueue_job(db, job_type="scrape", requested_by=user, payload=data.payload)
    return JobOut.model_validate(job, from_attributes=True)


@app.post("/jobs/precompute", response_model=JobOut)
def create_precompute_job(
    data: CreateJobIn,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> JobOut:
    job = _enqueue_job(db, job_type="precompute", requested_by=user, payload=data.payload)
    return JobOut.model_validate(job, from_attributes=True)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(
    job_id: uuid.UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> JobOut:
    job = db.query(Job).filter(Job.id == job_id).one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not user.is_admin and job.requested_by_user_id != user.id:
        raise HTTPException(status_code=403, detail="forbidden")
    return JobOut.model_validate(job, from_attributes=True)


@app.get("/runs", response_model=list[RunOut])
def list_runs(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    run_type: str | None = None,
    limit: int = 50,
) -> list[RunOut]:
    from .models import Run

    q = db.query(Run)
    if run_type:
        q = q.filter(Run.run_type == run_type)
    runs = q.order_by(Run.started_at.desc()).limit(min(200, max(1, limit))).all()
    return [RunOut.model_validate(r, from_attributes=True) for r in runs]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(
    run_id: uuid.UUID,
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> RunOut:
    from .models import Run

    run = db.query(Run).filter(Run.id == run_id).one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunOut.model_validate(run, from_attributes=True)
