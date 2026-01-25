from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from redis import Redis
from rq import Worker
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Job
from .scrape import scrape_into_db
from .settings import REDIS_URL


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def execute_job(*, db_job_id: str, job_type: str, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    db: Session = SessionLocal()
    try:
        job_uuid = uuid.UUID(str(db_job_id))
        job = db.query(Job).filter(Job.id == job_uuid).one_or_none()
        if job is None:
            return {"ok": False, "error": "job not found"}
        job.status = "running"
        job.started_at = _now()
        job.logs = (job.logs or "") + f"[{job.started_at.isoformat()}] started {job_type}\n"
        db.add(job)
        db.commit()

        log_lines: list[str] = []
        result: dict[str, Any] = {"ok": True, "job_type": job_type}
        if job_type == "scrape":
            result = scrape_into_db(db, payload=job.payload, job_log=log_lines)
        elif job_type == "precompute":
            # Placeholder for now; will be wired to metrics computation later.
            time.sleep(0.5)
            result = {"ok": True, "job_type": job_type, "note": "placeholder"}
        else:
            result = {"ok": False, "job_type": job_type, "error": "unknown job_type"}

        job.status = "done" if result.get("ok") else "failed"
        job.finished_at = _now()
        job.result = result
        if log_lines:
            job.logs = (job.logs or "") + "".join(line + ("\n" if not line.endswith("\n") else "") for line in log_lines)
        job.logs = (job.logs or "") + f"[{job.finished_at.isoformat()}] {job.status}\n"
        db.add(job)
        db.commit()
        return result
    except Exception as exc:  # noqa: BLE001
        try:
            job_uuid = uuid.UUID(str(db_job_id))
            job = db.query(Job).filter(Job.id == job_uuid).one_or_none()
            if job is not None:
                job.status = "failed"
                job.finished_at = _now()
                job.error = repr(exc)
                job.logs = (job.logs or "") + f"[{job.finished_at.isoformat()}] failed: {exc!r}\n"
                db.add(job)
                db.commit()
        finally:
            return {"ok": False, "error": repr(exc)}
    finally:
        db.close()


def main() -> None:
    redis = Redis.from_url(REDIS_URL)
    worker = Worker(["default"], connection=redis)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
