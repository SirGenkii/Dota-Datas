from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.opendota.com/api"


def load_api_key(env_var: str = "OPENDOTA_KEY", load_env_file: bool = True) -> str:
    """Return the OpenDota API key (env var OPENDOTA_KEY by default)."""
    if load_env_file:
        load_dotenv()
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"API key not found in environment variable {env_var}")
    return key


def build_session(api_key: Optional[str] = None, user_agent: str = "dota-data-v2/0.1") -> requests.Session:
    """Create a requests session with API key and UA."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    if api_key:
        session.params = {"api_key": api_key}
    return session


def load_team_list(csv_path: Path | str = "data/teams_to_look.csv") -> List[Dict[str, Any]]:
    """Load teams (TeamName, TeamID) from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Teams CSV not found: {path}")
    teams: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                team_id = int(row.get("TeamID", "").strip())
            except (TypeError, ValueError, AttributeError):
                continue
            teams.append({"TeamName": row.get("TeamName", ""), "TeamID": team_id})
    return teams


def _request_json(session: requests.Session, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    url = f"{BASE_URL}{path}"
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_team_matches(
    team_id: int,
    session: requests.Session,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch matches for a team from /teams/{team_id}/matches."""
    params: Dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    data = _request_json(session, f"/teams/{team_id}/matches", params=params, timeout=timeout)
    return data if isinstance(data, list) else []


def annotate_matches_with_team(matches: Iterable[Dict[str, Any]], team_id: int, team_name: str = "") -> List[Dict[str, Any]]:
    """Add source team metadata to match rows."""
    annotated: List[Dict[str, Any]] = []
    for m in matches:
        row = dict(m) if isinstance(m, dict) else {}
        row["_source_team_id"] = team_id
        row["_source_team_name"] = team_name
        annotated.append(row)
    return annotated


def filter_matches_since(matches: Iterable[Dict[str, Any]], min_start_time: int) -> List[Dict[str, Any]]:
    """Keep matches with start_time >= min_start_time (unix)."""
    return [m for m in matches if isinstance(m, dict) and m.get("start_time", 0) >= min_start_time]


def unique_match_ids(matches: Iterable[Dict[str, Any]]) -> List[int]:
    """Return unique match_ids (ints) preserving insertion order."""
    seen = {}
    for m in matches:
        mid = m.get("match_id")
        if isinstance(mid, int) and mid not in seen:
            seen[mid] = None
    return list(seen.keys())


def fetch_match_detail(match_id: int, session: requests.Session, timeout: int = 60) -> Dict[str, Any]:
    """Fetch a single match detail (/matches/{match_id})."""
    data = _request_json(session, f"/matches/{match_id}", timeout=timeout)
    return data if isinstance(data, dict) else {}


def fetch_match_details(
    match_ids: Sequence[int],
    session: requests.Session,
    sleep: float = 1.0,
    timeout: int = 60,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Exception]]]:
    """
    Fetch many match details with simple throttling.
    Returns (results, errors) where errors is list of (match_id, exception).
    """
    results: List[Dict[str, Any]] = []
    errors: List[Tuple[int, Exception]] = []
    for mid in match_ids:
        try:
            results.append(fetch_match_detail(mid, session=session, timeout=timeout))
        except Exception as exc:  # noqa: BLE001
            errors.append((mid, exc))
        if sleep and sleep > 0:
            time.sleep(sleep)
    return results, errors


def wrap_raw_match(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap match payload in the expected raw format."""
    return {"json": match_payload, "pairedItem": {"item": 0}}


def write_json(data: Any, path: Path | str) -> Path:
    """Write JSON with indentation and ensure parent directory exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return p


def _existing_chunk_indices(out_dir: Path, prefix: str) -> List[int]:
    """Return sorted chunk indices based on filenames {prefix}_0001.json."""
    indices: List[int] = []
    for file in out_dir.glob(f"{prefix}_*.json"):
        stem = file.stem
        if stem.startswith(prefix + "_"):
            suffix = stem.split("_")[-1]
            try:
                indices.append(int(suffix))
            except ValueError:
                continue
    return sorted(indices)


def fetch_matches_chunked(
    match_ids: Sequence[int],
    session: requests.Session,
    out_dir: Path | str,
    chunk_size: int = 100,
    resume: bool = True,
    sleep: float = 1.0,
    timeout: int = 60,
    prefix: str = "matches_chunk",
    retry_failed: bool = True,
) -> Dict[str, Any]:
    """
    Fetch matches in chunks and persist each chunk to JSON.

    - Saves `{prefix}_{idx:04d}.json` with wrapped matches per chunk.
    - If resume=True, skips existing chunk files based on prefix/idx.
    - Logs errors to `errors.json`; retries once at the end if retry_failed=True.
    """
    ids = list(dict.fromkeys(match_ids))  # ensure unique and preserve order
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_chunk_indices(out_dir, prefix) if resume else []
    start_chunk = max(existing) + 1 if existing else 0
    skipped_chunks = len(existing)

    errors: List[Tuple[int, Exception]] = []
    chunks_written = 0

    total_chunks = (len(ids) + chunk_size - 1) // chunk_size
    for chunk_idx in range(start_chunk, total_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_ids = ids[start:end]
        if not chunk_ids:
            break

        print(f"[chunk {chunk_idx+1}/{total_chunks}] fetching {len(chunk_ids)} matches (ids {chunk_ids[0]}..{chunk_ids[-1]})")
        chunk_results, chunk_errors = fetch_match_details(
            chunk_ids, session=session, sleep=sleep, timeout=timeout
        )
        errors.extend(chunk_errors)

        chunk_path = out_dir / f"{prefix}_{chunk_idx:04d}.json"
        write_json([wrap_raw_match(m) for m in chunk_results], chunk_path)
        print(f"[chunk {chunk_idx+1}/{total_chunks}] saved {len(chunk_results)} matches -> {chunk_path.name}")
        chunks_written += 1

    retry_errors: List[Tuple[int, Exception]] = []
    retry_success: List[Dict[str, Any]] = []
    if retry_failed and errors:
        retry_ids = [mid for mid, _ in errors]
        print(f"[retry] retrying {len(retry_ids)} failed match_ids")
        retry_success, retry_errors = fetch_match_details(
            retry_ids, session=session, sleep=sleep, timeout=timeout
        )
        retry_path = out_dir / f"{prefix}_retry.json"
        write_json([wrap_raw_match(m) for m in retry_success], retry_path)
        print(f"[retry] saved {len(retry_success)} recovered matches -> {retry_path.name}")

    final_errors = retry_errors if retry_failed else errors
    errors_path = out_dir / "errors.json"
    if final_errors:
        error_payload = [{"match_id": mid, "error": str(exc)} for mid, exc in final_errors]
        write_json(error_payload, errors_path)
        print(f"[errors] {len(final_errors)} remaining errors -> {errors_path.name}")
    else:
        if errors_path.exists():
            errors_path.unlink()

    summary = {
        "total_ids": len(ids),
        "chunk_size": chunk_size,
        "resume": resume,
        "chunks_written": chunks_written,
        "skipped_chunks": skipped_chunks,
        "errors_initial": len(errors),
        "errors_remaining": len(final_errors),
        "retry_saved": len(retry_success),
        "out_dir": str(out_dir),
        "prefix": prefix,
    }
    write_json(summary, out_dir / "summary.json")
    return summary
