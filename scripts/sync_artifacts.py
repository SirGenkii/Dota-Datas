from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from dotenv import load_dotenv


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _is_ssh_remote(remote: str) -> bool:
    # Heuristic for rsync/ssh remote syntax: [user@]host:/abs/path
    if ":/" not in remote:
        return False
    host, path = remote.split(":", 1)
    return bool(host.strip()) and path.startswith("/")


def _git_rev(root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
        return out.decode().strip()
    except Exception:
        return None


def _dataset_stats(processed_dir: Path, metrics_dir: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "processed_dir": str(processed_dir),
        "metrics_dir": str(metrics_dir),
    }
    matches_path = processed_dir / "matches.parquet"
    if matches_path.exists():
        lf = pl.scan_parquet(matches_path)
        cols = lf.collect_schema().names()
        if "match_id" in cols:
            stats["matches_unique"] = int(lf.select(pl.col("match_id").n_unique()).collect().item())
        if "start_time" in cols:
            stats["start_time_min"] = lf.select(pl.col("start_time").min()).collect().item()
            stats["start_time_max"] = lf.select(pl.col("start_time").max()).collect().item()
    series_path = processed_dir / "series.parquet"
    if series_path.exists():
        stats["series_rows"] = int(pl.scan_parquet(series_path).select(pl.len()).collect().item())
    extras_path = processed_dir / "extras.parquet"
    if extras_path.exists():
        lf_e = pl.scan_parquet(extras_path)
        cols_e = lf_e.collect_schema().names()
        if "match_id" in cols_e:
            stats["extras_unique"] = int(lf_e.select(pl.col("match_id").n_unique()).collect().item())

    # lightweight inventory
    def size_map(base: Path) -> Dict[str, int]:
        out: Dict[str, int] = {}
        if not base.exists():
            return out
        for p in sorted(base.glob("*.parquet")):
            try:
                out[p.name] = int(p.stat().st_size)
            except Exception:
                continue
        return out

    stats["processed_files_bytes"] = size_map(processed_dir)
    stats["metrics_files_bytes"] = size_map(metrics_dir)
    return stats


def _ensure_rsync_available() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync not found. Install rsync (Linux/macOS) or use WSL/Git Bash on Windows.")


def _ensure_ssh_available() -> None:
    if shutil.which("ssh") is None:
        raise RuntimeError("ssh not found. Install OpenSSH client.")


def _remote_mkdir(remote: str, remote_subdir: str) -> None:
    # remote looks like user@host:/base/path
    if not _is_ssh_remote(remote):
        Path(remote_subdir).mkdir(parents=True, exist_ok=True)
        return
    host, base = remote.split(":", 1)
    base = base.rstrip("/")
    dest = f"{base}/{remote_subdir.lstrip('/')}"
    _run(["ssh", host, "mkdir", "-p", dest])


def _rsync_dir(local_dir: Path, remote: str, remote_subdir: str, *, direction: str, delete: bool) -> None:
    local_dir = Path(local_dir)
    if direction not in {"upload", "download"}:
        raise ValueError("direction must be upload or download")

    if _is_ssh_remote(remote):
        host, base = remote.split(":", 1)
        base = base.rstrip("/")
        remote_path = f"{host}:{base}/{remote_subdir.lstrip('/')}"
    else:
        remote_path = str(Path(remote) / remote_subdir)

    # Trailing slashes matter for rsync semantics.
    src = f"{str(local_dir).rstrip('/')}/"
    dst = f"{remote_path.rstrip('/')}/"
    if direction == "download":
        src, dst = dst, src

    cmd = [
        "rsync",
        "-avh",
        "--partial",
        "--info=progress2",
    ]
    if delete:
        cmd.append("--delete")
    cmd += [src, dst]
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync processed + metrics artifacts to/from a VPS (rsync over SSH).")
    parser.add_argument("direction", choices=["upload", "download"], help="Upload local -> remote, or download remote -> local.")
    parser.add_argument("--remote", default=None, help="Remote base path: user@host:/abs/path (or local path for testing).")
    parser.add_argument("--processed", default="data/processed", help="Local processed dir.")
    parser.add_argument("--metrics", default="data/metrics", help="Local metrics dir.")
    parser.add_argument("--remote-subdir", default="dota-datas", help="Subdirectory under remote base to store artifacts.")
    parser.add_argument("--delete", action="store_true", help="Mirror mode: delete files not present on source.")
    args = parser.parse_args()

    load_dotenv()
    remote = args.remote or os.getenv("DOTA_DATA_REMOTE")
    if not remote:
        raise SystemExit("Missing --remote or env var DOTA_DATA_REMOTE (e.g. user@host:/srv/data).")

    processed_dir = Path(args.processed)
    metrics_dir = Path(args.metrics)
    project_root = Path(__file__).resolve().parent.parent

    _ensure_rsync_available()
    if _is_ssh_remote(remote):
        _ensure_ssh_available()

    manifest_path = metrics_dir / "artifacts_manifest.json"

    # Ensure remote layout.
    base_subdir = args.remote_subdir.rstrip("/")
    _remote_mkdir(remote, f"{base_subdir}/processed")
    _remote_mkdir(remote, f"{base_subdir}/metrics")

    # For upload, write a manifest so it gets synced alongside metrics.
    if args.direction == "upload":
        manifest = _dataset_stats(processed_dir, metrics_dir)
        manifest["git_commit"] = _git_rev(project_root)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Sync
    _rsync_dir(processed_dir, remote, f"{base_subdir}/processed", direction=args.direction, delete=args.delete)
    _rsync_dir(metrics_dir, remote, f"{base_subdir}/metrics", direction=args.direction, delete=args.delete)

    # For download, keep the remote manifest if present; otherwise generate one locally.
    if args.direction == "download" and not manifest_path.exists():
        manifest = _dataset_stats(processed_dir, metrics_dir)
        manifest["git_commit"] = _git_rev(project_root)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        {
            "direction": args.direction,
            "remote": remote,
            "remote_subdir": args.remote_subdir,
            "processed": str(processed_dir),
            "metrics": str(metrics_dir),
            "manifest": str(manifest_path),
        }
    )


if __name__ == "__main__":
    main()
