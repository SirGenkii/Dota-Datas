from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from dotenv import load_dotenv

MANIFEST_NAME = "manifest.json"


def _is_ssh_remote(remote: str) -> bool:
    # Heuristic for rsync/ssh remote syntax: [user@]host:/abs/path
    if ":/" not in remote:
        return False
    host, path = remote.split(":", 1)
    return bool(host.strip()) and path.startswith("/")


def _ensure_rsync_available() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync not found. Install rsync (Linux/macOS) or use WSL/Git Bash on Windows.")


def _ensure_ssh_available() -> None:
    if shutil.which("ssh") is None:
        raise RuntimeError("ssh not found. Install OpenSSH client.")


def _ensure_sshpass_available() -> None:
    if shutil.which("sshpass") is None:
        raise RuntimeError("sshpass not found. Install it (e.g. `sudo apt install sshpass`).")


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


def _resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def _validate_key(key: str) -> str:
    key = key.strip()
    if not key:
        raise ValueError("Empty item key")
    if "/" in key or "\\" in key or ".." in key:
        raise ValueError(f"Invalid item key: {key!r}")
    return key


def _load_items(project_root: Path, *, default_items: Dict[str, str]) -> Dict[str, Path]:
    config_path_raw = (os.getenv("DOTA_DATA_SYNC_CONFIG") or "").strip()
    items: Dict[str, Path] = {}

    if config_path_raw:
        config_path = _resolve_path(project_root, config_path_raw)
        if not config_path.exists():
            raise SystemExit(f"DOTA_DATA_SYNC_CONFIG not found: {config_path}")
        try:
            cfg = json.loads(config_path.read_text())
        except Exception as e:
            raise SystemExit(f"Failed to parse JSON in DOTA_DATA_SYNC_CONFIG={config_path}: {e}") from e

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                key = _validate_key(str(k))
                items[key] = _resolve_path(project_root, str(v))
            return items

        if isinstance(cfg, list):
            for entry in cfg:
                if isinstance(entry, dict):
                    key = _validate_key(str(entry.get("key") or ""))
                    path = str(entry.get("path") or "").strip()
                    if not path:
                        raise SystemExit(f"Invalid item in {config_path} (missing path) for key={key!r}")
                    items[key] = _resolve_path(project_root, path)
                    continue
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    key = _validate_key(str(entry[0]))
                    items[key] = _resolve_path(project_root, str(entry[1]))
                    continue
                raise SystemExit(f"Invalid item in {config_path}: expected {{key,path}} or [key,path], got {type(entry)}")
            return items

        raise SystemExit(f"Invalid JSON in {config_path}: expected object or array.")

    for key, path in default_items.items():
        items[_validate_key(key)] = _resolve_path(project_root, path)
    return items


@dataclass(frozen=True)
class Remote:
    raw: str
    is_ssh: bool
    host: Optional[str]
    base_path_local: Optional[Path]
    base_path_posix: Optional[str]


def _parse_remote(remote: str) -> Remote:
    if _is_ssh_remote(remote):
        host, base = remote.split(":", 1)
        return Remote(raw=remote, is_ssh=True, host=host, base_path_local=None, base_path_posix=base.rstrip("/"))
    base_local = Path(remote).expanduser().resolve()
    return Remote(raw=remote, is_ssh=False, host=None, base_path_local=base_local, base_path_posix=None)


def _remote_root(remote: Remote, remote_subdir: str) -> str:
    remote_subdir = (remote_subdir or "").strip()
    if remote.is_ssh:
        root = PurePosixPath(remote.base_path_posix or "/")
        if remote_subdir and remote_subdir not in {".", "/"}:
            root = root / remote_subdir
        return str(root)
    root_local = remote.base_path_local or Path(".")
    if remote_subdir and remote_subdir not in {".", "/"}:
        root_local = (root_local / remote_subdir).resolve()
    return str(root_local)


def _build_ssh_opts(project_root: Path, control_dir: Path) -> Tuple[List[str], Optional[str], bool]:
    ssh_password = os.getenv("DOTA_DATA_SSH_PASSWORD") or None
    identity = (os.getenv("DOTA_DATA_SSH_IDENTITY") or "").strip()
    use_identity = bool(identity)
    use_sshpass = (ssh_password is not None) and not use_identity

    if use_identity:
        identity_path = _resolve_path(project_root, identity)
        if not identity_path.exists():
            raise RuntimeError(f"SSH identity file not found: {identity_path}")
        ssh_password = None

    opts: List[str] = []
    control_path = str((control_dir / "cm-%C").resolve())
    opts += ["-o", "ControlMaster=auto", "-o", "ControlPersist=60s", "-o", f"ControlPath={control_path}"]

    port = (os.getenv("DOTA_DATA_SSH_PORT") or "").strip()
    if port:
        opts += ["-p", port]

    known_hosts = (os.getenv("DOTA_DATA_SSH_KNOWN_HOSTS") or "").strip()
    if known_hosts:
        known_hosts_path = _resolve_path(project_root, known_hosts)
        known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
        opts += ["-o", f"UserKnownHostsFile={known_hosts_path}"]

    strict = (os.getenv("DOTA_DATA_SSH_STRICT_HOST_KEY_CHECKING") or "").strip()
    if strict:
        opts += ["-o", f"StrictHostKeyChecking={strict}"]

    if use_identity:
        opts += ["-i", str(_resolve_path(project_root, identity))]
        opts += ["-o", "IdentitiesOnly=yes"]
        opts += ["-o", "BatchMode=yes"]

    if use_sshpass:
        _ensure_sshpass_available()

    return opts, ssh_password, use_sshpass


def _rsh_cmd(ssh_opts: List[str], *, use_sshpass: bool) -> List[str]:
    if use_sshpass:
        return ["sshpass", "-e", "ssh", *ssh_opts]
    return ["ssh", *ssh_opts]


def _run_ssh(host: str, args: List[str], *, rsh: List[str], ssh_password: Optional[str]) -> None:
    env = None
    if ssh_password is not None:
        env = os.environ.copy()
        env["SSHPASS"] = ssh_password
    subprocess.run([*rsh, host, *args], check=True, env=env)


def _ssh_capture(host: str, args: List[str], *, rsh: List[str], ssh_password: Optional[str]) -> str:
    env = None
    if ssh_password is not None:
        env = os.environ.copy()
        env["SSHPASS"] = ssh_password
    out = subprocess.check_output([*rsh, host, *args], env=env)
    return out.decode("utf-8", errors="replace")


def _remote_mkdir(remote: Remote, *, rsh: List[str], ssh_password: Optional[str], path: str) -> None:
    if remote.is_ssh:
        assert remote.host is not None
        _run_ssh(remote.host, ["mkdir", "-p", path], rsh=rsh, ssh_password=ssh_password)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def _remote_list_dirs(remote: Remote, *, rsh: List[str], ssh_password: Optional[str], path: str) -> List[str]:
    if remote.is_ssh:
        assert remote.host is not None
        try:
            out = _ssh_capture(remote.host, ["ls", "-1", path], rsh=rsh, ssh_password=ssh_password)
        except subprocess.CalledProcessError:
            return []
        return [r.strip() for r in out.splitlines() if r.strip()]
    p = Path(path)
    if not p.exists():
        return []
    return sorted([c.name for c in p.iterdir() if c.is_dir()])


def _remote_read_json(remote: Remote, *, rsh: List[str], ssh_password: Optional[str], path: str) -> Optional[Dict[str, Any]]:
    try:
        if remote.is_ssh:
            assert remote.host is not None
            out = _ssh_capture(remote.host, ["cat", path], rsh=rsh, ssh_password=ssh_password)
            return json.loads(out)
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception:
        return None


def _rsync(
    *,
    direction: str,
    local_path: Path,
    remote: Remote,
    remote_abs_path: str,
    delete: bool,
    ssh_opts: List[str],
    use_sshpass: bool,
    ssh_password: Optional[str],
) -> None:
    if direction not in {"upload", "download"}:
        raise ValueError("direction must be upload or download")

    cmd = ["rsync", "-avh", "--partial", "--info=progress2"]
    env = None
    if remote.is_ssh:
        cmd += ["-e", shlex.join(_rsh_cmd(ssh_opts, use_sshpass=use_sshpass))]
        if ssh_password is not None:
            env = os.environ.copy()
            env["SSHPASS"] = ssh_password
    if delete:
        cmd.append("--delete")

    if local_path.is_dir():
        local_dir = f"{str(local_path).rstrip('/')}/"
        local_file = local_dir
    else:
        local_dir = str(local_path)
        local_file = str(local_path)

    if remote.is_ssh:
        remote_endpoint = f"{remote.host}:{remote_abs_path}"
    else:
        remote_endpoint = remote_abs_path

    if local_path.is_dir():
        remote_endpoint = f"{remote_endpoint.rstrip('/')}/"

    if direction == "upload":
        src, dst = local_dir if local_path.is_dir() else local_file, remote_endpoint
    else:
        src, dst = remote_endpoint, local_dir if local_path.is_dir() else local_file

    cmd += [src, dst]
    subprocess.run(cmd, check=True, env=env)


def _snapshot_id_now() -> str:
    return f"snapshot_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _pick_latest_snapshot(snapshot_names: List[str]) -> Optional[str]:
    candidates = [s for s in snapshot_names if s.startswith("snapshot_")]
    return max(candidates) if candidates else None


def _confirm(prompt: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise SystemExit("Refusing to prompt on non-interactive stdin. Re-run with --yes.")
    ans = input(f"{prompt} [y/N] ").strip().lower()
    return ans in {"y", "yes"}


def _safe_replace(target: Path, staging: Path) -> None:
    target = Path(target)
    staging = Path(staging)
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot-based sync to/from a VPS (rsync over SSH).")
    parser.add_argument("direction", choices=["upload", "download"], help="Upload local -> remote, or download latest snapshot -> local.")
    parser.add_argument("--remote", default=None, help="Remote base path: user@host:/abs/path (or local path for testing).")
    parser.add_argument("--remote-subdir", default="dota-datas", help="Subdirectory under remote base (project namespace).")
    parser.add_argument(
        "--snapshots-dir",
        default=".",
        help="Where to store snapshots under the remote subdir (default: '.' -> snapshots directly under <remote>/<subdir>/).",
    )
    parser.add_argument("--snapshot", default=None, help="Download: snapshot name to fetch (default: latest).")
    parser.add_argument("--yes", action="store_true", help="Download: skip confirmation prompt.")

    parser.add_argument("--processed", default="data/processed", help="Default local processed dir (used unless DOTA_DATA_SYNC_CONFIG is set).")
    parser.add_argument("--metrics", default="data/metrics", help="Default local metrics dir (used unless DOTA_DATA_SYNC_CONFIG is set).")
    parser.add_argument("--teams-csv", default="data/teams_to_look.csv", help="Default teams csv (used unless DOTA_DATA_SYNC_CONFIG is set).")
    parser.add_argument("--aliases-csv", default="data/team_aliases.csv", help="Default aliases csv (used unless DOTA_DATA_SYNC_CONFIG is set).")

    parser.add_argument("--delete", action="store_true", help="Use rsync --delete for directory items.")
    parser.add_argument("--allow-missing", action="store_true", help="Upload: skip items missing locally.")
    args = parser.parse_args()

    load_dotenv()
    remote_raw = args.remote or os.getenv("DOTA_DATA_REMOTE")
    if not remote_raw:
        raise SystemExit("Missing --remote or env var DOTA_DATA_REMOTE (e.g. user@host:/home/user/dota-data).")

    project_root = Path(__file__).resolve().parent.parent
    remote = _parse_remote(remote_raw)

    default_items = {
        "processed": args.processed,
        "metrics": args.metrics,
        "teams_to_look.csv": args.teams_csv,
        "team_aliases.csv": args.aliases_csv,
    }
    items = _load_items(project_root, default_items=default_items)

    _ensure_rsync_available()
    if remote.is_ssh:
        _ensure_ssh_available()

    with tempfile.TemporaryDirectory(prefix="dota-datas-ssh-") as td:
        ssh_opts, ssh_password, use_sshpass = _build_ssh_opts(project_root, Path(td))
        rsh = _rsh_cmd(ssh_opts, use_sshpass=use_sshpass)

        remote_root_abs = _remote_root(remote, args.remote_subdir)
        snapshots_abs = str(PurePosixPath(remote_root_abs) / args.snapshots_dir) if remote.is_ssh else str(Path(remote_root_abs) / args.snapshots_dir)

        if args.direction == "upload":
            snapshot_id = _snapshot_id_now()
            snapshot_abs = str(PurePosixPath(snapshots_abs) / snapshot_id) if remote.is_ssh else str(Path(snapshots_abs) / snapshot_id)
            _remote_mkdir(remote, rsh=rsh, ssh_password=ssh_password, path=snapshots_abs)
            _remote_mkdir(remote, rsh=rsh, ssh_password=ssh_password, path=snapshot_abs)

            manifest: Dict[str, Any] = {
                "snapshot_id": snapshot_id,
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "git_commit": _git_rev(project_root),
                "items": [],
            }

            processed_dir = items.get("processed")
            metrics_dir = items.get("metrics")
            if processed_dir and metrics_dir and processed_dir.exists() and metrics_dir.exists():
                if processed_dir.is_dir() and metrics_dir.is_dir():
                    manifest["stats"] = _dataset_stats(processed_dir, metrics_dir)

            for key, local_path in items.items():
                if not local_path.exists():
                    if args.allow_missing:
                        continue
                    raise SystemExit(f"Missing local path for item {key!r}: {local_path}")

                item_type = "dir" if local_path.is_dir() else "file"
                remote_item_abs = str(PurePosixPath(snapshot_abs) / key) if remote.is_ssh else str(Path(snapshot_abs) / key)
                if item_type == "dir":
                    _remote_mkdir(remote, rsh=rsh, ssh_password=ssh_password, path=remote_item_abs)

                _rsync(
                    direction="upload",
                    local_path=local_path,
                    remote=remote,
                    remote_abs_path=remote_item_abs,
                    delete=args.delete if item_type == "dir" else False,
                    ssh_opts=ssh_opts,
                    use_sshpass=use_sshpass,
                    ssh_password=ssh_password,
                )
                manifest["items"].append({"key": key, "type": item_type, "local_path": str(local_path.relative_to(project_root))})

            tmp_manifest = Path(td) / MANIFEST_NAME
            tmp_manifest.write_text(json.dumps(manifest, indent=2))
            remote_manifest_abs = str(PurePosixPath(snapshot_abs) / MANIFEST_NAME) if remote.is_ssh else str(Path(snapshot_abs) / MANIFEST_NAME)
            _rsync(
                direction="upload",
                local_path=tmp_manifest,
                remote=remote,
                remote_abs_path=remote_manifest_abs,
                delete=False,
                ssh_opts=ssh_opts,
                use_sshpass=use_sshpass,
                ssh_password=ssh_password,
            )

            if remote.is_ssh and remote.host is not None:
                latest_link = str(PurePosixPath(snapshots_abs) / "latest")
                _run_ssh(remote.host, ["ln", "-sfn", snapshot_id, latest_link], rsh=rsh, ssh_password=ssh_password)

            print({"direction": "upload", "snapshot": snapshot_id, "remote_root": remote_root_abs, "snapshots": snapshots_abs})
            return

        # download
        _remote_mkdir(remote, rsh=rsh, ssh_password=ssh_password, path=snapshots_abs)
        available = _remote_list_dirs(remote, rsh=rsh, ssh_password=ssh_password, path=snapshots_abs)
        snapshot_id = args.snapshot or _pick_latest_snapshot(available)
        if not snapshot_id:
            raise SystemExit(f"No snapshots found under {snapshots_abs}")

        snapshot_abs = str(PurePosixPath(snapshots_abs) / snapshot_id) if remote.is_ssh else str(Path(snapshots_abs) / snapshot_id)
        remote_manifest_abs = str(PurePosixPath(snapshot_abs) / MANIFEST_NAME) if remote.is_ssh else str(Path(snapshot_abs) / MANIFEST_NAME)
        manifest = _remote_read_json(remote, rsh=rsh, ssh_password=ssh_password, path=remote_manifest_abs)

        if manifest and isinstance(manifest, dict):
            created = manifest.get("generated_at") or "unknown"
            commit = manifest.get("git_commit") or "unknown"
            print(f"Latest snapshot: {snapshot_id} | generated_at={created} | git_commit={commit}")
        else:
            print(f"Latest snapshot: {snapshot_id}")

        if not _confirm("Download this snapshot and overwrite local data?", assume_yes=args.yes):
            print("Canceled.")
            return

        staging_root = (project_root / ".sync_tmp" / snapshot_id).resolve()
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

        try:
            download_items: List[Tuple[str, Path, str]] = []
            if manifest and isinstance(manifest.get("items"), list):
                for it in manifest["items"]:
                    if not isinstance(it, dict):
                        continue
                    key = _validate_key(str(it.get("key") or ""))
                    local_path_rel = str(it.get("local_path") or "")
                    item_type = str(it.get("type") or "dir")
                    local_target = _resolve_path(project_root, local_path_rel)
                    if project_root not in local_target.parents and local_target != project_root:
                        raise SystemExit(f"Refusing to overwrite path outside project: {local_target}")
                    download_items.append((key, local_target, item_type))
            else:
                for key, local_target in items.items():
                    item_type = "dir" if local_target.suffix == "" else "file"
                    download_items.append((key, local_target, item_type))

            for key, local_target, item_type in download_items:
                remote_item_abs = str(PurePosixPath(snapshot_abs) / key) if remote.is_ssh else str(Path(snapshot_abs) / key)
                staging_path = (staging_root / key).resolve()
                if item_type == "dir":
                    staging_path.mkdir(parents=True, exist_ok=True)
                    _rsync(
                        direction="download",
                        local_path=staging_path,
                        remote=remote,
                        remote_abs_path=remote_item_abs,
                        delete=args.delete,
                        ssh_opts=ssh_opts,
                        use_sshpass=use_sshpass,
                        ssh_password=ssh_password,
                    )
                    _safe_replace(local_target, staging_path)
                else:
                    staging_path.parent.mkdir(parents=True, exist_ok=True)
                    _rsync(
                        direction="download",
                        local_path=staging_path,
                        remote=remote,
                        remote_abs_path=remote_item_abs,
                        delete=False,
                        ssh_opts=ssh_opts,
                        use_sshpass=use_sshpass,
                        ssh_password=ssh_password,
                    )
                    _safe_replace(local_target, staging_path)

        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)

        print({"direction": "download", "snapshot": snapshot_id, "remote_root": remote_root_abs, "snapshots": snapshots_abs})


if __name__ == "__main__":
    main()
