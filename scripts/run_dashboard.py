from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _ensure_tmpdir() -> str | None:
    """
    Streamlit (or its deps) may create AF_UNIX sockets on some platforms.
    Those socket paths have a small maximum length; long temp paths can crash with:
      "unix path too long for unix domain socket"

    We allow forcing a short temp dir via DOTA_DATA_TMPDIR, otherwise we try to pick
    a short directory automatically.
    """
    forced = os.getenv("DOTA_DATA_TMPDIR")
    if forced:
        tmp = Path(forced).expanduser()
        tmp.mkdir(parents=True, exist_ok=True)
        return str(tmp)

    current = Path(tempfile.gettempdir())

    # Heuristic: if the temp dir path is long, prefer a short project-local temp.
    # On Windows, the default is often "...\\AppData\\Local\\Temp" which can be long.
    if len(str(current)) <= 40:
        return None

    root = Path(__file__).resolve().parents[1]
    candidate = root / ".tmp"
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def main() -> int:
    tmp = _ensure_tmpdir()
    if tmp:
        # Set all common temp vars before importing/starting Streamlit.
        os.environ["TMPDIR"] = tmp
        os.environ["TMP"] = tmp
        os.environ["TEMP"] = tmp

    cmd = [sys.executable, "-m", "streamlit", "run", "app/dashboard_streamlit.py"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

