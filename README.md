# Dota Data v2 — Quickstart

## Prerequisites
- Python 3.12.3 and `venv`
- `make` (Windows: see section below)
- Optional: Git

### Windows (PowerShell)
1) Install Python 3.12.3 (and ensure `python` is on PATH).
2) Allow PowerShell scripts (needed to run `Activate.ps1`): `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
3) Install `make`:
   - Option A (recommended): Chocolatey
     - Install Chocolatey: https://chocolatey.org/install
     - Then: `choco install make`
   - Option B: winget (if available)
     - `winget install -e --id GnuWin32.Make`
     - Ensure `make.exe` is available in your `PATH`
   - Option C: Git for Windows + Make (if you already use it)
     - Install Git for Windows, then use “Git Bash” and a `make` provided/added to your PATH.
4) Create venv & install deps:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Optional (if you want to use the Makefile for venv creation): the Makefile supports Windows venv paths (`.venv\Scripts\python.exe`), so you can also run:
   ```powershell
   make install
   # If your Python is only available via the launcher:
   make venv SYSTEM_PYTHON="py -3.12"
   ```
5) Copy `.env.example` to `.env` (add `OPENDOTA_KEY`).

### Windows — Run the dashboard
1) Activate the venv:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2) Run the dashboard via Make:
   ```powershell
   make dashboard
   ```
   (equivalent without make: `python -m streamlit run app/dashboard_streamlit.py`)

### Linux/macOS
1) Install Python 3.12.3 and make (e.g., `sudo apt install python3 python3-venv make` then ensure version 3.12.3).
2) Create venv & install deps:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3) Copy `.env.example` to `.env` (add `OPENDOTA_KEY`).

## Data pipeline
### 1) Scraping (notebooks)
- Notebooks in `notebooks/scrap/`:
  - `01_team_matches.ipynb`: fetch `/teams/{id}/matches` into `data/interim/team_matches_raw.json`.
  - `02_filter_and_sample.ipynb`: filter >= 2025-01-01, inspect sample match detail.
  - `03_full_scrap.ipynb`: full fetch of filtered match_ids -> chunks in `data/raw/chunks_v2/` + combined `data/raw/data_v2.json`.
- Requires `OPENDOTA_KEY` in `.env`.

### 1b) Incremental sync (new matches)
Downloads only missing recent matches (for the teams in `data/teams_to_look.csv`), stores raw responses as JSON chunks under `data/raw/updates/`, then appends into `data/processed/`:
```bash
make update OUT=data/processed
```
Useful options:
- `SINCE=YYYY-MM-DD` to set a lower bound (e.g. `make update SINCE=2025-01-01`)
- `MAX_NEW=200` to cap downloaded matches (newest first)
- `DRY_RUN=1` to discover/write the run without downloading match details
- `LIMIT`, `MAX_PAGES`, `CHUNK_SIZE`, `SLEEP_MATCH_DETAIL` (see `Makefile`)

Sanity check on the latest run (or a specific run):
```bash
make healthcheck OUT=data/processed
# or: make healthcheck RUN_DIR=data/raw/updates/run_YYYYMMDD_HHMMSS OUT=data/processed
```
Outputs (inside the run directory):
- `healthcheck.json`
- `matches_trace.parquet` (match_id, teams, date, etc.)
- `series_trace.parquet` (teams, final series score if `data/processed/series.parquet` is available)

### 2) Process raw -> parquet
Generate processed tables (matches, players, objectives, teamfights):
```bash
make parquet RAW=data/raw/data_v2.json OUT=data/processed
```
Default RAW/OUT are set in the Makefile (`data/raw/data_v2.json` -> `data/processed`).

If you want to avoid rewriting the big parquets and only create the “raw fields needed for metrics”:
```bash
make extras OUT=data/processed OVERWRITE=1
# or (multiple sources, e.g. historic raw + update chunks):
make extras EXTRAS_RAW_SOURCES="backup/data/raw/data_v2.json data/raw/updates" OUT=data/processed OVERWRITE=1
```
This generates `data/processed/extras.parquet` (adv arrays + picks/bans) so `precompute` no longer needs raw JSON.

### 3) Precompute metrics
Compute Elo, firsts, Roshan/Aegis, gold/xp buckets, series stats:
```bash
make precompute OUT=data/processed METRICS_OUT=data/metrics
```
If `precompute` fails with an “extras coverage” error, regenerate `data/processed/extras.parquet` with `make extras` (or pass `--allow-partial-extras` to `scripts/precompute_metrics.py` to force it).

Generates:
- `data/metrics/elo_timeseries.parquet`, `elo_latest.parquet`, `elo_latest_all.parquet`
- `data/metrics/glicko2_latest.parquet`, `glicko2_latest_all.parquet`, `glicko2_timeseries.parquet` (Glicko-2 is computed on *series outcomes*, keyed by `(leagueid, series_id)`)
- `data/metrics/series_results.parquet` (one row per series; used as the rating input)
- `firsts.parquet`
- `roshan.parquet`
- `gold_buckets.parquet`, `xp_buckets.parquet`
- `series_maps.parquet`, `series_team_stats.parquet`
- `tracked_teams.parquet`
- `draft_meta.parquet` (first/last pick team per match)
- `adv_snapshots.parquet` (gold/xp advantage snapshot per tracked team & minute)

Optional: grid-search Glicko-2 hyperparameters (writes `data/metrics/glicko2_calibration.json`):
```bash
make calibrate-glicko OUT=data/processed METRICS_OUT=data/metrics
```

To pin Glicko-2 parameters for `precompute`, copy `glicko2_config.example.json` to `glicko2_config.json` and run:
```bash
make precompute OUT=data/processed METRICS_OUT=data/metrics GLICKO_CONFIG=glicko2_config.json
```

## Sharing data artifacts (VPS sync)
If you don’t want to use Git LFS for large parquet files, you can share data via `rsync` over SSH to a VPS. The sync is snapshot-based: each upload creates a new dated snapshot directory on the server, and download pulls the latest snapshot (with a confirmation prompt before overwriting local files).

### Prerequisites
- SSH access to your VPS (recommended: SSH key auth)
- `rsync` + `ssh` available locally
  - Linux/macOS: typically available (or `sudo apt install rsync openssh-client`)
  - Windows: easiest is WSL (Ubuntu) and run the Make commands from WSL

### Setup
1) On the VPS, create a directory to store artifacts:
   ```bash
   sudo mkdir -p /srv/dota-datas
   sudo chown -R $USER:$USER /srv/dota-datas
   ```
2) Add `DOTA_DATA_REMOTE` to your `.env` (or export it in your shell):
   ```bash
   DOTA_DATA_REMOTE=user@your-vps:/srv/dota-datas
   ```
   If your VPS layout is like `/home/dota_uploaduser/dota-data`, you can either:
   - set `DOTA_DATA_REMOTE=dota_uploaduser@your-vps:/home/dota_uploaduser` and run `make ... SYNC_SUBDIR=dota-data`
   - or set `DOTA_DATA_REMOTE=dota_uploaduser@your-vps:/home/dota_uploaduser/dota-data` and run `make ... SYNC_SUBDIR=.`

### Authentication (no prompts)
Recommended: SSH keys (no password prompts, easiest for collaboration).
1) On each machine (you and your coworker), generate a key:
   ```bash
   ssh-keygen -t ed25519 -C "dota-datas"
   ```
2) Add the public key to the VPS user (run once per person):
   ```bash
   ssh-copy-id user@your-vps
   # or manually append ~/.ssh/id_ed25519.pub into ~/.ssh/authorized_keys on the server
   ```
Notes:
- Don’t share private keys between people; add multiple public keys on the server instead.
- First connection may ask to trust the host key: run `ssh user@your-vps` once to accept it.

#### Project-local key file (optional)
If you want the sync to always use a specific key stored next to the project (but still not committed), put it under `secrets/` and point the script to it via `.env`:
```bash
DOTA_DATA_SSH_IDENTITY=secrets/id_ed25519
DOTA_DATA_SSH_KNOWN_HOSTS=secrets/known_hosts
DOTA_DATA_SSH_STRICT_HOST_KEY_CHECKING=accept-new
```
This makes the sync fully non-interactive (no password prompt, no host-key prompt).

Windows note: easiest is to run the Make commands from WSL; then paths like `secrets/id_ed25519` work normally. If you run from native Windows OpenSSH, you can also use an absolute Windows path, but `rsync` is usually missing unless you use WSL/Git Bash.

Alternative (not recommended): store a password and use `sshpass` (fully non-interactive).
1) Install `sshpass` locally: `sudo apt install sshpass` (WSL/Linux).
2) Add to `.env`:
   ```bash
   DOTA_DATA_SSH_PASSWORD=your_password_here
   ```

### Upload / download
- Upload local artifacts to the VPS:
  ```bash
  make sync-upload OUT=data/processed METRICS_OUT=data/metrics
  ```
- Download the latest snapshot from the VPS (asks confirmation before overwrite):
  ```bash
  make sync-download OUT=data/processed METRICS_OUT=data/metrics
  ```
- Chain precompute + upload:
  ```bash
  make precompute-upload OUT=data/processed METRICS_OUT=data/metrics
  ```

Notes:
- If you use password-based SSH without `sshpass`, you’ll be prompted in the terminal (the script reuses connections so it’s usually once per run).
- Remote layout becomes: `<DOTA_DATA_REMOTE>/<SYNC_SUBDIR>/snapshot_YYYYMMDD_HHMMSS/...` (default `SYNC_SUBDIR=.` so snapshots land directly under `DOTA_DATA_REMOTE`)
- `make sync-download` prompts by default; use `SYNC_YES=1` to skip the prompt.
- Use `SYNC_DELETE=1` to enable `rsync --delete` for directory items.
- Items included in snapshots are configurable via a JSON config file:
  - Create `secrets/sync_items.json` (you can start from `sync_items.example.json`)
  - Set `DOTA_DATA_SYNC_CONFIG=secrets/sync_items.json`
- If `players.parquet` is too large, consider uploading a pruned dataset (e.g. `OUT=data/processed_pruned`) and keeping the full dataset only on the VPS.

### 4) Analysis notebooks
- `01_data_overview.ipynb`, `02_match_flow.ipynb`, `03_hero_lane.ipynb`, `05_dictionaries.ipynb` use processed parquet.
- `07_metrics_overview.ipynb` inspects precomputed metrics (Elo, firsts, Roshan, buckets, series).

### 5) Streamlit dashboards
- Legacy: `app/streamlit_app.py` (full exploration).
- New dashboard: `app/dashboard_streamlit.py` (side-by-side Team A/B with precomputed metrics).

Run:
```bash
streamlit run app/dashboard_streamlit.py
```

## Project structure (key paths)
- `data/raw/`: raw JSON (`data_v2.json`, chunks).
- `data/processed/`: parquet tables (matches, players, objectives, teamfights).
- `data/metrics/`: precomputed metrics parquet.
- `data/interim/`: intermediate scraping outputs.
- `data/teams_to_look.csv`: tracked teams (24 teams).
- `scripts/precompute_metrics.py`: compute metrics from processed + raw.
- `app/`: Streamlit dashboards.
- `notebooks/`: EDA and scraping notebooks.

## Environment / commands
- Install deps: `make install` (creates venv, installs requirements).
- Generate parquet: `make parquet`
- Precompute metrics: `make precompute`
- Run Streamlit: `streamlit run app/dashboard_streamlit.py`

## Notes
- Ensure `OPENDOTA_KEY` is set in `.env` for scraping.
- Precompute uses `data/processed/extras.parquet` when present (otherwise it falls back to `--raw`).
- Series mapping uses `series_type`: 0=BO1, 1=BO3, 2=BO5, 3=BO2.

Example:
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Streamlit: `streamlit run app/dashboard_streamlit.py`
