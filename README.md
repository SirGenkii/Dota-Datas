# Dota Data v2 — Quickstart

## Prerequisites
- Python 3.12.3 and `venv`
- `make`
- Optional: Git

### Windows (PowerShell)
1) Install Python 3.12.3 (and ensure `python` is on PATH).
2) Allow PowerShell scripts (needed to run `Activate.ps1`): `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
3) Install Make (e.g., via `choco install make`).
4) Create venv & install deps:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5) Copy `.env.example` to `.env` (add `OPENDOTA_KEY`).

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

### 2) Process raw -> parquet
Generate processed tables (matches, players, objectives, teamfights):
```bash
make parquet RAW=data/raw/data_v2.json OUT=data/processed
```
Default RAW/OUT are set in the Makefile (`data/raw/data_v2.json` -> `data/processed`).

### 3) Precompute metrics
Compute Elo, firsts, Roshan/Aegis, gold/xp buckets, series stats:
```bash
make precompute OUT=data/processed METRICS_OUT=data/metrics
```
Generates:
- `data/metrics/elo_timeseries.parquet`, `elo_latest.parquet`
- `firsts.parquet`
- `roshan.parquet`
- `gold_buckets.parquet`, `xp_buckets.parquet`
- `series_maps.parquet`, `series_team_stats.parquet`
- `tracked_teams.parquet`
- `draft_meta.parquet` (first/last pick team per match)
- `adv_snapshots.parquet` (gold/xp advantage snapshot per tracked team & minute)

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
- Precompute relies on both processed parquet and raw JSON for gold/xp advantages, but Streamlit dashboards now load only the parquet outputs (no raw JSON at runtime).
- Series mapping uses `series_type`: 0=BO1, 1=BO3, 2=BO5, 3=BO2.

```bash
 .\.venv\Scripts\Activate.ps1

 streamlit run app/dashboard_streamlit.py
 ```