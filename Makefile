VENV := .venv

# Cross-platform venv paths:
# - Linux/macOS: .venv/bin/python
# - Windows:     .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	PYTHON := $(VENV_BIN)/python.exe
	ACTIVATE_HINT := $(VENV_BIN)/Activate.ps1
	SYSTEM_PYTHON ?= python
else
	VENV_BIN := $(VENV)/bin
	PYTHON := $(VENV_BIN)/python
	ACTIVATE_HINT := source $(VENV_BIN)/activate
	SYSTEM_PYTHON ?= python3
endif

# Make sure modules under ./src are importable when running via `python -m ...`.
export PYTHONPATH := .

.PHONY: venv install activate notebook deps-clean parquet extras update healthcheck
.PHONY: sync-upload sync-download precompute-upload
.PHONY: calibrate-glicko
.PHONY: prune

venv:
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip

install: venv
	@$(PYTHON) -m pip install -q -r requirements.txt && echo "Dependencies up to date."

activate:
	@echo "To activate the venv: $(ACTIVATE_HINT)"

notebook: install 
	$(PYTHON) -m ipykernel install --user --name dota-data --display-name "Python (dota-data)"
	$(PYTHON) -m jupyter lab

deps-clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; root=Path('$(VENV)');\nfor d in root.rglob('__pycache__'):\n    shutil.rmtree(d, ignore_errors=True)\nprint('Removed __pycache__ under', root)"

# Génère les parquets à partir du brut (par défaut data/raw/data_v2.json -> data/processed)
# Usage: make parquet [RAW=path/to/raw.json] [OUT=path/to/processed_dir]
RAW ?= data/raw/data_v2.json
OUT ?= data/processed
ALIASES ?= data/team_aliases.csv
parquet: install
	$(PYTHON) -m src.dota_data.io --raw $(RAW) --out $(OUT) --aliases $(ALIASES)

# Génère uniquement `extras.parquet` (adv arrays + picks/bans) depuis le raw, sans réécrire players/objectives/etc.
# Usage:
#   make extras OUT=data/processed OVERWRITE=1
#   make extras EXTRAS_RAW_SOURCES="backup/data/raw/data_v2.json data/raw/updates" OUT=data/processed OVERWRITE=1
OVERWRITE ?= 0
EXTRAS_RAW_SOURCES ?= data/raw/data_v2.json data/raw/updates
extras: install
	$(PYTHON) -m src.dota_data.extras $(foreach s,$(EXTRAS_RAW_SOURCES),--raw $(s)) --out $(OUT) $(if $(filter 1 true yes,$(OVERWRITE)),--overwrite,)

# Sync incrémentale: télécharge les matchs récents manquants et append dans data/processed
# Usage: make update [OUT=data/processed] [RAW_UPDATES=data/raw/updates] [LIMIT=100] [MAX_PAGES=3] [SINCE=YYYY-MM-DD]
RAW_UPDATES ?= data/raw/updates
LIMIT ?= 100
MAX_PAGES ?= 3
SINCE ?=
MAX_NEW ?=
DRY_RUN ?= 0
CHUNK_SIZE ?= 100
SLEEP_MATCH_DETAIL ?= 1.0
TIMEOUT ?= 90
TIMEOUT_TEAM_MATCHES ?= 60
update: install
	$(PYTHON) -m src.dota_data.update --teams data/teams_to_look.csv --processed $(OUT) --raw-updates $(RAW_UPDATES) --aliases $(ALIASES) --limit $(LIMIT) --max-pages $(MAX_PAGES) $(if $(SINCE),--since $(SINCE),) $(if $(MAX_NEW),--max-new $(MAX_NEW),) $(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,) --timeout $(TIMEOUT) --timeout-team-matches $(TIMEOUT_TEAM_MATCHES) --chunk-size $(CHUNK_SIZE) --sleep-match-detail $(SLEEP_MATCH_DETAIL) --apply-parquet

# Sanity checks pour le dernier run d'updates (ou un run précis)
# Usage: make healthcheck [RUN_DIR=data/raw/updates/run_...] [OUT=data/processed]
RUN_DIR ?=
healthcheck: install
	$(PYTHON) -m src.dota_data.healthcheck $(if $(RUN_DIR),--run-dir $(RUN_DIR),) --processed $(OUT)

# Prune processed parquet to remove old matches (writes to a new dir by default).
# Usage: make prune [OUT=data/processed] [PRUNE_OUT=data/processed_pruned] (MIN_DATE=YYYY-MM-DD | KEEP_DAYS=730)
PRUNE_OUT ?= data/processed_pruned
MIN_DATE ?=
KEEP_DAYS ?=
prune: install
	$(PYTHON) -m src.dota_data.prune --processed $(OUT) --out $(PRUNE_OUT) $(if $(MIN_DATE),--min-date $(MIN_DATE),) $(if $(KEEP_DAYS),--keep-days $(KEEP_DAYS),)

# Pré-calcul des métriques (Elo, firsts) à partir des parquets
METRICS_OUT ?= data/metrics
GLICKO_CONFIG ?=
precompute: install
	$(PYTHON) scripts/precompute_metrics.py --processed $(OUT) --teams data/teams_to_look.csv --out $(METRICS_OUT) $(if $(GLICKO_CONFIG),--glicko-config $(GLICKO_CONFIG),)

# Grid-search Glicko-2 hyperparameters (series-based) and write best config JSON.
# Usage: make calibrate-glicko OUT=data/processed METRICS_OUT=data/metrics
CAL_G2_SCOPE ?= tracked_v_any
CAL_G2_WARMUP ?= 0.2
CAL_G2_PERIODS ?= day week
CAL_G2_TAUS ?= 0.3 0.5 0.8
CAL_G2_INIT_RDS ?= 300 350
CAL_G2_INIT_SIGMAS ?= 0.06
CAL_G2_SCORE_RD_MULTS ?= 2.0
calibrate-glicko: install
	$(PYTHON) scripts/calibrate_glicko2.py --processed $(OUT) --teams data/teams_to_look.csv --out $(METRICS_OUT)/glicko2_calibration.json --scope $(CAL_G2_SCOPE) --warmup-frac $(CAL_G2_WARMUP) --periods $(CAL_G2_PERIODS) --taus $(CAL_G2_TAUS) --init-rds $(CAL_G2_INIT_RDS) --init-sigmas $(CAL_G2_INIT_SIGMAS) --score-rd-mults $(CAL_G2_SCORE_RD_MULTS)

# Sync processed + metrics to/from a VPS (rsync over SSH).
# Requires: rsync + ssh, and DOTA_DATA_REMOTE=user@host:/abs/path (in env or .env).
# Usage:
#   make sync-upload OUT=data/processed METRICS_OUT=data/metrics SYNC_SUBDIR=dota-datas
#   make sync-download OUT=data/processed METRICS_OUT=data/metrics SYNC_SUBDIR=dota-datas
SYNC_REMOTE ?=
SYNC_SUBDIR ?= .
SYNC_DELETE ?= 0
SYNC_YES ?= 0
sync-upload: install
	$(PYTHON) scripts/sync_artifacts.py upload $(if $(SYNC_REMOTE),--remote $(SYNC_REMOTE),) --processed $(OUT) --metrics $(METRICS_OUT) --remote-subdir $(SYNC_SUBDIR) $(if $(filter 1 true yes,$(SYNC_DELETE)),--delete,)

sync-download: install
	$(PYTHON) scripts/sync_artifacts.py download $(if $(SYNC_REMOTE),--remote $(SYNC_REMOTE),) --processed $(OUT) --metrics $(METRICS_OUT) --remote-subdir $(SYNC_SUBDIR) $(if $(filter 1 true yes,$(SYNC_DELETE)),--delete,) $(if $(filter 1 true yes,$(SYNC_YES)),--yes,)

precompute-upload: precompute sync-upload

.PHONY: dashboard
dashboard: install
	$(PYTHON) scripts/run_dashboard.py
