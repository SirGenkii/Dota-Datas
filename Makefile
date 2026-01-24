VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install activate notebook deps-clean parquet extras update healthcheck
.PHONY: sync-upload sync-download precompute-upload
.PHONY: prune

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip >/dev/null

install: venv
	@$(PIP) install -q -r requirements.txt && echo "Dependencies up to date."

activate:
	@echo "Pour activer le venv: source $(VENV)/bin/activate"

notebook: install 
	$(PYTHON) -m ipykernel install --user --name dota-data --display-name "Python (dota-data)"
	PYTHONPATH=. $(PYTHON) -m jupyter lab

deps-clean:
	find $(VENV)/lib -name '__pycache__' -type d -prune -exec rm -rf {} +

# Génère les parquets à partir du brut (par défaut data/raw/data_v2.json -> data/processed)
# Usage: make parquet [RAW=path/to/raw.json] [OUT=path/to/processed_dir]
RAW ?= data/raw/data_v2.json
OUT ?= data/processed
ALIASES ?= data/team_aliases.csv
parquet: install
	$(PYTHON) -m src.dota_data.io --raw $(RAW) --out $(OUT) --aliases $(ALIASES)

# Génère uniquement `extras.parquet` (adv arrays + picks/bans) depuis le raw, sans réécrire players/objectives/etc.
# Usage: make extras [RAW=data/raw/data_v2.json] [OUT=data/processed] [OVERWRITE=1]
OVERWRITE ?= 0
extras: install
	PYTHONPATH=. $(PYTHON) -m src.dota_data.extras --raw $(RAW) --out $(OUT) $(if $(filter 1 true yes,$(OVERWRITE)),--overwrite,)

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
	PYTHONPATH=. $(PYTHON) -m src.dota_data.update --teams data/teams_to_look.csv --processed $(OUT) --raw-updates $(RAW_UPDATES) --aliases $(ALIASES) --limit $(LIMIT) --max-pages $(MAX_PAGES) $(if $(SINCE),--since $(SINCE),) $(if $(MAX_NEW),--max-new $(MAX_NEW),) $(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,) --timeout $(TIMEOUT) --timeout-team-matches $(TIMEOUT_TEAM_MATCHES) --chunk-size $(CHUNK_SIZE) --sleep-match-detail $(SLEEP_MATCH_DETAIL) --apply-parquet

# Sanity checks pour le dernier run d'updates (ou un run précis)
# Usage: make healthcheck [RUN_DIR=data/raw/updates/run_...] [OUT=data/processed]
RUN_DIR ?=
healthcheck: install
	PYTHONPATH=. $(PYTHON) -m src.dota_data.healthcheck $(if $(RUN_DIR),--run-dir $(RUN_DIR),) --processed $(OUT)

# Prune processed parquet to remove old matches (writes to a new dir by default).
# Usage: make prune [OUT=data/processed] [PRUNE_OUT=data/processed_pruned] (MIN_DATE=YYYY-MM-DD | KEEP_DAYS=730)
PRUNE_OUT ?= data/processed_pruned
MIN_DATE ?=
KEEP_DAYS ?=
prune: install
	PYTHONPATH=. $(PYTHON) -m src.dota_data.prune --processed $(OUT) --out $(PRUNE_OUT) $(if $(MIN_DATE),--min-date $(MIN_DATE),) $(if $(KEEP_DAYS),--keep-days $(KEEP_DAYS),)

# Pré-calcul des métriques (Elo, firsts) à partir des parquets
METRICS_OUT ?= data/metrics
precompute: install
	PYTHONPATH=. $(PYTHON) scripts/precompute_metrics.py --processed $(OUT) --teams data/teams_to_look.csv --out $(METRICS_OUT)

# Sync processed + metrics to/from a VPS (rsync over SSH).
# Requires: rsync + ssh, and DOTA_DATA_REMOTE=user@host:/abs/path (in env or .env).
# Usage:
#   make sync-upload OUT=data/processed METRICS_OUT=data/metrics SYNC_SUBDIR=dota-datas
#   make sync-download OUT=data/processed METRICS_OUT=data/metrics SYNC_SUBDIR=dota-datas
SYNC_REMOTE ?=
SYNC_SUBDIR ?= dota-datas
SYNC_DELETE ?= 0
sync-upload: install
	PYTHONPATH=. $(PYTHON) scripts/sync_artifacts.py upload $(if $(SYNC_REMOTE),--remote $(SYNC_REMOTE),) --processed $(OUT) --metrics $(METRICS_OUT) --remote-subdir $(SYNC_SUBDIR) $(if $(filter 1 true yes,$(SYNC_DELETE)),--delete,)

sync-download: install
	PYTHONPATH=. $(PYTHON) scripts/sync_artifacts.py download $(if $(SYNC_REMOTE),--remote $(SYNC_REMOTE),) --processed $(OUT) --metrics $(METRICS_OUT) --remote-subdir $(SYNC_SUBDIR) $(if $(filter 1 true yes,$(SYNC_DELETE)),--delete,)

precompute-upload: precompute sync-upload

.PHONY: dashboard
dashboard: install
	$(PYTHON) -m streamlit run app/dashboard_streamlit.py
