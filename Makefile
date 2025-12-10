VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install activate notebook deps-clean parquet

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip >/dev/null

install: venv
	@$(PIP) install -q -r requirements.txt && echo "Dependencies up to date."

activate:
	@echo "Pour activer le venv: source $(VENV)/bin/activate"

notebook: install 
	$(PYTHON) -m ipykernel install --user --name dota-data --display-name "Python (dota-data)"
	$(PYTHON) -m jupyter lab

deps-clean:
	find $(VENV)/lib -name '__pycache__' -type d -prune -exec rm -rf {} +

# Génère les parquets à partir du brut (par défaut data/raw/data_v2.json -> data/processed)
# Usage: make parquet [RAW=path/to/raw.json] [OUT=path/to/processed_dir]
RAW ?= data/raw/data_v2.json
OUT ?= data/processed
parquet: install
	$(PYTHON) -m src.dota_data.io --raw $(RAW) --out $(OUT)

# Pré-calcul des métriques (Elo, firsts) à partir des parquets
METRICS_OUT ?= data/metrics
precompute: install
	PYTHONPATH=. $(PYTHON) scripts/precompute_metrics.py --processed $(OUT) --teams data/teams_to_look.csv --out $(METRICS_OUT)

.PHONY: dashboard
dashboard: install
	$(PYTHON) -m streamlit run app/dashboard_streamlit.py
