VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install activate notebook deps-clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

activate:
	@echo "Pour activer le venv: source $(VENV)/bin/activate"

notebook: install
	$(PYTHON) -m ipykernel install --user --name dota-data --display-name "Python (dota-data)"
	$(PYTHON) -m jupyter lab

deps-clean:
	find $(VENV)/lib -name '__pycache__' -type d -prune -exec rm -rf {} +
