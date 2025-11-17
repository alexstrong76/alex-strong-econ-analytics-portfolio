# ---- Config ----
PY := python
VENV := .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

# ---- Setup ----
.PHONY: setup
setup: ## Create venv & install deps (prod + dev)
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r dev-requirements.txt
	$(VENV)/bin/pre-commit install

# ---- Quality ----
.PHONY: lint
lint: ## Run linters/formatters
	$(VENV)/bin/black .
	$(VENV)/bin/isort .
	$(VENV)/bin/flake8 .

.PHONY: check
check: ## Check without writing
	$(VENV)/bin/black --check .
	$(VENV)/bin/isort --check-only .
	$(VENV)/bin/flake8 .

# ---- Notebooks ----
.PHONY: nb
nb: ## Generate Jupyter notebooks from scripts (tools/make_notebooks.py)
	$(PYBIN) tools/make_notebooks.py

.PHONY: nb-clean
nb-clean: ## Strip output from notebooks (pre-commit hook does this too)
	$(VENV)/bin/nbstripout --install

# ---- Data ----
.PHONY: data
data: ## Create synthetic CSVs for BI/visualization demos
	$(PYBIN) data/seed_datasets.py

# ---- Examples ----
.PHONY: examples
examples: ## Run a few example scripts
	$(PYBIN) macro_models/solow_growth/solow.py
	$(PYBIN) econometrics/linear_regression/ols_sklearn.py
	$(PYBIN) labor_econ/wage_gap_analysis/wage_gap.py
	$(PYBIN) labor_law_compliance/wage_hour_audit/wage_hour_audit.py

# ---- Help ----
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
