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

## dev-requirements.txt
black
isort
flake8
pre-commit
nbstripout
nbformat

## .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
make setup 

## tools/make_notebooks.py
"""
Generate minimal Jupyter notebooks from selected scripts so reviewers can
open & run cells easily. Keeps notebooks lightweight (1-2 cells each).
"""
import json
import os
from pathlib import Path
import nbformat as nbf

# map: output notebook path -> list of code snippets (str)
TARGETS = {
    "notebooks/solow_growth.ipynb": [
        "# Solow Growth Model — Notebook Demo\n"
        "Use this to visualize convergence and steady-state intuition.\n",
        (Path("macro_models/solow_growth/solow.py").read_text() if Path("macro_models/solow_growth/solow.py").exists() else "print('Script missing')")
    ],
    "notebooks/ols_demo.ipynb": [
        "# OLS with Synthetic Data — Notebook Demo\n"
        "Simple linear regression with scikit-learn.\n",
        (Path("econometrics/linear_regression/ols_sklearn.py").read_text() if Path("econometrics/linear_regression/ols_sklearn.py").exists() else "print('Script missing')")
    ],
    "notebooks/wage_hour_audit.ipynb": [
        "# Wage & Hour Audit — Notebook Demo\n"
        "Synthetic overtime and pay flags (FLSA-style toy example).\n",
        (Path("labor_law_compliance/wage_hour_audit/wage_hour_audit.py").read_text() if Path("labor_law_compliance/wage_hour_audit/wage_hour_audit.py").exists() else "print('Script missing')")
    ],
}

def to_notebook(md_text: str, code_text: str):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(md_text))
    nb.cells.append(nbf.v4.new_code_cell(code_text))
    return nb

def main():
    Path("notebooks").mkdir(parents=True, exist_ok=True)
    for nb_path, parts in TARGETS.items():
        md, code = parts
        nb = to_notebook(md, code)
        nbf.write(nb, nb_path)
        print(f"written -> {nb_path}")

if __name__ == "__main__":
    main()
make nb

## data/seed_datasets.py
"""
Generate synthetic CSVs for BI dashboards and examples.
Outputs -> data/processed/*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(123)

def wages_dataset():
    n = 2000
    years_exp = rng.integers(0, 30, size=n)
    edu = rng.choice([12, 14, 16, 18], size=n, p=[0.2, 0.3, 0.4, 0.1])
    gender = rng.choice(["F", "M"], size=n)
    industry = rng.choice(["tech", "health", "retail", "public"], size=n, p=[0.35,0.25,0.25,0.15])

    base = 15 + 1.2*years_exp + 2.0*(edu-12) + (gender == "M")*3.0
    prem = {"tech": 7, "health": 4, "retail": 1, "public": 2}
    noise = rng.normal(0, 3, size=n)
    wage = base + pd.Series(industry).map(prem).values + noise

    df = pd.DataFrame({
        "wage": wage.round(2),
        "years_exp": years_exp,
        "edu_years": edu,
        "gender": gender,
        "industry": industry
    })
    df.to_csv(OUT / "wages_synthetic.csv", index=False)

def timesheets_dataset():
    n = 1200
    df = pd.DataFrame({
        "emp_id": rng.integers(1000, 1100, size=n),
        "week": rng.integers(1, 13, size=n),
        "hourly_rate": rng.choice([15, 18, 22, 28], size=n, p=[0.3, 0.3, 0.25, 0.15]),
        "hours": np.clip(rng.normal(42, 6, size=n), 20, 70),
        "rounded_to": rng.choice([1, 6, 15], size=n, p=[0.2, 0.5, 0.3]),
        "dept": rng.choice(["ops","sales","admin","it"], size=n, p=[0.4,0.25,0.2,0.15]),
    })
    # derived
    df["ot_hours"] = (df["hours"] - 40).clip(lower=0)
    df["reg_hours"] = df["hours"] - df["ot_hours"]
    df["reg_pay"] = df["reg_hours"] * df["hourly_rate"]
    df["ot_pay"]  = df["ot_hours"] * df["hourly_rate"] * 1.5
    df["gross"]   = df["reg_pay"] + df["ot_pay"]
    df.to_csv(OUT / "timesheets_synthetic.csv", index=False)

def gdp_growth_dataset():
    T = 40  # 10 years quarterly
    eps = rng.normal(0, 0.6, size=T)
    g = 0.5 + 0.6*np.roll(eps,1)
    g[0] = eps[0]
    gdp_idx = 100 * np.cumprod(1 + g/100.0)
    periods = pd.period_range("2016Q1", periods=T, freq="Q")
    pd.DataFrame({"period": periods.astype(str), "g": g.round(3), "gdp_index": gdp_idx.round(2)}).to_csv(
        OUT / "gdp_growth_synthetic.csv", index=False
    )

def qcew_mock_dataset():
    industries = ["NAICS-54","NAICS-62","NAICS-44","NAICS-92"]
    quarters = pd.period_range("2019Q1","2025Q4", freq="Q")
    rows = []
    for ind in industries:
        level = 1000 + rng.integers(-100, 100)
        for q in quarters:
            growth = rng.normal(0.0, 0.02)
            level = max(100, level*(1+growth))
            rows.append((ind, str(q), int(level)))
    df = pd.DataFrame(rows, columns=["industry","quarter","employment"])
    df["employment_lag4"] = df.groupby("industry")["employment"].shift(4)
    df["employment_lag1"] = df.groupby("industry")["employment"].shift(1)
    df["YoY_%"] = 100*(df["employment"]/df["employment_lag4"] - 1)
    df["QoQ_%"] = 100*(df["employment"]/df["employment_lag1"] - 1)
    df.to_csv(OUT / "employment_qcew_mock.csv", index=False)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wages_dataset()
    timesheets_dataset()
    gdp_growth_dataset()
    qcew_mock_dataset()
    print(f"Wrote CSVs to: {OUT.resolve()}")
make data

## Recap
make setup
make nb          # builds the 3 notebooks under notebooks/
make data        # builds BI CSVs under data/processed/
make examples    # runs a few scripts end-to-end
make nb-addons     # build IV & DiD notebooks
make data-more     # write new CSVs to data/processed/
make test          # run unit tests






