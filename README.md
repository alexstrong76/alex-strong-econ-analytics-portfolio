<!-- ============================= -->
<!-- HEADER + BADGES -->
<!-- ============================= -->

# Economics & Analytics Portfolio — Alex Strong  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Build-Passing-success.svg)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)
![Notebooks](https://img.shields.io/badge/Notebooks-available-blueviolet.svg)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)
![Data Science](https://img.shields.io/badge/Focus-Econometrics%20%7C%20Labor%20Analytics-lightgrey.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Contributions%20Welcome-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

> Pytest unit tests for key econometrics modules, and example Jupyter notebooks for IV/2SLS and Difference-in-Differences.

---

## Overview  

This portfolio showcases **academic and professional analytics projects** bridging *economics, data science,* and *labor policy*.  
Each project demonstrates an applied approach to modeling, analysis, and storytelling through data — ranging from macroeconomic simulations to compliance analytics.

**Key themes:**
- **Economic modeling:** Solow growth, CES demand, econometric regression, ARIMA forecasting  
- **Labor & policy analytics:** Wage-gap regression, synthetic Department of Labor “Wage & Hour” audit data  
- **Applied research methods:** OLS estimation, causal analysis foundations, time-series decomposition  
- **Data tools:** Python, pandas, NumPy, scikit-learn, statsmodels, Jupyter, Tableau-ready data outputs  

> All data are **synthetic** and created for educational or demonstration purposes only.
>
> Code tested and run using GitHub Codespaces
> Data for dashboards in data/processed/
> Dashboards built externally in Tableau using these CSVs

---

## Quickstart

```bash
# 1) Clone
git clone https://github.com/<your-username>/alex-strong-econ-analytics-portfolio.git
cd alex-strong-econ-analytics-portfolio

# 2) Create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run example scripts
python macro_models/solow_growth/solow.py
python micro_models/ces_demand/ces_utility.py

# Econometrics (examples)
python econometrics/iv_2sls/iv_2sls.py
python econometrics/diff_in_diff/did_basic.py
python econometrics/heteroskedasticity_robust_inference/robust_se_diagnostics.py

# Time series (examples)
python time_series/arima_gdp/arima_simulated.py
python time_series/var_irf_cointegration/var_irf_coint.py
python time_series/garch_volatility/garch_demo.py

# Labor & compliance (examples)
python labor_econ/wage_gap_analysis/wage_gap.py
python labor_law_compliance/wage_hour_audit/wage_hour_audit.py
python bls_programs/qcew_ces_mock/qcew_ces_mock.py

## Repository Structure 
macro_models/
  └── solow_growth/
micro_models/
  └── ces_demand/
econometrics/
  ├── linear_regression/
  ├── lpm_logit_probit/
  ├── iv_2sls/
  ├── panel_fixed_random/
  ├── diff_in_diff/
  ├── rd_design/
  ├── psm_matching/
  └── heteroskedasticity_robust_inference/
time_series/
  ├── arima_gdp/
  ├── var_irf_cointegration/
  └── garch_volatility/
labor_econ/
  └── wage_gap_analysis/
labor_law_compliance/
  └── wage_hour_audit/
bls_programs/
  └── qcew_ces_mock/
data/
  └── processed/          
notebooks/
tools/
.github/
  └── workflows/

## Requirements <requirements.txt>
numpy>=1.24,<3.0
pandas>=2.0,<3.0
scipy>=1.10,<2.0
statsmodels>=0.14,<0.15
scikit-learn
matplotlib
arch
linearmodels

Testing & Linting
# Tests (if tests/ exists)
python -m pip install pytest
pytest -q

# Lint
python -m pip install flake8
flake8 .
[flake8]
exclude = .git,__pycache__,.github,venv,env
max-line-length = 100

Contributing
PRs and issues welcome. Please format with Black and pass flake8 before submitting.

License
MIT License — see LICENSE for details.


