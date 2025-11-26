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

## Bayesian Inference, Machine Learning, and Deep Learning Projects

This repository now includes advanced projects bridging econometric modeling, causal inference, and applied data science.

| Area | Folder | Description |
|------|---------|-------------|
| Bayesian Inference | `bayesian/bayes_posterior_policy_effect.py` | Bayesian posterior estimation of a policy effect using a conjugate normal-normal model. |
| Regression | `regression/regularized_employment_regression.py` | Ridge and OLS regression for employment data, exploring feature regularization. |
| Causal Inference | `causal_inference/diff_in_diff_policy_evaluation.py` | Simulated DiD policy analysis with pre/post treatment evaluation. |
| Deep Learning | `deep_learning/pytorch_income_classifier.py` | PyTorch MLP classifier predicting high-income earners from synthetic demographic data. |
| Machine Learning | `ml/sklearn_credit_risk_model.py` | Credit risk classification using logistic regression and random forest models. |

Each script generates reproducible datasets in `data/processed/` for further visualization and model benchmarking.

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
    wage_gap.py
labor_law_compliance/
  └── wage_hour_audit/
bls_programs/
  └── qcew_ces_mock/
data/
  └── processed/          
notebooks/
scripts/
tableau/
  └── screenshots/   
tools/
tests/
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

## Synthetic Datasets (Auto-Generated via Python Scripts)
This portfolio includes a suite of reproducible synthetic datasets generated by purpose-built Python scripts. These datasets mirror real-world structures seen in labor economics, compliance analytics, corporate forecasting, and macroeconomic time-series work. All CSVs are programmatically created and saved to:
data/processed/

## Below is a reference guide to every dataset, the script that produces it, and typical use cases.

# Labor Economics & Compliance
wages_synthetic.csv
Generated by: labor_econ/wage_gap_analysis/wage_gap.py
Description:
Simulated cross-sectional wage data containing years of experience, education, gender, and industry dummies. Used for wage regression, decomposition, and OLS demonstrations.
timesheets_synthetic.csv
Generated by: labor_econ/wage_hour_audit.py
Description:
A realistic timesheet panel including hours worked, rounding increments, overtime, department assignment, and payroll estimates.
Use cases:
Wage & hour compliance analysis
Overtime cost modeling
Rounding impact dashboards (Storyboard 2)

# Industry, Employment, and Growth Analytics
employment_qcew_mock.csv
Generated by: macro/qcew_ces_mock.py
Description:
Quarterly industry employment and wage data modeled on BLS QCEW/CES fields.
Use cases:
Trends dashboards
Industry comparison
Growth decomposition
solow_simulation.csv
Generated by: macro/solow.py
Description:
Simulated Solow model with capital, output, and consumption paths.
Use cases:
Growth theory visualization
Policy sensitivity analysis
Teaching/academic applications

# Microeconomics & Behavioral Modeling
ces_utility_sim.csv
Generated by: micro/ces_utility.py
Description:
CES utility and demand data across consumers with varying prices and income levels.
Use cases:
Consumer behavior modeling
Utility surfaces / indifference curves
Microeconomic fundamentals

# Time-Series, Forecasting & Financial Econometrics
gdp_growth_synthetic.csv
Generated by: time_series/arima_simulated.py
Description:
AR(1)-style GDP growth series with cumulative index.
Use cases:
Forecasting demonstrations
Trend & shock visualizations
GDP dashboards (Storyboard 3)
garch_like_returns.csv
Generated by: time_series/garch_demo.py
Description:
Financial returns and volatility paths from a GARCH(1,1)-like model.
Use cases:
Risk dashboards
Volatility modeling

# Portfolio analytics demos
var_cointegration.csv
Generated by: time_series/var_irf_coint.py
Description:
Two cointegrated series that track long-run equilibrium relationships.
Use cases:
VAR estimation
IRF visualization
Cointegration testing (Storyboard 3)
All datasets are reproducible — simply run the corresponding script from the project root:
python path/to/script.py

## Tableau Storyboards

This repository includes three Tableau story-driven dashboards built on the synthetic datasets in `data/processed/`.

1. **Labor Market & Wage Analytics (Storyboard 1)**  
   - Workbook: `tableau/Storyboard1_Labor_Market_Wage_Analytics.twbx`  
   - Data: `data/processed/wages_synthetic.csv`  

2. **Compliance & Timesheet Risk Analytics (Storyboard 2)**  
   - Workbook: `tableau/Storybboard2_Compliance_Timesheet_Risk.twbx`  
   - Data: `data/processed/timesheets_synthetic.csv`  

3. **Macro & Time-Series Economic Trends (Storyboard 3)**  
   - Workbook: `tableau/Storyboard3_Macro_Time_Series_Trends.twbx`  
   - Data:  
     - `data/processed/gdp_growth_synthetic.csv`  
     - `data/processed/garch_like_returns.csv`  
     - `data/processed/var_cointegration.csv`

## Advanced Inference & Machine Learning Projects

This portfolio also includes a set of advanced, production-style examples that bridge traditional econometrics with modern machine learning and Bayesian methods.

### Bayesian Inference

- **Script:** `bayesian/bayes_posterior_policy_effect.py`  
- **Dataset:** `data/processed/bayes_policy_effect.csv`  
- **Description:**  
  Simulates a simple policy evaluation setting (treated vs. control group) and applies a conjugate normal–normal Bayesian update to estimate the posterior distribution of the treatment effect. Demonstrates the relationship between classical diff-in-means estimators and Bayesian posterior inference.

### Regularized Regression Modeling

- **Script:** `regression/regularized_employment_regression.py`  
- **Dataset:** `data/processed/regularized_employment_regression.csv`  
- **Description:**  
  Generates a synthetic employment-related wage dataset and fits both OLS and Ridge regression models using scikit-learn. Designed to illustrate regularization, multicollinearity, and model comparison in an applied labor economics context.

### Causal Inference (Difference-in-Differences)

- **Script:** `causal_inference/diff_in_diff_policy_evaluation.py`  
- **Dataset:** `data/processed/did_policy_sim.csv`  
- **Description:**  
  Implements a two-way panel difference-in-differences setup with treated and control units before and after a policy change. Estimates a DiD model using statsmodels, providing a clean example of policy evaluation with simulated data.

### Deep Learning with PyTorch

- **Script:** `deep_learning/pytorch_income_classifier.py`  
- **Dataset:** `data/processed/income_classifier_data.csv`  
- **Description:**  
  Builds a simple feed-forward neural network in PyTorch to classify individuals into high vs. non-high income categories based on age, education, and experience. Includes a minimal training loop and evaluation output, demonstrating MLPs and modern DL tooling on tabular data.

### Machine Learning with scikit-learn

- **Script:** `ml/sklearn_credit_risk_model.py`  
- **Dataset:** `data/processed/credit_risk_synthetic.csv`  
- **Description:**  
  Simulates a credit risk classification problem and fits both Logistic Regression and Random Forest models using scikit-learn. Evaluates model performance using ROC AUC and classification metrics, illustrating how ML methods complement econometric approaches in risk modeling and decision support.

---

All of these scripts are designed to be:

- **Reproducible:** They generate their own synthetic datasets in `data/processed/`.
- **Self-contained:** They can be run from the repository root using `python path/to/script.py`.
- **Portfolio-ready:** Each example aligns with real-world use cases in economic research, policy analysis, compliance analytics, and risk modeling.


  
