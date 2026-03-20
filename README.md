<!-- ============================= -->
<!-- HEADER + BADGES -->
<!-- ============================= -->

# Economics & Analytics Portfolio — Alex Strong

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org)
[![Build](https://img.shields.io/badge/Build-Passing-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)]()

Applied economist and quantitative analyst with 7 years of professional
experience across federal labor economics, litigation analytics, and
private-sector data science. This portfolio contains reproducible Python
projects spanning causal inference, predictive modeling, time-series
forecasting, machine learning, and compliance analytics — each grounded
in real-world decision problems.

**M.A. Economics, American University · Former U.S. Bureau of Labor
Statistics Economist**

> All datasets are synthetic and generated programmatically for
> demonstration purposes. Code is tested and runs in GitHub Codespaces.

---

## Projects at a Glance

| Project | Method | Key Output |
|---|---|---|
| [Geographic Market Forecasting](#geographic-market-forecasting) | Ridge regression, holdout validation | 7.21% overall MAPE on unseen metros |
| [Causal Inference — DiD](#causal-inference--difference-in-differences) | Difference-in-differences | ATT estimate with parallel trends validation |
| [Bayesian Policy Evaluation](#bayesian-policy-evaluation) | Conjugate normal-normal Bayesian update | Posterior distribution of treatment effect |
| [Time-Series Modeling](#time-series-modeling) | ARIMA, VAR/IRF, GARCH | GDP forecasts, IRFs, volatility paths |
| [Wage Gap Analysis](#wage-gap-analysis) | OLS, Oaxaca-Blinder decomposition | Explained vs. unexplained wage gap |
| [Compliance & Timesheet Risk](#compliance--timesheet-risk-analytics) | Rule-based flagging, statistical analysis | Overtime exposure and back-wage liability |
| [Econometrics Suite](#econometrics-suite) | IV/2SLS, RD design, panel FE, PSM | Full applied econometrics toolkit |
| [Regularized Regression](#regularized-regression) | Ridge vs. OLS, cross-validation | Out-of-sample RMSE comparison |
| [Deep Learning — Income Classifier](#deep-learning--income-classifier) | PyTorch MLP | ROC-AUC vs. logistic baseline |
| [Credit Risk Classification](#credit-risk-classification) | Logistic regression, random forest | ROC-AUC, feature importance |
| [Solow Growth Model](#solow-growth-model) | Neoclassical growth simulation | Steady-state convergence paths |
| [CES Demand Estimation](#ces-demand-estimation) | CES utility maximization, NLS | Elasticity of substitution recovery |
| [BLS QCEW/CES Mock Program](#bls-qcewces-mock-program) | Synthetic data generation | BLS-structured employment and wage CSVs |

---

## Quickstart
```bash
# 1. Clone the repository
git clone https://github.com/alexstrong76/alex-strong-econ-analytics-portfolio.git
cd alex-strong-econ-analytics-portfolio

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run any project script from the repo root
python causal_inference/diff_in_diff_policy_evaluation.py
python geographic_market_forecasting/geo_market_forecast.py
python time_series/arima_gdp/arima_simulated.py
```

---

## Project Details

### Geographic Market Forecasting

**Folder:** `geographic_market_forecasting/`

Using employment history from established U.S. metro areas to forecast
labor market dynamics in unseen geographies before market entry —
simulating the core analytical problem in expansion planning.

- **Data:** BLS State and Metro Area Employment (CES), 2015–2023,
  via public API
- **Model:** Ridge regression on lagged employment, growth rate,
  and trend; holdout metros withheld entirely during training
- **Result:** Overall holdout MAPE of **7.21%** (Phoenix AZ: 5.59%,
  New York NY: 8.84%) using lagged features only, with no
  market-specific knowledge at prediction time
```bash
python geographic_market_forecasting/fetch_bls_data.py
python geographic_market_forecasting/geo_market_forecast.py
```

---

### Causal Inference — Difference-in-Differences

**Folder:** `causal_inference/`

Estimates the causal effect of a policy intervention using a two-way
panel DiD design with treated and control units observed before and
after a policy change.

- **Method:** OLS with treatment, time, and interaction terms via
  `statsmodels`; pre-trend visualization for assumption validation
- **Output:** ATT estimate with standard errors and parallel trends
  diagnostic plot
```bash
python causal_inference/diff_in_diff_policy_evaluation.py
```

---

### Bayesian Policy Evaluation

**Folder:** `bayesian/`

Applies a conjugate normal-normal Bayesian update to estimate the full
posterior distribution of a treatment effect, demonstrating the
relationship between Bayesian credible intervals and classical
confidence intervals.

- **Output:** Posterior mean, 95% credible interval, and probability
  that the treatment effect exceeds zero
```bash
python bayesian/bayes_posterior_policy_effect.py
```

---

### Time-Series Modeling

**Folder:** `time_series/`

Three complementary time-series methods applied to simulated
macroeconomic and financial data.

| Subfolder | Method | Output |
|---|---|---|
| `arima_gdp/` | ARIMA forecasting | GDP growth forecast with confidence intervals |
| `var_irf_cointegration/` | VAR, IRF, Engle-Granger cointegration | Impulse response functions, cointegration test |
| `garch_volatility/` | GARCH(1,1) | Conditional variance and volatility clustering |
```bash
python time_series/arima_gdp/arima_simulated.py
python time_series/var_irf_cointegration/var_irf_coint.py
python time_series/garch_volatility/garch_demo.py
```

---

### Wage Gap Analysis

**Folder:** `labor_econ/wage_gap_analysis/`

Estimates the wage gap between demographic groups using OLS regression
and an Oaxaca-Blinder decomposition, separating the raw gap into
explained (endowment) and unexplained (returns) components.

- **Data:** Synthetic cross-sectional wage data with experience,
  education, gender, and industry controls
```bash
python labor_econ/wage_gap_analysis/wage_gap.py
```

---

### Compliance and Timesheet Risk Analytics

**Folder:** `labor_law_compliance/wage_hour_audit/`

Identifies wage and hour compliance exposure — including overtime
violations and time-rounding liability — from synthetic timesheet
data modeled on real DOL audit structures.

- **Output:** Department-level risk rankings, estimated back-wage
  liability, and Tableau-ready CSVs
```bash
python labor_law_compliance/wage_hour_audit/wage_hour_audit.py
```

---

### Econometrics Suite

**Folder:** `econometrics/`

Full applied econometrics toolkit covering the core identification
strategies used in empirical economics and policy research.

| Subfolder | Method |
|---|---|
| `linear_regression/` | OLS estimation and diagnostics |
| `lpm_logit_probit/` | Binary outcome models |
| `iv_2sls/` | Instrumental variables / 2SLS |
| `panel_fixed_random/` | Fixed and random effects |
| `diff_in_diff/` | Difference-in-differences |
| `rd_design/` | Regression discontinuity |
| `psm_matching/` | Propensity score matching |
| `heteroskedasticity_robust_inference/` | Robust standard errors |
```bash
python econometrics/iv_2sls/iv_2sls.py
python econometrics/rd_design/rd_design.py
python econometrics/panel_fixed_random/panel_fe.py
```

---

### Regularized Regression

**Folder:** `regression/`

Compares Ridge regression against OLS on a synthetic labor market
dataset with correlated predictors, demonstrating regularization,
coefficient shrinkage paths, and cross-validated hyperparameter
selection.
```bash
python regression/regularized_employment_regression.py
```

---

### Deep Learning — Income Classifier

**Folder:** `deep_learning/`

Feed-forward neural network (PyTorch MLP) classifying individuals
into high vs. non-high income categories from demographic and human
capital features; compared against a logistic regression baseline
using ROC-AUC.
```bash
python deep_learning/pytorch_income_classifier.py
```

---

### Credit Risk Classification

**Folder:** `ml/`

Logistic regression and random forest classifiers applied to a
synthetic credit risk dataset; evaluated on ROC-AUC, precision-recall,
and feature importance to illustrate how ML methods complement
econometric approaches in risk modeling.
```bash
python ml/sklearn_credit_risk_model.py
```

---

### Solow Growth Model

**Folder:** `macro_models/solow_growth/`

Simulates capital accumulation and convergence to steady state under
the Solow-Swan neoclassical growth framework, with sensitivity
analysis across savings rates and identification of the golden rule.
```bash
python macro_models/solow_growth/solow.py
```

---

### CES Demand Estimation

**Folder:** `micro_models/ces_demand/`

Simulates consumer choice under a CES utility function and recovers
the elasticity of substitution via non-linear least squares, with
visualization of indifference curves and demand responses to price
changes.
```bash
python micro_models/ces_demand/ces_utility.py
```

---

### BLS QCEW/CES Mock Program

**Folder:** `bls_programs/qcew_ces_mock/`

Generates synthetic quarterly employment and wage data structured
to match BLS QCEW and CES field definitions, enabling reproducible
development of regional labor market analytics pipelines without
live API access.
```bash
python bls_programs/qcew_ces_mock/qcew_ces_mock.py
```

---

## Tableau Dashboards

Three story-driven Tableau dashboards built on the synthetic datasets
in `data/processed/`. Screenshots in `tableau/screenshots/`.

| Dashboard | Data Source |
|---|---|
| Labor Market & Wage Analytics | `wages_synthetic.csv` |
| Compliance & Timesheet Risk Analytics | `timesheets_synthetic.csv` |
| Macro & Time-Series Economic Trends | `gdp_growth_synthetic.csv`, `garch_like_returns.csv`, `var_cointegration.csv` |

---

## Synthetic Datasets

All datasets are auto-generated by their corresponding scripts and
saved to `data/processed/`. Run the script to regenerate any file.

| File | Generated by | Description |
|---|---|---|
| `wages_synthetic.csv` | `labor_econ/wage_gap_analysis/wage_gap.py` | Cross-sectional wage data with experience, education, gender, industry |
| `timesheets_synthetic.csv` | `labor_law_compliance/wage_hour_audit/wage_hour_audit.py` | Timesheet panel with hours, rounding, overtime, payroll |
| `employment_qcew_mock.csv` | `bls_programs/qcew_ces_mock/qcew_ces_mock.py` | QCEW-structured quarterly employment and wages |
| `solow_simulation.csv` | `macro_models/solow_growth/solow.py` | Capital, output, consumption convergence paths |
| `ces_utility_sim.csv` | `micro_models/ces_demand/ces_utility.py` | CES demand data across price and income levels |
| `gdp_growth_synthetic.csv` | `time_series/arima_gdp/arima_simulated.py` | AR(1)-style GDP growth series |
| `garch_like_returns.csv` | `time_series/garch_volatility/garch_demo.py` | Financial returns with GARCH volatility path |
| `var_cointegration.csv` | `time_series/var_irf_cointegration/var_irf_coint.py` | Two cointegrated macroeconomic series |
| `bayes_policy_effect.csv` | `bayesian/bayes_posterior_policy_effect.py` | Treated/control outcomes for Bayesian update |
| `did_policy_sim.csv` | `causal_inference/diff_in_diff_policy_evaluation.py` | Panel DiD simulation data |
| `credit_risk_synthetic.csv` | `ml/sklearn_credit_risk_model.py` | Synthetic borrower data for credit classification |
| `income_classifier_data.csv` | `deep_learning/pytorch_income_classifier.py` | Demographic features for income classification |
| `bls_metro_employment.csv` | `geographic_market_forecasting/fetch_bls_data.py` | Real BLS CES employment data for 20 U.S. metros |
| `holdout_forecast_results.csv` | `geographic_market_forecasting/geo_market_forecast.py` | Predicted vs. actual employment — holdout metros |

---

## Repository Structure
```
geographic_market_forecasting/
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
causal_inference/
bayesian/
regression/
ml/
deep_learning/
data/
  ├── raw/
  └── processed/
notebooks/
tableau/
  └── screenshots/
tests/
.github/
  └── workflows/
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Languages** | Python, R, SQL |
| **Econometrics** | statsmodels, linearmodels |
| **Machine Learning** | scikit-learn, PyTorch |
| **Time-Series** | statsmodels, arch |
| **Data** | pandas, NumPy, SciPy |
| **Visualization** | Tableau, Power BI, matplotlib, seaborn |
| **Workflow** | Git, Jupyter, GitHub Actions, Makefile |

---

## Testing and Linting
```bash
# Run tests
pytest -q

# Lint
flake8 .
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Alex Strong**
[strong.nalex@gmail.com](mailto:strong.nalex@gmail.com) ·
[LinkedIn](https://linkedin.com/in/alex-s-b049a4390) ·
[GitHub](https://github.com/alexstrong76)
