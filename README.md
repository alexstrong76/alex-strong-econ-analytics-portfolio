# Economics & Analytics Portfolio — Alex Strong

This repository showcases academic and professional projects at the intersection of **economics**, **data science**, and **labor policy**.

**Highlights**
- Macroeconomics & Microeconomics modeling (Solow, CES demand)
- Econometrics (OLS), **time series** (ARIMA)
- **Labor economics**: wage gap analysis, BLS-style employment trends
- **Wage & Hour** compliance analytics (synthetic data)

> All datasets here are **synthetic**, for demonstration only.

## Quickstart
```bash
# create & activate a virtual environment (optional)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run a project
python macro_models/solow_growth/solow.py
python micro_models/ces_demand/ces_utility.py
python econometrics/linear_regression/ols_sklearn.py
python time_series/arima_gdp/arima_simulated.py
python labor_econ/wage_gap_analysis/wage_gap.py
python labor_law_compliance/wage_hour_audit/wage_hour_audit.py
python bls_programs/qcew_ces_mock/qcew_ces_mock.py

## Structure 
macro_models/
  solow_growth/
micro_models/
  ces_demand/
econometrics/
  linear_regression/
time_series/
  arima_gdp/
labor_econ/
  wage_gap_analysis/
labor_law_compliance/
  wage_hour_audit/
bls_programs/
  qcew_ces_mock/

## License 
## Python requirements (`requirements.txt`)
```txt
pandas
numpy
matplotlib
scikit-learn
statsmodels
