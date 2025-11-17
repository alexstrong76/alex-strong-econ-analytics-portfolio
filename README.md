<!-- ============================= -->
<!-- HEADER + BADGES -->
<!-- ============================= -->

# Economics & Analytics Portfolio — Alex Strong  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Build-Passing-success.svg)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)
![Data Science](https://img.shields.io/badge/Focus-Econometrics%20%7C%20Labor%20Analytics-lightgrey.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Contributions%20Welcome-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 🌍 Overview  

This portfolio showcases **academic and professional analytics projects** bridging *economics, data science,* and *labor policy*.  
Each project demonstrates an applied approach to modeling, analysis, and storytelling through data — ranging from macroeconomic simulations to compliance analytics.

**Key themes:**
- **Economic modeling:** Solow growth, CES demand, econometric regression, ARIMA forecasting  
- **Labor & policy analytics:** Wage-gap regression, synthetic Department of Labor “Wage & Hour” audit data  
- **Applied research methods:** OLS estimation, causal analysis foundations, time-series decomposition  
- **Data tools:** Python, pandas, NumPy, scikit-learn, statsmodels, Jupyter, Tableau-ready data outputs  

> All data are **synthetic** and created for educational or demonstration purposes only.

---

## 🚀 Quickstart  

```bash
# 1️⃣ Clone the repo
git clone https://github.com/<your-username>/alex-strong-econ-analytics-portfolio.git
cd alex-strong-econ-analytics-portfolio

# 2️⃣ Create & activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
pip install -r dev-requirements.txt   # optional

# 4️⃣ Generate example datasets & notebooks
make data
make nb

# 5️⃣ Run sample models
make examples

---

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

## Repository Structure
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
data/
  processed/
notebooks/
tools/

## License 
## Python requirements (`requirements.txt`)
```txt
pandas
numpy
matplotlib
scikit-learn
statsmodels

🧠 Learn More
🧮 Econometrics: Classic OLS + modern scikit-learn regression examples
📈 Macroeconomics: Solow-style convergence visualization
⚙️ Policy analytics: Wage-and-hour audit modeling for compliance insights
📊 Visualization: Outputs compatible with Tableau and Power BI
🤝 Contributing

Pull requests, issues, and suggestions are welcome!
Please ensure code is formatted with Black and passes flake8 linting before submitting.

🪪 License
This project is distributed under the MIT License — see LICENSE for details.
