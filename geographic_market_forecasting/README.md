# Geographic Market Expansion Forecasting

## Business Question
Given historical employment and growth data from established U.S.
metro areas, how accurately can we forecast labor market dynamics
in unseen metros before market entry?

This mirrors a real expansion planning problem: a company with
data from existing markets needs to predict performance in new
geographies where no historical observations exist. The model
is trained exclusively on metros it has never seen during
evaluation — simulating the conditions of a genuine pre-entry
forecast.

## Method
- **Data source:** U.S. Bureau of Labor Statistics — State and
  Metro Area Employment, Hours & Earnings (CES), 2015–2023,
  pulled via the BLS public API
- **Training set:** Metro and state-level employment series
  with sufficient historical coverage
- **Holdout set:** 2 metros withheld entirely during training
  (New York NY, Phoenix AZ) — the model never sees these
  during fitting
- **Features:**
  - Lagged employment level (t-1)
  - Lagged employment level (t-2)
  - Year-over-year growth rate (t-1)
  - Linear trend term
- **Model:** Ridge regression with L2 regularization
  (alpha=10.0), features standardized via `StandardScaler`
- **Evaluation:** Mean Absolute Percentage Error (MAPE)
  on the held-out metros

## Results

| Holdout Metro | Forecast MAPE |
|---------------|---------------|
| New York NY   | 8.84%         |
| Phoenix AZ    | 5.59%         |
| **Overall**   | **7.21%**     |

Phoenix forecast error (5.59%) is within typical bounds for
regional employment forecasting using lagged features alone,
without any market-specific covariates. New York's higher error
(8.84%) reflects the outsized volatility in its employment
series during the 2020–2021 period, which the lagged-feature
model partially absorbs but does not fully capture. Both results
are consistent with a baseline model that uses no market-specific
knowledge about either metro at prediction time.

## Visualizations
![Predicted vs actual employment — holdout metros](figures/holdout_forecast_by_metro.png)
![Forecast error by metro](figures/mape_by_metro.png)
![Ridge regression feature coefficients](figures/feature_coefficients.png)

## How to Run
```bash
# From the geographic_market_forecasting/ directory:

# Step 1 — fetch employment data from BLS API
python3 fetch_bls_data.py

# Step 2 — build model, generate figures, save results
python3 geo_market_forecast.py
```

Output figures are saved to `figures/`.
Forecast results are saved to `data/processed/holdout_forecast_results.csv`.

## Limitations and Next Steps

**Data coverage:** The current model relies on state-level series
as proxies for several metros where MSA-level BLS series IDs
were unavailable via the public API. A production version would
use confirmed MSA-level QCEW or CES data for each target market,
improving geographic precision.

**Feature set:** The model uses only lagged employment and a
trend term. Adding wage growth rates, industry composition
(share of employment in high-growth sectors), population growth,
and local cost-of-living indices would likely reduce holdout
error meaningfully — particularly for structurally unusual
markets like New York.

**COVID distortion:** The 2020–2021 employment shock creates a
discontinuity in the training data that lagged features partially
absorb but do not fully model. A production version would include
an explicit recession/recovery indicator or restrict the
evaluation window to 2022–2023 post-shock data.

**Uncertainty quantification:** The current model produces point
estimates only. A production forecast should report prediction
intervals — for example via quantile regression or bootstrapped
residuals — so that decision-makers understand the range of
plausible outcomes, not just the expected value.

**Model architecture:** Ridge regression is a strong and
interpretable baseline. Gradient boosting (XGBoost, LightGBM)
or a per-metro ARIMA ensemble would likely reduce MAPE on
markets with non-linear growth trajectories, at the cost of
interpretability.

## Tools
Python · pandas · scikit-learn · matplotlib · BLS Public API