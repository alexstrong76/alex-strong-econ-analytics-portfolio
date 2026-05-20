# U.S. Housing Market Econometric Dashboard
**Fed Policy, Affordability & Regional Dynamics | 2018–2024**

## Overview
A macroeconomic analysis of the U.S. housing market across the 2018–2024 interest rate cycle, linking Federal Reserve policy to home price dynamics, affordability, regional market divergence, and supply-demand indicators. Includes OLS regression estimating the lagged effect of mortgage rate changes on home price growth, calibrated to publicly available data.

This project demonstrates skills in macroeconomic forecasting, regional economic analysis, data visualization, and policy-relevant research communication — applicable to real estate economics, housing finance, and applied macro research roles.

## Research Questions
1. How strongly do mortgage rate changes (lagged) predict home price growth?
2. What is the current state of housing affordability relative to historical norms?
3. Which metro markets outperformed or underperformed the national average across the rate cycle?
4. Where does the supply-demand balance sit heading into 2025?

## Key Findings

### Monetary Policy Transmission
- A 1pp increase in the 30-year mortgage rate (lagged 3 months) is associated with a **1.74pp decrease in months of supply**, indicating rapid demand destruction as rates rise.
- OLS model (HPI growth ~ mortgage rate lag 3mo + months of supply lag 1mo + existing sales): **R² = 0.36**, suggesting macro factors explain a substantial but not complete share of short-run price variation.

### Affordability Crisis
- The payment-to-income ratio breached the traditional 28% guideline in 2022 and remained elevated through 2024, driven by the simultaneous peak in home prices and mortgage rates.
- The 2020–2022 pandemic boom produced the fastest sustained home price growth in the dataset (+18%+ YoY at peak).

### Regional Divergence
- **Dallas, TX** and **Austin, TX** outperformed the national average across the full cycle, driven by population inflows, relative supply responsiveness, and strong labor market fundamentals.
- **Chicago, IL** showed the most muted appreciation, consistent with slower population growth and higher existing supply.

## Methodology

### Data
All data is simulated/modeled to match publicly reported indicators:
- Federal Funds Rate: Federal Reserve H.15 release
- 30-yr Mortgage Rate: Freddie Mac PMMS
- Home Price Index: calibrated to Case-Shiller (Jan 2018 = 100)
- Housing Starts: U.S. Census Bureau
- Existing Home Sales: National Association of Realtors

### OLS Regression
```
HPI_growth_YoY = β₀ + β₁(Mortgage_rate, lag 3mo) + β₂(Months_supply, lag 1mo) + β₃(Existing_sales) + ε
```

### Affordability Index
Monthly payment-to-income ratio using:
- Median home price (HPI-scaled from 2018 baseline of $260,000)
- 30-year fixed mortgage rate
- Median household income (4% annual growth assumed)

## Dashboard Panels
1. National HPI growth vs. mortgage rates (dual-axis, 2018–2024)
2. Housing affordability: payment-to-income vs. 28% guideline
3. Regional HPI comparison (6 metros, Jan 2018 = 100)
4. OLS regression: actual vs. fitted HPI growth
5. Housing starts vs. existing home sales
6. Months of supply (seller's vs. buyer's market threshold)
7. Key statistics and regression coefficient summary

## Outputs

| File | Description |
|------|-------------|
| `housing_market_analysis.py` | Main analysis and visualization script |
| `housing_market_dashboard.png` | 7-panel dashboard |
| `housing_market_data.csv` | Monthly panel dataset (84 observations) |

## Policy Implications
The analysis illustrates how Fed rate decisions transmit to housing markets with meaningful lags (3–6 months to price, 1–2 months to transaction volume), and how structural undersupply acts as a floor on price declines even in periods of severe affordability deterioration — a dynamic central to the post-2022 price stickiness observed nationally.

---
*Built by Alex Strong | Economist & Quantitative Analyst | [GitHub Portfolio](https://github.com/alexstrong76/alex-strong-econ-analytics-portfolio)*
