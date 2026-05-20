# Spotify Revenue Driver Model
**Driver-Based Financial Forecast | FY2021A – FY2027E**

## Overview
A bottom-up, driver-based financial model forecasting Spotify's Premium subscription and Ad-Supported revenue through FY2027, built in Excel using publicly available data from Spotify's annual reports and shareholder letters.

This project demonstrates financial modeling skills directly applicable to strategic finance, FP&A, and business analytics roles — including driver-based forecasting, scenario analysis, sensitivity tables, and executive-ready visualization.

## Why Spotify?
Spotify operates a two-sided marketplace (listeners and advertisers) with a freemium conversion model — structurally similar to Instacart, Airbnb, and other consumer marketplace businesses. Modeling it requires understanding unit economics (MAU → Subscriber conversion, ARPU, ad load monetization) that translate directly across industries.

## Model Structure

| Tab | Contents |
|-----|----------|
| **Cover** | Navigation guide and model overview |
| **Raw Data** | Historical actuals (FY2021–FY2024) sourced from Spotify annual reports |
| **Assumptions** | All key drivers — edit blue cells; scenario toggle (Base/Bull/Bear) |
| **Income Statement** | Full revenue build + P&L summary, dynamically linked to Assumptions |
| **Scenario Analysis** | Side-by-side Bear/Base/Bull outputs + sensitivity table (heat map) |
| **Charts** | Dynamic dashboard: revenue mix, MAU trajectory, ARPU trends |

## Key Modeling Drivers

**Premium Revenue**
- MAU growth rate (YoY%)
- Premium conversion rate (Subscribers / MAU)
- Monthly ARPU (EUR)
- → Premium Revenue = Subscribers × ARPU × 12

**Ad-Supported Revenue**
- Ad-Supported MAU = Total MAU − Premium Subscribers
- Ad Revenue per Ad MAU (EUR/year, driven by ad load and CPM)
- → Ad Revenue = Ad MAU × Ad Revenue per MAU

**P&L**
- Gross margin assumption (trending toward 30%+ as podcast/audiobook mix improves)
- R&D, S&M, G&A as % of revenue

## Scenarios

| Scenario | MAU Growth (2027E) | Gross Margin (2027E) | Operating Margin |
|----------|--------------------|----------------------|-----------------|
| Bear | 6% | 27.5% | ~(1.2%) |
| Base | 9.5% | 31.5% | ~10.0% |
| Bull | 12% | 35.0% | ~20.0% |

## Sensitivity Analysis
The Scenario Analysis tab includes a 6×6 heat map of 2027E Operating Income across combinations of MAU growth rate and gross margin — the two highest-leverage variables in the model.

## Color Coding (Industry Standard)
- **Blue text**: Hardcoded inputs (change these to run scenarios)
- **Black text**: Formulas and calculations (do not edit)
- **Green text**: Cross-sheet links pulling from Raw Data or Assumptions

## Data Sources
All historical actuals sourced from:
- Spotify Q4 Shareholder Letters (FY2021–FY2024)
- Spotify Annual Reports on Form 20-F (SEC EDGAR)
- Figures reported in EUR millions unless noted

## File
`Spotify_Revenue_Model.xlsx` — open in Excel (2016 or later recommended for full chart rendering)

---
*Built by Alex Strong | Economist & Quantitative Analyst | [GitHub Portfolio](https://github.com/alexstrong76/alex-strong-econ-analytics-portfolio)*
