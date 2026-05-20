# Consumer Credit Risk Scorecard
**Logistic Regression + Weight of Evidence Analysis | FICO-Style Scoring Model**

## Overview
A production-style consumer credit risk scorecard built using logistic regression with Weight of Evidence (WoE) feature analysis and Information Value (IV) assessment — the standard methodology used by FICO, credit bureaus, and financial institutions for retail lending decisions.

The model scores 15,000 simulated consumer loan applicants on a 300–850 scale and evaluates scorecard performance using industry-standard metrics (AUC, KS statistic, calibration).

This project demonstrates skills in predictive modeling, credit analytics, and financial risk assessment — directly applicable to roles at FICO, Experian, fintech companies, and bank risk analytics teams.

## Methodology

### Data
Simulated dataset of 15,000 consumer loan applicants with features calibrated to published credit bureau distributions:

| Feature | Description | IV |
|---------|-------------|-----|
| FICO score | Credit bureau score | Strong |
| Debt-to-income (DTI) | Monthly obligations / gross income | Moderate |
| Revolving utilization | Credit card balances / limits | Moderate |
| # Derogatory marks | Collections, charge-offs, bankruptcies | Strong |
| Income | Annual gross income ($k) | Moderate |
| # Hard inquiries (6mo) | Recent credit applications | Moderate |
| Months employed | Employment stability | Weak-Moderate |
| Loan amount | Requested loan size | Weak |
| Homeowner flag | Property ownership | Weak |

**Default rate**: ~9% (calibrated to consumer lending norms)

### Weight of Evidence (WoE) & Information Value (IV)
WoE transforms features into log-odds units, enabling monotonic binning and interpretable coefficients:

```
WoE_i = ln(Distribution of Events_i / Distribution of Non-Events_i)
IV = Σ (Good_rate_i - Bad_rate_i) × WoE_i
```

IV benchmarks:
- IV < 0.02: Useless predictor
- 0.02–0.10: Weak predictor  
- 0.10–0.30: Medium predictor
- **> 0.30: Strong predictor**

### Logistic Regression
L2-regularized logistic regression (C=1.0) trained on standardized features, with 5-fold stratified cross-validation.

### Scorecard Scaling (PDO Method)
Standard FICO-style scaling:
- **Base Score**: 600 (corresponding to ~10% default odds)
- **PDO**: 20 points doubles the odds

```
Score = Offset + Factor × ln(Odds)
Factor = PDO / ln(2)
Offset = Base_Score - Factor × ln(Base_Odds)
```

## Model Performance

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Test AUC | **0.745** | Good (>0.70) |
| 5-Fold CV AUC | **0.724 ± 0.016** | Stable |
| KS Statistic | **0.362** | Good (>0.30) |
| Default rate (9% band) | 800+ score band: <2% | ✓ |
| Default rate (9% band) | <500 score band: >40% | ✓ |

### Scorecard Band Performance

| Score Band | Default Rate | Interpretation |
|------------|-------------|----------------|
| 800+ | ~1–2% | Prime / Super-Prime |
| 750–799 | ~3–5% | Near-Prime |
| 700–749 | ~6–9% | Standard |
| 640–699 | ~12–18% | Subprime |
| <580 | >30% | Deep Subprime |

## Dashboard Panels
1. ROC curve with AUC and cross-validated performance
2. Score distribution by default / non-default
3. Logistic regression coefficients (standardized)
4. WoE analysis: FICO score binning
5. WoE analysis: Debt-to-Income binning
6. Default rate by score band
7. Information Value comparison across features
8. Calibration curve (predicted vs. actual default rates)
9. Model summary statistics

## Outputs

| File | Description |
|------|-------------|
| `credit_risk_model.py` | Full modeling pipeline |
| `credit_risk_scorecard.png` | 9-panel performance dashboard |
| `credit_applicants.csv` | Simulated applicant dataset with scores |
| `scorecard_band_performance.csv` | Default rate by score band |

## Relevance to FICO / Fintech Roles
This project demonstrates the core analytical workflow used in scorecard development:
- **Feature assessment**: WoE binning and IV ranking
- **Model development**: Regularized logistic regression on credit features  
- **Scorecard scaling**: PDO-based transformation to interpretable scores
- **Performance validation**: AUC, KS, calibration, and band-level default rate analysis

---
*Built by Alex Strong | Economist & Quantitative Analyst | [GitHub Portfolio](https://github.com/alexstrong76/alex-strong-econ-analytics-portfolio)*
