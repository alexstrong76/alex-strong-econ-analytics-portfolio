# Healthcare Program ROI Evaluation
**Difference-in-Differences Analysis | Value-Based Care Intervention Study**

## Overview
A quasi-experimental evaluation of a proactive primary care management program using Difference-in-Differences (DiD) methodology. The study estimates the causal effect of care management enrollment on emergency department (ED) utilization and hospitalization rates, then translates those estimates into an annualized ROI framework.

This project demonstrates applied econometrics in a healthcare context — directly relevant to Decision Sciences, health economics, and program evaluation roles.

## Research Design

**Setting**: Oak Street Health-style value-based primary care model  
**Intervention**: Proactive care management program for high-risk Medicare-eligible patients  
**Design**: Difference-in-Differences with patient-level panel data  
**Sample**: 1,600 patients (800 treatment, 800 control) across 8 quarters  

| Group | Description |
|-------|-------------|
| Treatment | High-risk patients enrolled in care management program |
| Control | Similar patients on waitlist (not yet enrolled) |

**Pre-period**: Q1–Q4 2021 (4 quarters baseline)  
**Post-period**: Q1–Q4 2022 (4 quarters post-intervention)

## Methods

### Difference-in-Differences
The DiD estimator identifies the Average Treatment Effect on the Treated (ATT):

```
ATT = (Ȳ_treat,post - Ȳ_treat,pre) - (Ȳ_ctrl,post - Ȳ_ctrl,pre)
```

**Key Assumption**: Parallel pre-period trends between treatment and control groups. The analysis includes a formal parallel trends test (pre-period trend comparison by group).

### Regression-Based DiD
Augments the simple 2×2 estimator with patient-level controls:

```
ED_visits = β₀ + β₁(Treated) + β₂(Post) + β₃(Treated×Post) + β₄(Age) + β₅(Chronic Conditions) + β₆(Baseline Risk) + ε
```

β₃ is the DiD estimator (ATT).

### Inference
95% confidence intervals computed via nonparametric bootstrap (n=500 iterations).

## Key Results

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| DiD ATT (ED visits/quarter) | −0.216 | [−0.266, −0.162] |
| DiD ATT (hospitalization rate) | Significant reduction | p < 0.001 |
| Parallel trends test | ✓ Supported | |

## ROI Analysis

| Component | Value | Assumption |
|-----------|-------|------------|
| ED visits saved (annual/pt) | ~0.86 | 4 quarters × ATT |
| ED cost savings | Per published literature | $2,200/visit |
| Hospitalization savings | Per published literature | $15,000/admission |
| Program cost | $3,200/patient/year | Illustrative |
| **Net Annual Savings** | **$1.4M** | N=800 enrolled |
| **ROI** | **53%** | Net savings / program cost |

## Outputs

| File | Description |
|------|-------------|
| `healthcare_did_analysis.py` | Main analysis script |
| `healthcare_roi_analysis.png` | 6-panel dashboard |
| `patient_panel_data.csv` | Simulated patient-level panel dataset |
| `descriptive_stats.csv` | Pre-period baseline comparability table |

## Dashboard Panels
1. Pre-period parallel trends check
2. Full event study (pre + post by group)
3. DiD coefficient with bootstrap confidence interval
4. Bootstrap distribution of ATT estimates
5. ROI waterfall chart
6. Baseline risk score distribution (comparability check)
7. Results summary table

## Note on Data
All patient data is simulated using distributions calibrated to published value-based care literature. No real patient data is used. Results are for portfolio demonstration purposes.

---
*Built by Alex Strong | Economist & Quantitative Analyst | [GitHub Portfolio](https://github.com/alexstrong76/alex-strong-econ-analytics-portfolio)*
