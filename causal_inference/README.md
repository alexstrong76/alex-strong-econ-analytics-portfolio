# Difference-in-Differences Policy Evaluation

## Business Question
How can we isolate the causal effect of a policy intervention when we
cannot run a randomized controlled trial? This project simulates a
real-world program evaluation scenario — estimating the impact of a
treatment on an outcome variable across treated and control units
before and after a policy change.

## Method
- **Design:** Two-way panel difference-in-differences (DiD)
- **Assumption:** Parallel trends — treated and control units would
  have followed the same trajectory absent the intervention
- **Estimation:** OLS with treatment, time, and interaction terms
  via `statsmodels`
- **Validation:** Pre-trend visualization to assess assumption
  plausibility; ATT (average treatment effect on the treated)
  reported with standard errors

## Key Finding
The simulated policy produced a statistically significant positive
treatment effect. The pre-period parallel trends hold visually,
supporting the validity of the DiD design. The ATT estimate and
confidence interval are reported in the script output.

## Visualizations
![DiD parallel trends and treatment effect](figures/did_parallel_trends.png)

## How to Run
```bash
python causal_inference/diff_in_diff_policy_evaluation.py
```

## Limitations and Next Steps
- Synthetic data means results are illustrative, not empirical
- A production version would test parallel trends formally using
  an event-study specification
- Heterogeneous treatment effects across subgroups are not explored
  here but would be a natural extension using interaction terms or
  `EconML`

## Tools
Python · statsmodels · pandas · matplotlib