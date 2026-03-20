# Macroeconomic Growth Modeling — Solow Model

## Business Question
How do savings rates, population growth, and technology determine
long-run living standards across economies? The Solow growth model
provides a foundational framework for understanding why some
countries are richer than others and what policy levers affect
the steady-state level of output per worker.

## Method
- **Model:** Solow-Swan neoclassical growth model with
  Cobb-Douglas production function
- **Simulation:** Discrete-time capital accumulation path from
  an initial capital stock to steady-state
- **Sensitivity analysis:** Compares steady-state outcomes across
  different savings rates and depreciation rates
- **Output:** Capital, output, and consumption paths over time;
  steady-state values reported analytically and confirmed
  numerically

## Key Finding
Higher savings rates increase the steady-state capital stock and
output per worker but do not affect the long-run growth rate,
consistent with Solow model predictions. The golden rule savings
rate that maximizes steady-state consumption is identified
numerically.

## Visualizations
![Capital and output convergence to steady state](figures/solow_convergence.png)
![Sensitivity — steady state vs savings rate](figures/solow_sensitivity.png)

## How to Run
```bash
python macro_models/solow_growth/solow.py
```

## Limitations and Next Steps
- The basic Solow model abstracts from human capital, technology
  diffusion, and institutional quality
- Extending to the augmented Solow model (Mankiw, Romer, Weil
  1992) with human capital would improve empirical fit
- Calibrating the model to real Penn World Tables data would
  enable cross-country comparison

## Tools
Python · NumPy · matplotlib · pandas