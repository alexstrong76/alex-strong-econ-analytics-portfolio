import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(6)
n = 1200
x1 = np.random.randn(n)
x2 = 0.9*x1 + np.random.randn(n)*0.2  # multicollinearity
sigma = np.exp(0.3*x1)                # heteroskedasticity
eps = np.random.randn(n)*sigma
y = 2 + 1.5*x1 + 0.2*x2 + eps

X = sm.add_constant(np.c_[x1,x2])
ols = sm.OLS(y, X).fit()
print("=== OLS (naive) ===")
print(ols.summary())

print("\n=== OLS (HC1 robust) ===")
print(ols.get_robustcov_results(cov_type="HC1").summary())

# BP test
bp = het_breuschpagan(ols.resid, X)
print("\nBreusch-Pagan (LM stat, p-value, f-stat, p-value):", bp)

# VIF
vifs = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
print("VIF for x1,x2:", vifs)
