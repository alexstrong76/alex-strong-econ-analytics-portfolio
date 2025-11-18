import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit

np.random.seed(0)
n = 2000
x1 = np.random.randn(n)
x2 = np.random.binomial(1, 0.4, n)
linpred = -0.2 + 1.0*x1 + 0.8*x2
p = 1/(1+np.exp(-linpred))
y = np.random.binomial(1, p)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
X = sm.add_constant(df[["x1","x2"]])

# LPM
lpm = sm.OLS(df["y"], X).fit(cov_type="HC1")
print("=== LPM (HC1 robust) ===")
print(lpm.summary())

# Logit
logit = Logit(df["y"], X).fit(disp=False)
print("\n=== Logit ===")
print(logit.summary())
print("\nMarginal effects (dy/dx at mean):")
print(logit.get_margeff(at="mean").summary())

# Probit
probit = Probit(df["y"], X).fit(disp=False)
print("\n=== Probit ===")
print(probit.summary())
print("\nMarginal effects (dy/dx at mean):")
print(probit.get_margeff(at="mean").summary())
