import numpy as np, pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

np.random.seed(1)
n = 2000
z = np.random.randn(n)               # instrument
u = np.random.randn(n)               # structural error
v = 0.5*np.random.randn(n) + 0.4*u   # endog error (correlated with u)
x = 0.8*z + v                        # endogenous regressor
y = 1.5 + 2.0*x + u

df = pd.DataFrame({"y":y, "x":x, "z":z})
X = sm.add_constant(df["x"])
ols = sm.OLS(df["y"], X).fit()
print("=== OLS (biased due to endogeneity) ===")
print(ols.summary())

iv = IV2SLS.from_formula("y ~ 1 + [x ~ z]", data=df).fit()
print("\n=== 2SLS (IV) ===")
print(iv.summary)

# First-stage relevancy (F-stat roughly)
fs = sm.OLS(df["x"], sm.add_constant(df["z"])).fit()
print("\nFirst-stage F-stat (x ~ z):", fs.fvalue)
