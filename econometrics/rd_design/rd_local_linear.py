import numpy as np, pandas as pd
import statsmodels.api as sm

np.random.seed(4)
n = 2000
x = np.random.uniform(-2, 2, n)
tau = 1.5
y0 = 2 + 0.8*x + np.random.randn(n)*0.5
y1 = y0 + tau
y = np.where(x >= 0, y1, y0)

df = pd.DataFrame({"y":y, "x":x, "t":(x>=0).astype(int)})

# Local linear around cutoff (|x| <= h)
h = 0.5
local = df[np.abs(df["x"]) <= h].copy()
X = sm.add_constant(np.c_[local["x"], local["t"], local["x"]*local["t"]])
res = sm.OLS(local["y"], X).fit(cov_type="HC1")
print(res.summary())
print("\nLocal RD (t coefficient):", res.params[2])


