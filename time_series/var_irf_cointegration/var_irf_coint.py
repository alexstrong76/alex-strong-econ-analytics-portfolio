import numpy as np, pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint

np.random.seed(7)
T = 300
e1, e2 = np.random.randn(T), np.random.randn(T)
y1 = np.cumsum(e1) + 0.5*np.random.randn(T)
y2 = y1 + e2*0.5  # cointegrated with y1

df = pd.DataFrame({"y1":y1, "y2":y2})

# Cointegration test
score, pvalue, _ = coint(df["y1"], df["y2"])
print("Engle–Granger test p-value:", pvalue)

# VAR on differences (stationary)
d = df.diff().dropna()
model = VAR(d).fit(2)
print(model.summary())

irf = model.irf(10)
print("\nIRF (first 3 steps, y1 shock on y2):")
print(irf.irfs[:3,1,0])
