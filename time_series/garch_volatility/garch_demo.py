import numpy as np, pandas as pd
from arch import arch_model

np.random.seed(8)
T = 1000
eps = np.random.randn(T)
# Volatility clustering via GARCH structure
am = arch_model(None, vol="Garch", p=1, o=0, q=1, dist="normal")
# Simulate
sim = am.simulate([0.0, 0.05, 0.1, 0.85], T)  # [mu, omega, alpha, beta]
r = sim["data"]
m = arch_model(r, mean="Constant", vol="Garch", p=1, q=1).fit(disp="off")
print(m.summary())
