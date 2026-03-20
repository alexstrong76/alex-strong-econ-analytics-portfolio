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

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("econometrics/figures", exist_ok=True)

# Rebuild data for visualization
rng2 = np.random.default_rng(99)
n2 = 500
z2   = rng2.normal(0, 1, n2)
u2   = rng2.normal(0, 1, n2)
x2   = 0.8 * z2 + 0.6 * u2
y2   = 1.5 * x2 + 2.0 * u2 + rng2.normal(0, 1, n2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x2, y2, alpha=0.25, color="#1B4F8A", s=12)
m_ols, b_ols = np.polyfit(x2, y2, 1)
xr = np.linspace(x2.min(), x2.max(), 200)
axes[0].plot(xr, m_ols * xr + b_ols, color="#E8593C",
             linewidth=2, label=f"OLS slope={m_ols:.2f} (biased)")
axes[0].set_xlabel("x (endogenous)", fontsize=10)
axes[0].set_ylabel("y (outcome)", fontsize=10)
axes[0].set_title("OLS Regression\nBiased due to endogeneity",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

x2_hat = 0.8 * z2
m_2sls, b_2sls = np.polyfit(x2_hat, y2, 1)
axes[1].scatter(x2_hat, y2, alpha=0.25, color="#0F6E56", s=12)
axes[1].plot(np.linspace(x2_hat.min(), x2_hat.max(), 200),
             m_2sls * np.linspace(x2_hat.min(), x2_hat.max(), 200) + b_2sls,
             color="#E8593C", linewidth=2,
             label=f"2SLS slope={m_2sls:.2f} (approx. true=1.5)")
axes[1].set_xlabel("x_hat (first-stage fitted values)", fontsize=10)
axes[1].set_ylabel("y (outcome)", fontsize=10)
axes[1].set_title("IV / 2SLS Second Stage\nEndogeneity corrected via instrument z",
                  fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
path = "econometrics/figures/iv_2sls_results.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
