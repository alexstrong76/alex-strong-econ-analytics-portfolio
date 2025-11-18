import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

np.random.seed(2)
N, T = 200, 8
alpha_i = np.random.normal(0, 1, N)   # unit effects
x = np.random.randn(N*T)
u = np.random.randn(N*T)
df = pd.DataFrame({
    "unit": np.repeat(np.arange(N), T),
    "time": np.tile(np.arange(T), N),
    "x": x,
})
df["alpha"] = np.repeat(alpha_i, T)
df["y"] = 1.0 + 1.5*df["x"] + df["alpha"] + u

df = df.set_index(["unit","time"])
exog = sm.add_constant(df[["x"]])
# FE
fe = PanelOLS(df["y"], exog, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
print("=== Fixed Effects ===")
print(fe.summary)

# RE
re = RandomEffects(df["y"], exog).fit()
print("\n=== Random Effects ===")
print(re.summary)

# Hausman (manual small utility)
b_fe = fe.params
b_re = re.params
cov_diff = fe.cov - re.cov
diff = (b_fe - b_re).dropna()
stat = float(diff.T @ np.linalg.pinv(cov_diff.loc[diff.index, diff.index]) @ diff)
print("\nHausman stat (approx):", stat)
