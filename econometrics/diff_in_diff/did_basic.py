import numpy as np, pandas as pd
import statsmodels.formula.api as smf

np.random.seed(3)
n = 1000
treat = np.random.binomial(1, 0.5, n)
post = np.random.binomial(1, 0.5, n)
tau = 2.0
eps = np.random.randn(n)
y = 5 + 1.0*treat + 1.0*post + tau*(treat*post) + eps

df = pd.DataFrame({"y":y,"treat":treat,"post":post})
mod = smf.ols("y ~ treat + post + treat:post", data=df).fit(cov_type="HC1")
print(mod.summary())
print("\nDID estimate (treat×post):", mod.params["treat:post"])
