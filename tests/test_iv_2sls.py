
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

def simulate_iv(n=1000, seed=123):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    v = 0.5*rng.normal(size=n) + 0.4*u
    x = 0.8*z + v
    y = 1.5 + 2.0*x + u
    return pd.DataFrame({"y":y, "x":x, "z":z})

def test_iv_vs_ols_bias():
    df = simulate_iv()
    ols = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    iv = IV2SLS.from_formula("y ~ 1 + [x ~ z]", data=df).fit()
    assert abs(ols.params["x"] - 2.0) > abs(iv.params["x"] - 2.0)
