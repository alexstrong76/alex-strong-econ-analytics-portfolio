
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def simulate_did(n=2000, tau=2.0, seed=7):
    rng = np.random.default_rng(seed)
    treat = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    eps = rng.normal(size=n)
    y = 5 + 1.0*treat + 1.0*post + tau*(treat*post) + eps
    return pd.DataFrame({"y":y, "treat":treat, "post":post}), tau

def test_did_estimator_close_to_true():
    df, tau = simulate_did()
    mod = smf.ols("y ~ treat + post + treat:post", data=df).fit(cov_type="HC1")
    est = mod.params["treat:post"]
    assert abs(est - tau) < 0.3
