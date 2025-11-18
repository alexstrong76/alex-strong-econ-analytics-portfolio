
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

def test_breusch_pagan_detects_heteroskedasticity():
    rng = np.random.default_rng(11)
    n = 1500
    x1 = rng.normal(size=n)
    x2 = 0.9*x1 + rng.normal(scale=0.2, size=n)
    sigma = np.exp(0.3*x1)
    eps = rng.normal(scale=sigma, size=n)
    y = 2 + 1.5*x1 + 0.2*x2 + eps
    X = sm.add_constant(np.c_[x1, x2])
    ols = sm.OLS(y, X).fit()
    lm, pval, *_ = het_breuschpagan(ols.resid, X)
    assert pval < 0.05
