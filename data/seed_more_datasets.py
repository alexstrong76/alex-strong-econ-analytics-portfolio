
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2025)

def matched_psm():
    n = 1500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    ps = 1/(1+np.exp(-(0.5*x1 + 0.8*x2)))
    t = rng.binomial(1, ps)
    tau = 1.5
    eps = rng.normal(scale=0.5, size=n)
    y = 3 + 1.0*x1 + 0.5*x2 + tau*t + eps
    pd.DataFrame({"y":y,"t":t,"x1":x1,"x2":x2,"ps":ps}).to_csv(OUT / "matched_psm_source.csv", index=False)

def panel_sim():
    N, T = 120, 8
    alpha_i = rng.normal(0, 1, N)
    rows = []
    for i in range(N):
        for tt in range(T):
            x = rng.normal()
            u = rng.normal()
            y = 1.0 + 1.5*x + alpha_i[i] + u
            rows.append((i, tt, x, y))
    pd.DataFrame(rows, columns=["unit","time","x","y"]).to_csv(OUT / "panel_sim.csv", index=False)

def did_sim():
    n = 2000
    treat = rng.binomial(1, 0.5, n)
    post = rng.binomial(1, 0.5, n)
    eps = rng.normal(size=n)
    tau = 2.0
    y = 5 + 1.0*treat + 1.0*post + tau*(treat*post) + eps
    pd.DataFrame({"y":y,"treat":treat,"post":post}).to_csv(OUT / "did_sim.csv", index=False)

def rd_sim():
    n = 3000
    x = rng.uniform(-2, 2, n)
    tau = 1.5
    y0 = 2 + 0.8*x + rng.normal(scale=0.5, size=n)
    y1 = y0 + tau
    y = np.where(x >= 0, y1, y0)
    pd.DataFrame({"x":x,"y":y,"t":(x>=0).astype(int)}).to_csv(OUT / "rd_sim.csv", index=False)

def var_data():
    T = 200
    e1, e2 = rng.normal(size=T), rng.normal(size=T)
    y1 = np.cumsum(e1) + 0.5*rng.normal(size=T)
    y2 = y1 + 0.5*e2
    pd.DataFrame({"y1":y1,"y2":y2}).to_csv(OUT / "var_cointegration.csv", index=False)

def garch_returns():
    T = 1000
    vol = np.abs(rng.normal(size=T))*0.8 + 0.2
    r = rng.normal(scale=vol)
    pd.DataFrame({"returns": r, "approx_vol": vol}).to_csv(OUT / "garch_like_returns.csv", index=False)

if __name__ == "__main__":
    matched_psm()
    panel_sim()
    did_sim()
    rd_sim()
    var_data()
    garch_returns()
    print(f"Wrote CSVs to: {OUT.resolve()}")
