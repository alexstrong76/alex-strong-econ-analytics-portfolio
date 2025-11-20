#!/usr/bin/env python
"""
garch_demo.py

Simulate a simple GARCH(1,1)-style process for returns and volatility.

Output:
    data/processed/garch_like_returns.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running garch_demo.py ...")

    rng = np.random.default_rng(2025)

    T = 1000
    omega = 0.0005
    alpha = 0.05
    beta = 0.9

    returns = np.zeros(T)
    var = np.zeros(T)
    var[0] = 0.0004

    for t in range(1, T):
        eps = rng.normal()
        returns[t] = np.sqrt(var[t - 1]) * eps
        var[t] = omega + alpha * returns[t] ** 2 + beta * var[t - 1]

    approx_vol = np.sqrt(var)

    df = pd.DataFrame(
        {
            "t": np.arange(T),
            "returns": returns,
            "approx_vol": approx_vol,
        }
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "garch_like_returns.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved GARCH-like returns to: {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()
