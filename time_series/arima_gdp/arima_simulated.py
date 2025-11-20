#!/usr/bin/env python
"""
arima_simulated.py

Simulate GDP growth as an AR(1) process and build a GDP index series.

Output:
    data/processed/gdp_growth_synthetic.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running arima_simulated.py ...")

    rng = np.random.default_rng(2025)
    T = 160  # e.g., 40 years of quarterly data

    phi = 0.5
    mu = 0.005
    sigma = 0.01

    eps = rng.normal(0, sigma, size=T)
    g = np.zeros(T)
    g[0] = mu + eps[0]

    for t in range(1, T):
        g[t] = mu + phi * g[t - 1] + eps[t]

    gdp_index = 100 * np.exp(np.cumsum(g))

    dates = pd.period_range("1985Q1", periods=T, freq="Q")

    df = pd.DataFrame(
        {
            "period": dates.astype(str),
            "g": g,
            "gdp_index": gdp_index,
        }
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gdp_growth_synthetic.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved GDP growth data to: {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()
