#!/usr/bin/env python
"""
var_irf_coint.py

Simulate two cointegrated time series suitable for VAR / IRF demos.

Output:
    data/processed/var_cointegration.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running var_irf_coint.py ...")

    rng = np.random.default_rng(2025)

    T = 200
    e1 = rng.normal(scale=1.0, size=T)
    e2 = rng.normal(scale=0.5, size=T)

    y1 = np.cumsum(e1)
    y2 = y1 + e2  # y2 tracks y1 plus stationary noise

    df = pd.DataFrame(
        {
            "t": np.arange(T),
            "y1": y1,
            "y2": y2,
        }
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "var_cointegration.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved VAR/cointegration data to: {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()
