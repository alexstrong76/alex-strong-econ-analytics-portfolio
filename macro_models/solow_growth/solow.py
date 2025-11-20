#!/usr/bin/env python
"""
solow.py

Simulate a basic Solow growth model with:
- Capital per worker
- Output per worker
- Consumption per worker

Output:
    data/processed/solow_simulation.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running solow.py ...")

    T = 100
    alpha = 0.33
    s = 0.2
    delta = 0.05
    n = 0.01
    g = 0.02

    k = np.zeros(T)
    y = np.zeros(T)
    c = np.zeros(T)

    k[0] = 5.0

    for t in range(T):
        y[t] = k[t] ** alpha
        c[t] = (1 - s) * y[t]
        if t < T - 1:
            k[t + 1] = (1 - delta) * k[t] + s * y[t]

    df = pd.DataFrame(
        {
            "period": np.arange(T),
            "k": k,
            "y": y,
            "c": c,
        }
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solow_simulation.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved Solow simulation to: {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()
