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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("time_series/figures", exist_ok=True)

    df = pd.read_csv("data/processed/var_cointegration.csv")
    cols = [c for c in df.columns if c != "period"][:2]
    s1, s2 = df[cols[0]].values, df[cols[1]].values
    t = np.arange(len(s1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, s1, color="#1B4F8A", linewidth=1.5, label=cols[0])
    axes[0].plot(t, s2, color="#E8593C", linewidth=1.5,
                 linestyle="--", label=cols[1])
    axes[0].set_title("VAR / Cointegration — Two Cointegrated Series",
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Level", fontsize=10)

    spread = s1 - s2
    axes[1].plot(t, spread, color="#0F6E56", linewidth=1.5)
    axes[1].axhline(spread.mean(), color="#888", linestyle="--", linewidth=1)
    axes[1].set_title("Spread (Series 1 - Series 2) — Mean-reverting behavior",
                      fontsize=11)
    axes[1].set_xlabel("Period", fontsize=10)
    axes[1].set_ylabel("Spread", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "time_series/figures/var_irf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
