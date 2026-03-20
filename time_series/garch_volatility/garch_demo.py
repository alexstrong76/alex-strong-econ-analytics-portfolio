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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("time_series/figures", exist_ok=True)

    df = pd.read_csv("data/processed/garch_like_returns.csv")
    ret_col = [c for c in df.columns if "return" in c.lower()][0]               if any("return" in c.lower() for c in df.columns)               else df.columns[1]
    vol_col = [c for c in df.columns if "vol" in c.lower() or "sigma" in c.lower()
               or "std" in c.lower()]
    vol_col = vol_col[0] if vol_col else None
    t = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, df[ret_col].values, color="#1B4F8A",
                 linewidth=0.8, alpha=0.8)
    axes[0].set_title("GARCH Volatility Modeling\nSimulated financial returns",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Return", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    if vol_col:
        axes[1].plot(t, df[vol_col].values, color="#E8593C", linewidth=1.2)
        axes[1].set_ylabel("Conditional volatility", fontsize=10)
        axes[1].set_title("Conditional volatility path — volatility clustering",
                          fontsize=11)
    else:
        roll_std = pd.Series(df[ret_col].values).rolling(20).std()
        axes[1].plot(t, roll_std.values, color="#E8593C", linewidth=1.2)
        axes[1].set_ylabel("Rolling std (20-period)", fontsize=10)
        axes[1].set_title("Rolling volatility — volatility clustering",
                          fontsize=11)
    axes[1].set_xlabel("Period", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "time_series/figures/garch_volatility.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
