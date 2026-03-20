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

import os
os.makedirs("macro_models/figures", exist_ok=True)

plt.savefig("macro_models/figures/solow_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: macro_models/figures/solow_convergence.png")

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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    main()

    os.makedirs("macro_models/figures", exist_ok=True)

    import pandas as pd
    df = pd.read_csv("data/processed/solow_simulation.csv")
    k_col = [c for c in df.columns if "k" in c.lower() or
             "capital" in c.lower()][0]             if any("k" in c.lower() or "capital" in c.lower()
                   for c in df.columns)             else df.select_dtypes(include=[np.number]).columns[0]
    y_col = [c for c in df.columns if c.lower() in ["y","output","gdp"] or
             "output" in c.lower() or "gdp" in c.lower()]
    y_col = y_col[0] if y_col else             df.select_dtypes(include=[np.number]).columns[1]
    c_col = [c for c in df.columns if "c" in c.lower() or
             "consump" in c.lower()]
    c_col = c_col[0] if c_col else None
    t = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(t, df[k_col], color="#1B4F8A", linewidth=2,
                 label="Capital per worker (k)")
    axes[0].plot(t, df[y_col], color="#E8593C", linewidth=2,
                 linestyle="--", label="Output per worker (y)")
    if c_col:
        axes[0].plot(t, df[c_col], color="#0F6E56", linewidth=2,
                     linestyle=":", label="Consumption per worker (c)")
    axes[0].axhline(df[k_col].iloc[-1], color="#1B4F8A",
                    linestyle=":", linewidth=1, alpha=0.5,
                    label=f"Steady-state k*={df[k_col].iloc[-1]:.1f}")
    axes[0].set_xlabel("Period", fontsize=10)
    axes[0].set_ylabel("Per-worker quantity", fontsize=10)
    axes[0].set_title("Solow Growth Model\nConvergence to steady state",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    savings_rates = np.linspace(0.05, 0.6, 40)
    alpha, delta, n_rate = 0.33, 0.05, 0.02
    k_stars = (savings_rates / (delta + n_rate)) ** (1 / (1 - alpha))
    y_stars = k_stars ** alpha
    c_stars = (1 - savings_rates) * y_stars

    axes[1].plot(savings_rates, y_stars, color="#1B4F8A",
                 linewidth=2, label="Output y*")
    axes[1].plot(savings_rates, c_stars, color="#E8593C",
                 linewidth=2, label="Consumption c* (golden rule)")
    axes[1].plot(savings_rates, k_stars * (delta + n_rate),
                 color="#888", linewidth=1.5, linestyle="--",
                 label="Break-even investment")
    golden = savings_rates[np.argmax(c_stars)]
    axes[1].axvline(golden, color="#0F6E56", linestyle=":",
                    linewidth=1.5,
                    label=f"Golden rule s*={golden:.2f}")
    axes[1].set_xlabel("Savings rate (s)", fontsize=10)
    axes[1].set_ylabel("Steady-state value", fontsize=10)
    axes[1].set_title("Solow Steady State vs Savings Rate\n"
                      "Golden rule maximizes consumption",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "macro_models/figures/solow_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
