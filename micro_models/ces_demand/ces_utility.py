#!/usr/bin/env python
"""
ces_utility.py

Simulate CES utility over two goods for a set of consumers.
Not strictly used in the storyboards, but good micro / applied math content.

Output:
    data/processed/ces_utility_sim.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def ces_utility(x, y, sigma=0.5):
    # CES with elasticity sigma and rho = (sigma-1)/sigma
    rho = (sigma - 1.0) / sigma
    return (0.5 * (x ** rho) + 0.5 * (y ** rho)) ** (1.0 / rho)


def main():
    print("Running ces_utility.py ...")

    rng = np.random.default_rng(2025)
    n = 1000

    income = rng.lognormal(mean=3.5, sigma=0.5, size=n)
    px = rng.uniform(1, 5, size=n)
    py = rng.uniform(1, 5, size=n)

    alpha = 0.5
    x = alpha * income / px
    y = (1 - alpha) * income / py

    u = ces_utility(x, y, sigma=0.8)

    df = pd.DataFrame(
        {
            "income": income,
            "px": px,
            "py": py,
            "x": x,
            "y": y,
            "utility": u,
        }
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ces_utility_sim.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved CES utility simulation to: {out_path.resolve()}")
    print(df.head())

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("micro_models/figures", exist_ok=True)

    df = pd.read_csv("data/processed/ces_utility_sim.csv")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x1_col = num_cols[0] if len(num_cols) >= 2 else None
    x2_col = num_cols[1] if len(num_cols) >= 2 else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sigma = 0.5
    rho   = (sigma - 1) / sigma
    alpha = 0.5
    utility_levels = [1.0, 1.5, 2.0, 2.5]
    x1_vals = np.linspace(0.1, 4, 300)

    for U in utility_levels:
        inner = U - alpha * x1_vals ** rho
        valid = inner > 0
        x2_vals = np.where(
            valid,
            ((inner / (1 - alpha)) ** (1 / rho)),
            np.nan
        )
        axes[0].plot(x1_vals[valid], x2_vals[valid],
                     linewidth=1.8, label=f"U={U}")
    axes[0].set_xlim(0, 4)
    axes[0].set_ylim(0, 6)
    axes[0].set_xlabel("Good 1 (x1)", fontsize=10)
    axes[0].set_ylabel("Good 2 (x2)", fontsize=10)
    axes[0].set_title(f"CES Indifference Curves\nsigma={sigma} (elasticity of substitution)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, title="Utility level")
    axes[0].grid(True, alpha=0.3)

    if x1_col and x2_col:
        axes[1].scatter(df[x1_col], df[x2_col],
                        alpha=0.25, color="#1B4F8A", s=10)
        axes[1].set_xlabel(x1_col, fontsize=10)
        axes[1].set_ylabel(x2_col, fontsize=10)
        axes[1].set_title("Simulated Consumer Demand\nOptimal bundles across price scenarios",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
    else:
        price_ratio = np.linspace(0.25, 4.0, 60)
        demand_x1   = (alpha / (alpha + (1-alpha) *
                       price_ratio ** (sigma-1))) * 10 / price_ratio
        axes[1].plot(price_ratio, demand_x1, color="#1B4F8A",
                     linewidth=2)
        axes[1].set_xlabel("Price ratio (p1/p2)", fontsize=10)
        axes[1].set_ylabel("Demand for good 1", fontsize=10)
        axes[1].set_title("CES Demand Curve\nDemand response to relative price changes",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "micro_models/figures/ces_indifference.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
