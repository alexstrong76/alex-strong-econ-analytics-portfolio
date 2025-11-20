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
    main()
