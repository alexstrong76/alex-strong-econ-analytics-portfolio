#!/usr/bin/env python
"""
diff_in_diff_policy_evaluation.py

Simulate a simple difference-in-differences setup:

- Units: firms, regions, or states
- Treatment turns on for treated group after a policy date
- Estimate DiD via a basic regression

Output:
    data/processed/did_policy_sim.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf


def main():
    print("Running diff_in_diff_policy_evaluation.py ...")

    rng = np.random.default_rng(2025)

    n_units = 100
    n_periods = 8
    units = [f"U{str(i).zfill(3)}" for i in range(n_units)]
    periods = np.arange(1, n_periods + 1)

    treated_units = set(rng.choice(units, size=n_units // 2, replace=False))
    policy_start = 5  # period when treatment begins

    rows = []
    tau_true = 3.0

    for u in units:
        alpha_u = rng.normal(0, 2)
        for t in periods:
            post = int(t >= policy_start)
            treat = int(u in treated_units)
            trend = 0.5 * t
            eps = rng.normal(0, 2)

            # Outcome, e.g., average weekly hours, earnings, etc.
            y = 20 + alpha_u + trend + tau_true * treat * post + eps

            rows.append(
                {"unit": u, "t": t, "treat": treat, "post": post, "y": y}
            )

    df = pd.DataFrame(rows)

    # DiD regression: y ~ treat + post + treat*post
    model = smf.ols("y ~ treat + post + treat:post", data=df).fit()
    print(model.summary())

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "did_policy_sim.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved DiD policy dataset to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
