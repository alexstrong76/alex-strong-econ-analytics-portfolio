#!/usr/bin/env python
"""
Generate synthetic wage data for wage gap analysis, run an OLS regression,
and save the dataset for use in Tableau dashboards.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main():
    # Reproducible RNG
    rng = np.random.default_rng(123)

    n = 2000

    # Core features
    years_exp = rng.integers(0, 30, size=n)
    edu = rng.choice([12, 14, 16, 18], size=n, p=[0.2, 0.3, 0.4, 0.1])
    gender = rng.integers(0, 2, size=n)  # 0 = female, 1 = male
    industry = rng.choice(["tech", "health", "retail", "public"], size=n)

    # Structural wage equation
    base = 15 + 1.2 * years_exp + 2.0 * (edu - 12) + 3.0 * gender

    # Industry premia
    industry_prem = {"tech": 7, "health": 4, "retail": 1, "public": 2}
    industry_effect = pd.Series(industry).map(industry_prem).to_numpy()

    # Noise
    noise = rng.normal(loc=0.0, scale=3.0, size=n)

    # Final wage
    wage = base + industry_effect + noise

    # Build DataFrame
    df = pd.DataFrame(
        {
            "wage": wage,
            "years_exp": years_exp,
            "edu": edu,
            "gender": gender,      # 0 = female, 1 = male
            "industry": industry,  # tech/health/retail/public
        }
    )

    # Design matrix with dummies (all numeric)
    X_df = pd.get_dummies(
        df[["years_exp", "edu", "gender", "industry"]],
        drop_first=True
    )
    X_df = sm.add_constant(X_df)

    # Debug: show dtypes
    print("Design matrix dtypes:\n", X_df.dtypes)
    print("Target dtype:", df["wage"].dtype, "\n")

    # ✅ Force everything to plain float numpy arrays
    X = X_df.to_numpy(dtype=float)
    y = df["wage"].to_numpy(dtype=float)

    # OLS regression
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Save data for Tableau dashboards
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wages_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved synthetic wage dataset to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
