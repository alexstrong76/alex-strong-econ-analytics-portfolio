#!/usr/bin/env python
"""
regularized_employment_regression.py

Simulate an employment-related outcome (e.g., log wages) as a function of
age, tenure, education, and region. Fit OLS and Ridge regression models.

Output:
    data/processed/regularized_employment_regression.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge


def main():
    print("Running regularized_employment_regression.py ...")

    rng = np.random.default_rng(2025)
    n = 1500

    age = rng.integers(22, 65, size=n)
    tenure = np.clip(age - rng.integers(18, 30, size=n), 0, None)
    edu_years = rng.choice([12, 14, 16, 18], size=n, p=[0.25, 0.3, 0.3, 0.15])
    regions = rng.choice(["Northeast", "Midwest", "South", "West"], size=n)

    # Underlying wage equation (log wages)
    beta_age = 0.01
    beta_tenure = 0.02
    beta_edu = 0.07
    region_effect = {"Northeast": 0.08, "Midwest": 0.02, "South": -0.03, "West": 0.05}

    mu = (
        2.5
        + beta_age * age
        + beta_tenure * tenure
        + beta_edu * (edu_years - 12)
        + pd.Series(regions).map(region_effect).to_numpy()
    )

    eps = rng.normal(0, 0.2, size=n)
    log_wage = mu + eps

    df = pd.DataFrame(
        {
            "age": age,
            "tenure": tenure,
            "edu_years": edu_years,
            "region": regions,
            "log_wage": log_wage,
        }
    )

    # Design matrix with dummies
    X = pd.get_dummies(df[["age", "tenure", "edu_years", "region"]], drop_first=True)
    y = df["log_wage"].to_numpy()

    ols = LinearRegression()
    ols.fit(X, y)

    ridge = Ridge(alpha=1.0, random_state=2025)
    ridge.fit(X, y)

    print("OLS R^2:", ols.score(X, y))
    print("Ridge R^2:", ridge.score(X, y))

    # Save data
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regularized_employment_regression.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved regression dataset to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
