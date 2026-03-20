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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler

    main()

    os.makedirs("regression/figures", exist_ok=True)

    df = pd.read_csv("data/processed/regularized_employment_regression.csv")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = num_cols[0]
    features = num_cols[1:] if len(num_cols) > 1 else num_cols

    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    alphas = np.logspace(-3, 4, 60)
    coefs  = []
    for a in alphas:
        r = Ridge(alpha=a).fit(X_s, y)
        coefs.append(r.coef_)
    coefs = np.array(coefs)

    ols   = LinearRegression().fit(X_s, y)
    ridge = Ridge(alpha=1.0).fit(X_s, y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for j in range(coefs.shape[1]):
        axes[0].plot(np.log10(alphas), coefs[:, j],
                     linewidth=1.2, alpha=0.8)
    axes[0].axvline(0, color="#333", linestyle="--",
                    linewidth=1, label="alpha=1 (log10=0)")
    axes[0].set_xlabel("log10(alpha) — regularization strength", fontsize=10)
    axes[0].set_ylabel("Coefficient value", fontsize=10)
    axes[0].set_title("Ridge Regression Coefficient Shrinkage Path\n"
                      "Each line = one feature",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    feat_labels = features if len(features) <= 10 else                   [f"f{i}" for i in range(len(features))]
    x_pos = np.arange(len(feat_labels))
    w = 0.35
    axes[1].bar(x_pos - w/2, ols.coef_, width=w,
                color="#1B4F8A", label="OLS", alpha=0.85)
    axes[1].bar(x_pos + w/2, ridge.coef_, width=w,
                color="#E8593C", label="Ridge (alpha=1)", alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Coefficient", fontsize=10)
    axes[1].set_title("OLS vs Ridge Coefficients\n"
                      "Ridge shrinks toward zero",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = "regression/figures/ridge_coef_path.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
