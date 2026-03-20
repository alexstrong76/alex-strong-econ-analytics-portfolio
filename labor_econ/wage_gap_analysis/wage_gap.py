#!/usr/bin/env python
"""
Minimal wage gap example:
- Generate synthetic wage data
- Run an OLS regression with statsmodels
- Save wages_synthetic.csv for Tableau
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main():
    print("Running wage_gap.py ...")

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
            "gender": gender,
            "industry": industry,
        }
    )

    # Design matrix with dummy variables (all numeric)
    X_df = pd.get_dummies(
        df[["years_exp", "edu", "gender", "industry"]],
        drop_first=True,
    )

    # Add constant
    X_df = sm.add_constant(X_df)

    # DEBUG: Show dtypes and head so we know what we're feeding to OLS
    print("\nDesign matrix dtypes:")
    print(X_df.dtypes)
    print("\nFirst 5 rows of X_df:")
    print(X_df.head())
    print("\nTarget dtype:", df["wage"].dtype)

    # ✅ Force everything into plain NumPy float arrays
    X = X_df.to_numpy(dtype=float)
    y = df["wage"].to_numpy(dtype=float)

    print("\nShapes -> X:", X.shape, " y:", y.shape)

    # OLS regression
    model = sm.OLS(y, X).fit()
    print("\n=== OLS Results ===")
    print(model.summary())

    # Save data for Tableau
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wages_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved synthetic wage dataset to: {out_path.resolve()}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    main()

    os.makedirs("labor_econ/wage_gap_analysis/figures", exist_ok=True)

    df = pd.read_csv("data/processed/wages_synthetic.csv")
    wage_col = [c for c in df.columns if "wage" in c.lower() or
                "earn" in c.lower() or "salary" in c.lower() or
                "income" in c.lower() or "pay" in c.lower()][0]                if any("wage" in c.lower() or "earn" in c.lower() or
                      "salary" in c.lower() or "income" in c.lower() or
                      "pay" in c.lower() for c in df.columns)                else df.select_dtypes(include=[np.number]).columns[0]
    group_col = [c for c in df.columns if "gender" in c.lower() or
                 "group" in c.lower() or "female" in c.lower() or
                 "sex" in c.lower()]
    group_col = group_col[0] if group_col else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df[wage_col], bins=40, color="#1B4F8A",
                 edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Wage", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Wage Distribution\nAll workers", fontsize=11,
                      fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    if group_col:
        groups = df[group_col].unique()
        colors = ["#1B4F8A", "#E8593C", "#0F6E56", "#BA7517"]
        for i, g in enumerate(sorted(groups)):
            sub = df[df[group_col] == g][wage_col]
            axes[1].hist(sub, bins=30, alpha=0.6,
                         color=colors[i % len(colors)],
                         edgecolor="white", label=f"Group {g}")
        axes[1].legend(fontsize=9)
        axes[1].set_title("Wage Distribution by Group\nGap visualization",
                          fontsize=11, fontweight="bold")
    else:
        exp_col = [c for c in df.columns if "exp" in c.lower() or
                   "tenure" in c.lower() or "year" in c.lower()]
        exp_col = exp_col[0] if exp_col else df.select_dtypes(
            include=[np.number]).columns[1]
        axes[1].scatter(df[exp_col], df[wage_col],
                        alpha=0.3, color="#1B4F8A", s=10)
        axes[1].set_xlabel(exp_col, fontsize=10)
        axes[1].set_title(f"Wage vs {exp_col}", fontsize=11,
                          fontweight="bold")
    axes[1].set_xlabel("Wage", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "labor_econ/wage_gap_analysis/figures/wage_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
