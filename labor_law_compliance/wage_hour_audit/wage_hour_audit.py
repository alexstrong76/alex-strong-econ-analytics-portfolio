#!/usr/bin/env python
"""
wage_hour_audit.py

Generate synthetic timesheet data including:
- Hours worked
- Overtime hours
- Department
- Rounding increment
- Gross pay

Output:
    data/processed/timesheets_synthetic.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running wage_hour_audit.py ...")

    rng = np.random.default_rng(2025)

    n_employees = 150
    n_weeks = 26
    depts = ["Operations", "Sales", "HR", "Finance", "Warehouse"]
    rounding_choices = [1, 6, 15]

    rows = []
    emp_ids = [f"E{str(i).zfill(3)}" for i in range(1, n_employees + 1)]

    for emp in emp_ids:
        dept = rng.choice(depts)
        base_rate = rng.uniform(15, 45)

        for week in range(1, n_weeks + 1):
            base_hours = rng.normal(loc=40, scale=5)
            base_hours = max(20, min(70, base_hours))

            rounding_to = rng.choice(rounding_choices, p=[0.5, 0.3, 0.2])
            hours_rounded = np.round(base_hours / rounding_to) * rounding_to

            ot_hours = max(0, hours_rounded - 40)
            long_week_flag = int(hours_rounded > 50)

            gross_pay = hours_rounded * base_rate

            rows.append(
                {
                    "employee_id": emp,
                    "week": week,
                    "dept": dept,
                    "hourly_rate": round(base_rate, 2),
                    "hours": round(hours_rounded, 2),
                    "ot_hours": round(ot_hours, 2),
                    "rounded_to": rounding_to,
                    "long_week_flag": long_week_flag,
                    "gross": round(gross_pay, 2),
                }
            )

    df = pd.DataFrame(rows)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timesheets_synthetic.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved timesheet data to: {out_path.resolve()}")
    print(df.head())

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    main()

    os.makedirs("labor_law_compliance/wage_hour_audit/figures", exist_ok=True)

    df = pd.read_csv("data/processed/timesheets_synthetic.csv")
    dept_col = [c for c in df.columns if "dept" in c.lower() or
                "department" in c.lower() or "division" in c.lower()]
    dept_col = dept_col[0] if dept_col else None
    hours_col = [c for c in df.columns if "hour" in c.lower() or
                 "hrs" in c.lower()]
    hours_col = hours_col[0] if hours_col else                 df.select_dtypes(include=[np.number]).columns[0]
    ot_col = [c for c in df.columns if "over" in c.lower() or
              "ot" in c.lower()]
    ot_col = ot_col[0] if ot_col else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if dept_col:
        dept_means = df.groupby(dept_col)[hours_col].mean().sort_values(
            ascending=True)
        colors = ["#E8593C" if v > 40 else "#1B4F8A"
                  for v in dept_means.values]
        axes[0].barh(dept_means.index, dept_means.values,
                     color=colors, edgecolor="none", height=0.6)
        axes[0].axvline(40, color="#333", linestyle="--",
                        linewidth=1, label="40-hr threshold")
        axes[0].set_xlabel("Average hours worked", fontsize=10)
        axes[0].set_title("Average Hours by Department\nRed = above 40-hr threshold",
                          fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, axis="x", alpha=0.3)
    else:
        axes[0].hist(df[hours_col], bins=30, color="#1B4F8A",
                     edgecolor="white")
        axes[0].axvline(40, color="#E8593C", linestyle="--",
                        linewidth=1.5, label="40-hr threshold")
        axes[0].set_xlabel("Hours worked", fontsize=10)
        axes[0].set_ylabel("Count", fontsize=10)
        axes[0].set_title("Hours Distribution", fontsize=11,
                          fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

    if ot_col:
        ot_vals = df[ot_col]
        axes[1].hist(ot_vals, bins=30, color="#E8593C",
                     edgecolor="white", alpha=0.8)
        axes[1].set_xlabel("Overtime hours", fontsize=10)
        axes[1].set_ylabel("Count", fontsize=10)
        axes[1].set_title("Overtime Hours Distribution\nCompliance risk exposure",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
    else:
        excess = (df[hours_col] - 40).clip(lower=0)
        axes[1].hist(excess[excess > 0], bins=30,
                     color="#E8593C", edgecolor="white", alpha=0.8)
        axes[1].set_xlabel("Hours over 40", fontsize=10)
        axes[1].set_ylabel("Count", fontsize=10)
        axes[1].set_title("Excess Hours Distribution\nCompliance risk exposure",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "labor_law_compliance/wage_hour_audit/figures/overtime_by_dept.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
