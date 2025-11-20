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
    main()
