#!/usr/bin/env python
"""
qcew_ces_mock.py

Generate a synthetic QCEW/CES-style employment and wage dataset
by industry and quarter for use in dashboards and time-series demos.

Output:
    data/processed/employment_qcew_mock.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running qcew_ces_mock.py ...")

    rng = np.random.default_rng(2025)

    industries = ["manufacturing", "services", "retail", "public"]
    quarters = pd.period_range("2015Q1", periods=40, freq="Q")

    rows = []
    for ind in industries:
        base_emp = rng.integers(10_000, 50_000)
        emp_level = base_emp
        avg_wage = rng.uniform(18, 45)

        for q in quarters:
            growth = rng.normal(loc=0.003, scale=0.02)
            emp_level = max(1000, emp_level * (1 + growth))
            wage_shock = rng.normal(scale=0.5)
            avg_wage_t = max(10, avg_wage + wage_shock)

            total_wages = emp_level * avg_wage_t * 13  # approx. quarter

            rows.append(
                {
                    "quarter": str(q),
                    "industry": ind,
                    "employment": round(emp_level),
                    "avg_hourly_wage": round(avg_wage_t, 2),
                    "total_wages": round(total_wages, 2),
                }
            )

    df = pd.DataFrame(rows)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "employment_qcew_mock.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved employment data to: {out_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()
