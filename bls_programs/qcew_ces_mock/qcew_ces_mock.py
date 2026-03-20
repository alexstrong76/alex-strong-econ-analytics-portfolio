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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    main()

    os.makedirs("bls_programs/qcew_ces_mock/figures", exist_ok=True)

    df = pd.read_csv("data/processed/employment_qcew_mock.csv")
    ind_col = [c for c in df.columns if "industry" in c.lower() or
               "sector" in c.lower() or "naics" in c.lower() or
               "ind" in c.lower()]
    ind_col = ind_col[0] if ind_col else None
    emp_col = [c for c in df.columns if "emp" in c.lower() or
               "employ" in c.lower() or "worker" in c.lower()]
    emp_col = emp_col[0] if emp_col else               df.select_dtypes(include=[np.number]).columns[0]
    qtr_col = [c for c in df.columns if "qtr" in c.lower() or
               "quarter" in c.lower() or "period" in c.lower() or
               "date" in c.lower() or "year" in c.lower()]
    qtr_col = qtr_col[0] if qtr_col else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if ind_col:
        ind_totals = df.groupby(ind_col)[emp_col].mean().sort_values(
            ascending=True).tail(10)
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(ind_totals)))
        axes[0].barh(ind_totals.index.astype(str),
                     ind_totals.values, color=colors,
                     edgecolor="none", height=0.6)
        axes[0].set_xlabel("Average employment", fontsize=10)
        axes[0].set_title("Employment by Industry Sector\nTop 10 sectors",
                          fontsize=11, fontweight="bold")
        axes[0].grid(True, axis="x", alpha=0.3)
    else:
        axes[0].hist(df[emp_col], bins=30, color="#1B4F8A",
                     edgecolor="white")
        axes[0].set_xlabel("Employment level", fontsize=10)
        axes[0].set_ylabel("Count", fontsize=10)
        axes[0].set_title("Employment Distribution", fontsize=11,
                          fontweight="bold")
        axes[0].grid(True, alpha=0.3)

    if qtr_col and ind_col:
        pivot = df.groupby([qtr_col, ind_col])[emp_col].mean().unstack()
        top_inds = pivot.mean().nlargest(4).index
        for col in top_inds:
            axes[1].plot(pivot.index.astype(str), pivot[col],
                         marker="o", markersize=3, linewidth=1.5,
                         label=str(col))
        axes[1].set_xlabel("Period", fontsize=10)
        axes[1].set_ylabel("Employment", fontsize=10)
        axes[1].set_title("Employment Trends by Sector\nTop 4 sectors over time",
                          fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=8)
        tick_step = max(1, len(pivot) // 8)
        axes[1].set_xticks(range(0, len(pivot), tick_step))
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)
    elif qtr_col:
        trend = df.groupby(qtr_col)[emp_col].mean()
        axes[1].plot(trend.index.astype(str), trend.values,
                     color="#1B4F8A", linewidth=2, marker="o", markersize=3)
        axes[1].set_xlabel("Period", fontsize=10)
        axes[1].set_ylabel("Average employment", fontsize=10)
        axes[1].set_title("Employment Trend Over Time",
                          fontsize=11, fontweight="bold")
        tick_step = max(1, len(trend) // 8)
        axes[1].set_xticks(range(0, len(trend), tick_step))
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)
    else:
        wage_col = [c for c in df.columns if "wage" in c.lower() or
                    "pay" in c.lower() or "earn" in c.lower()]
        wage_col = wage_col[0] if wage_col else                    df.select_dtypes(include=[np.number]).columns[1]
        axes[1].scatter(df[emp_col], df[wage_col],
                        alpha=0.3, color="#1B4F8A", s=8)
        axes[1].set_xlabel("Employment", fontsize=10)
        axes[1].set_ylabel("Wage", fontsize=10)
        axes[1].set_title("Employment vs Wages",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "bls_programs/qcew_ces_mock/figures/employment_by_sector.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
