#!/usr/bin/env python
"""
diff_in_diff_policy_evaluation.py
Simulate a difference-in-differences setup and produce diagnostic plots:
  1. Pre/post outcome trends by treatment group (parallel trends visual)
  2. DiD coefficient plot with confidence interval
Output:
    data/processed/did_policy_sim.csv
    causal_inference/figures/did_figure_1.png
    causal_inference/figures/did_figure_2.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path
import statsmodels.formula.api as smf


def main():
    print("Running diff_in_diff_policy_evaluation.py ...")

    rng = np.random.default_rng(2025)
    n_units   = 100
    n_periods = 8
    units     = [f"U{str(i).zfill(3)}" for i in range(n_units)]
    periods   = np.arange(1, n_periods + 1)
    treated_units = set(rng.choice(units, size=n_units // 2, replace=False))
    policy_start  = 5
    tau_true      = 3.0

    rows = []
    for u in units:
        alpha_u = rng.normal(0, 2)
        for t in periods:
            post  = int(t >= policy_start)
            treat = int(u in treated_units)
            trend = 0.5 * t
            eps   = rng.normal(0, 2)
            y     = 20 + alpha_u + trend + tau_true * treat * post + eps
            rows.append({"unit": u, "t": t, "treat": treat, "post": post, "y": y})

    df = pd.DataFrame(rows)

    model = smf.ols("y ~ treat + post + treat:post", data=df).fit()
    print(model.summary())

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "did_policy_sim.csv", index=False)
    print(f"Saved DiD dataset to: {(out_dir / 'did_policy_sim.csv').resolve()}")

    os.makedirs("causal_inference/figures", exist_ok=True)

    # ── FIGURE 1: Parallel trends — mean outcome by group and period ──
    means = (df.groupby(["t", "treat"])["y"]
               .mean()
               .reset_index()
               .rename(columns={"y": "mean_y"}))

    treated_means = means[means["treat"] == 1].sort_values("t")
    control_means = means[means["treat"] == 0].sort_values("t")

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(treated_means["t"], treated_means["mean_y"],
             color="#1B4F8A", marker="o", linewidth=2, label="Treated group")
    ax1.plot(control_means["t"], control_means["mean_y"],
             color="#E8593C", marker="s", linewidth=2, linestyle="--",
             label="Control group")
    ax1.axvline(x=policy_start - 0.5, color="#888", linestyle=":",
                linewidth=1.5, label=f"Policy start (t={policy_start})")
    ax1.set_xlabel("Period", fontsize=11)
    ax1.set_ylabel("Mean outcome (y)", fontsize=11)
    ax1.set_title("Difference-in-Differences: Parallel Trends\nMean outcome by group over time",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = "causal_inference/figures/did_figure_1.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1}")
    plt.close(fig1)

    # ── FIGURE 2: DiD coefficient plot ────────────────────────────────
    params   = model.params
    conf     = model.conf_int()
    labels   = ["Intercept", "Treat", "Post", "Treat x Post (ATT)"]
    coef_names = params.index.tolist()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#1B4F8A" if "treat:post" in n else "#888888" for n in coef_names]
    for idx, (name, coef) in enumerate(params.items()):
        lo = conf.loc[name, 0]
        hi = conf.loc[name, 1]
        ax2.barh(idx, coef, color=colors[idx], height=0.5, alpha=0.85)
        ax2.plot([lo, hi], [idx, idx], color="#333", linewidth=2)
        ax2.plot([lo, lo], [idx - 0.15, idx + 0.15], color="#333", linewidth=1.5)
        ax2.plot([hi, hi], [idx - 0.15, idx + 0.15], color="#333", linewidth=1.5)

    ax2.set_yticks(range(len(coef_names)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.axvline(0, color="#333", linewidth=0.8)
    ax2.set_xlabel("Coefficient estimate", fontsize=10)
    ax2.set_title("DiD Regression Coefficients with 95% Confidence Intervals\n"
                  f"True ATT = {tau_true:.1f}  |  Estimated ATT = {params['treat:post']:.2f}",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path2 = "causal_inference/figures/did_figure_2.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()