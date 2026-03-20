#!/usr/bin/env python
"""
bayes_posterior_policy_effect.py

Bayesian inference for a simple policy effect using a normal-normal model.

- Generate synthetic data from a treated and control group
- Compute classical difference-in-means estimate
- Compute Bayesian posterior for the treatment effect
- Save dataset for use in visualizations

Output:
    data/processed/bayes_policy_effect.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running bayes_posterior_policy_effect.py ...")

    rng = np.random.default_rng(2025)

    # Sample sizes
    n_treat = 400
    n_control = 400

    # True parameters
    mu_control = 70.0
    tau_true = 5.0     # true treatment effect
    sigma = 10.0       # common standard deviation

    # Generate outcomes
    y_control = rng.normal(mu_control, sigma, size=n_control)
    y_treat = rng.normal(mu_control + tau_true, sigma, size=n_treat)

    df = pd.DataFrame(
        {
            "group": ["control"] * n_control + ["treat"] * n_treat,
            "y": np.concatenate([y_control, y_treat]),
        }
    )

    # Classical diff-in-means
    mean_c = df.loc[df["group"] == "control", "y"].mean()
    mean_t = df.loc[df["group"] == "treat", "y"].mean()
    tau_hat = mean_t - mean_c

    print(f"Classical diff-in-means estimate: {tau_hat:.2f}")

    # Bayesian normal-normal with known variance
    # Prior: tau ~ N(mu0, s0^2)
    mu0 = 0.0
    s0 = 10.0

    # Likelihood: tau_hat ~ N(tau, s^2), with s^2 = 2 * sigma^2 / n  (approx)
    s_lik = np.sqrt(2 * sigma**2 / min(n_treat, n_control))

    # Posterior parameters:
    post_var = 1.0 / (1.0 / s0**2 + 1.0 / s_lik**2)
    post_mean = post_var * (mu0 / s0**2 + tau_hat / s_lik**2)
    post_sd = np.sqrt(post_var)

    print(f"Posterior mean (tau): {post_mean:.2f}")
    print(f"Posterior sd (tau): {post_sd:.2f}")

    # Save data for visualization / analysis
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bayes_policy_effect.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved Bayesian policy dataset to: {out_path.resolve()}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    main()

    os.makedirs("bayesian/figures", exist_ok=True)

    # Reconstruct posterior for plotting
    rng = np.random.default_rng(42)
    n_control, n_treated = 50, 50
    mu_control = rng.normal(5.0, 1.5, n_control)
    mu_treated = rng.normal(7.5, 1.5, n_treated)
    prior_mean, prior_var = 0.0, 10.0
    obs_var = 1.5 ** 2
    n = n_treated
    obs_mean = mu_treated.mean() - mu_control.mean()
    post_var  = 1.0 / (1.0 / prior_var + n / obs_var)
    post_mean = post_var * (prior_mean / prior_var + n * obs_mean / obs_var)
    post_sd   = np.sqrt(post_var)

    x = np.linspace(post_mean - 5*post_sd, post_mean + 5*post_sd, 400)
    from scipy.stats import norm
    prior_pdf    = norm.pdf(x, prior_mean, np.sqrt(prior_var))
    posterior_pdf = norm.pdf(x, post_mean, post_sd)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, prior_pdf, color="#888888", linewidth=1.5,
            linestyle="--", label="Prior")
    ax.plot(x, posterior_pdf, color="#1B4F8A", linewidth=2.5,
            label=f"Posterior (mean={post_mean:.2f}, sd={post_sd:.2f})")
    ax.axvline(post_mean, color="#1B4F8A", linestyle=":", linewidth=1.2)
    lo, hi = norm.ppf(0.025, post_mean, post_sd), norm.ppf(0.975, post_mean, post_sd)
    ax.fill_between(x, posterior_pdf,
                    where=(x >= lo) & (x <= hi),
                    alpha=0.15, color="#1B4F8A",
                    label=f"95% credible interval [{lo:.2f}, {hi:.2f}]")
    ax.axvline(0, color="#E8593C", linewidth=1, linestyle="-.",
               label="No effect (0)")
    ax.set_xlabel("Treatment effect (tau)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Bayesian Policy Evaluation\nPosterior distribution of treatment effect",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "bayesian/figures/bayes_posterior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
