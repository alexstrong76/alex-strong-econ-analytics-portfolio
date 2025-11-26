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
    main()
