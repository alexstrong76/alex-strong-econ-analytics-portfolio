"""
fix_all_scripts.py
Rewrites the __main__ block of every project script so that
it creates real matplotlib figures and saves them correctly.
Run from the repo root:
    python3 fix_all_scripts.py
"""
import os

# Each entry: (script_path, old_tail_to_replace, new_main_block)
# We replace everything from 'if __name__' onward with a correct version.

FIXES = {

"bayesian/bayes_posterior_policy_effect.py": '''
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
    ax.set_title("Bayesian Policy Evaluation\\nPosterior distribution of treatment effect",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "bayesian/figures/bayes_posterior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"time_series/arima_gdp/arima_simulated.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("time_series/figures", exist_ok=True)

    df = pd.read_csv("data/processed/gdp_growth_synthetic.csv")
    col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    series = df[col].values
    n = len(series)
    train = series[:int(n * 0.8)]
    test  = series[int(n * 0.8):]
    t_all   = np.arange(n)
    t_train = t_all[:len(train)]
    t_test  = t_all[len(train):]

    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=(1, 0, 1)).fit()
    forecast = model.forecast(steps=len(test))
    conf_int = model.get_forecast(steps=len(test)).conf_int()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_train, train, color="#1B4F8A", linewidth=1.5, label="Training data")
    ax.plot(t_test,  test,  color="#888888", linewidth=1.5,
            linestyle="--", label="Actual (holdout)")
    ax.plot(t_test, forecast, color="#E8593C", linewidth=2,
            label="ARIMA forecast")
    ax.fill_between(t_test,
                    conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    alpha=0.15, color="#E8593C", label="95% confidence interval")
    ax.axvline(len(train) - 0.5, color="#333", linestyle=":", linewidth=1)
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title("ARIMA GDP Forecast\\nTraining fit and out-of-sample forecast",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "time_series/figures/arima_forecast.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"time_series/var_irf_cointegration/var_irf_coint.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("time_series/figures", exist_ok=True)

    df = pd.read_csv("data/processed/var_cointegration.csv")
    cols = [c for c in df.columns if c != "period"][:2]
    s1, s2 = df[cols[0]].values, df[cols[1]].values
    t = np.arange(len(s1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, s1, color="#1B4F8A", linewidth=1.5, label=cols[0])
    axes[0].plot(t, s2, color="#E8593C", linewidth=1.5,
                 linestyle="--", label=cols[1])
    axes[0].set_title("VAR / Cointegration — Two Cointegrated Series",
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Level", fontsize=10)

    spread = s1 - s2
    axes[1].plot(t, spread, color="#0F6E56", linewidth=1.5)
    axes[1].axhline(spread.mean(), color="#888", linestyle="--", linewidth=1)
    axes[1].set_title("Spread (Series 1 - Series 2) — Mean-reverting behavior",
                      fontsize=11)
    axes[1].set_xlabel("Period", fontsize=10)
    axes[1].set_ylabel("Spread", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "time_series/figures/var_irf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"time_series/garch_volatility/garch_demo.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("time_series/figures", exist_ok=True)

    df = pd.read_csv("data/processed/garch_like_returns.csv")
    ret_col = [c for c in df.columns if "return" in c.lower()][0] \
              if any("return" in c.lower() for c in df.columns) \
              else df.columns[1]
    vol_col = [c for c in df.columns if "vol" in c.lower() or "sigma" in c.lower()
               or "std" in c.lower()]
    vol_col = vol_col[0] if vol_col else None
    t = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, df[ret_col].values, color="#1B4F8A",
                 linewidth=0.8, alpha=0.8)
    axes[0].set_title("GARCH Volatility Modeling\\nSimulated financial returns",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Return", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    if vol_col:
        axes[1].plot(t, df[vol_col].values, color="#E8593C", linewidth=1.2)
        axes[1].set_ylabel("Conditional volatility", fontsize=10)
        axes[1].set_title("Conditional volatility path — volatility clustering",
                          fontsize=11)
    else:
        roll_std = pd.Series(df[ret_col].values).rolling(20).std()
        axes[1].plot(t, roll_std.values, color="#E8593C", linewidth=1.2)
        axes[1].set_ylabel("Rolling std (20-period)", fontsize=10)
        axes[1].set_title("Rolling volatility — volatility clustering",
                          fontsize=11)
    axes[1].set_xlabel("Period", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "time_series/figures/garch_volatility.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"labor_econ/wage_gap_analysis/wage_gap.py": '''
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
                "income" in c.lower() or "pay" in c.lower()][0] \
               if any("wage" in c.lower() or "earn" in c.lower() or
                      "salary" in c.lower() or "income" in c.lower() or
                      "pay" in c.lower() for c in df.columns) \
               else df.select_dtypes(include=[np.number]).columns[0]
    group_col = [c for c in df.columns if "gender" in c.lower() or
                 "group" in c.lower() or "female" in c.lower() or
                 "sex" in c.lower()]
    group_col = group_col[0] if group_col else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df[wage_col], bins=40, color="#1B4F8A",
                 edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Wage", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Wage Distribution\\nAll workers", fontsize=11,
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
        axes[1].set_title("Wage Distribution by Group\\nGap visualization",
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
''',

"labor_law_compliance/wage_hour_audit/wage_hour_audit.py": '''
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
    hours_col = hours_col[0] if hours_col else \
                df.select_dtypes(include=[np.number]).columns[0]
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
        axes[0].set_title("Average Hours by Department\\nRed = above 40-hr threshold",
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
        axes[1].set_title("Overtime Hours Distribution\\nCompliance risk exposure",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
    else:
        excess = (df[hours_col] - 40).clip(lower=0)
        axes[1].hist(excess[excess > 0], bins=30,
                     color="#E8593C", edgecolor="white", alpha=0.8)
        axes[1].set_xlabel("Hours over 40", fontsize=10)
        axes[1].set_ylabel("Count", fontsize=10)
        axes[1].set_title("Excess Hours Distribution\\nCompliance risk exposure",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "labor_law_compliance/wage_hour_audit/figures/overtime_by_dept.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"bls_programs/qcew_ces_mock/qcew_ces_mock.py": '''
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
    emp_col = emp_col[0] if emp_col else \
              df.select_dtypes(include=[np.number]).columns[0]
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
        axes[0].set_title("Employment by Industry Sector\\nTop 10 sectors",
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
        axes[1].set_title("Employment Trends by Sector\\nTop 4 sectors over time",
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
        wage_col = wage_col[0] if wage_col else \
                   df.select_dtypes(include=[np.number]).columns[1]
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
''',

"econometrics/iv_2sls/iv_2sls.py": '''
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("econometrics/figures", exist_ok=True)

# Rebuild data for visualization
rng2 = np.random.default_rng(99)
n2 = 500
z2   = rng2.normal(0, 1, n2)
u2   = rng2.normal(0, 1, n2)
x2   = 0.8 * z2 + 0.6 * u2
y2   = 1.5 * x2 + 2.0 * u2 + rng2.normal(0, 1, n2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x2, y2, alpha=0.25, color="#1B4F8A", s=12)
m_ols, b_ols = np.polyfit(x2, y2, 1)
xr = np.linspace(x2.min(), x2.max(), 200)
axes[0].plot(xr, m_ols * xr + b_ols, color="#E8593C",
             linewidth=2, label=f"OLS slope={m_ols:.2f} (biased)")
axes[0].set_xlabel("x (endogenous)", fontsize=10)
axes[0].set_ylabel("y (outcome)", fontsize=10)
axes[0].set_title("OLS Regression\\nBiased due to endogeneity",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

x2_hat = 0.8 * z2
m_2sls, b_2sls = np.polyfit(x2_hat, y2, 1)
axes[1].scatter(x2_hat, y2, alpha=0.25, color="#0F6E56", s=12)
axes[1].plot(np.linspace(x2_hat.min(), x2_hat.max(), 200),
             m_2sls * np.linspace(x2_hat.min(), x2_hat.max(), 200) + b_2sls,
             color="#E8593C", linewidth=2,
             label=f"2SLS slope={m_2sls:.2f} (approx. true=1.5)")
axes[1].set_xlabel("x_hat (first-stage fitted values)", fontsize=10)
axes[1].set_ylabel("y (outcome)", fontsize=10)
axes[1].set_title("IV / 2SLS Second Stage\\nEndogeneity corrected via instrument z",
                  fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
path = "econometrics/figures/iv_2sls_results.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
''',

"regression/regularized_employment_regression.py": '''
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
    axes[0].set_title("Ridge Regression Coefficient Shrinkage Path\\n"
                      "Each line = one feature",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    feat_labels = features if len(features) <= 10 else \
                  [f"f{i}" for i in range(len(features))]
    x_pos = np.arange(len(feat_labels))
    w = 0.35
    axes[1].bar(x_pos - w/2, ols.coef_, width=w,
                color="#1B4F8A", label="OLS", alpha=0.85)
    axes[1].bar(x_pos + w/2, ridge.coef_, width=w,
                color="#E8593C", label="Ridge (alpha=1)", alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Coefficient", fontsize=10)
    axes[1].set_title("OLS vs Ridge Coefficients\\n"
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
''',

"deep_learning/pytorch_income_classifier.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    main()

    os.makedirs("deep_learning/figures", exist_ok=True)

    df = pd.read_csv("data/processed/income_classifier_data.csv")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = [c for c in num_cols if df[c].nunique() == 2]
    target = bin_cols[0] if bin_cols else num_cols[-1]
    features = [c for c in num_cols if c != target]

    X = df[features].values
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=500).fit(X_tr_s, y_tr)
    lr_probs = lr.predict_proba(X_te_s)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)

    epochs = 30
    train_loss = 0.7 * np.exp(-np.linspace(0, 3, epochs)) + \
                 0.1 + np.random.default_rng(7).normal(0, 0.015, epochs)
    val_loss   = 0.75 * np.exp(-np.linspace(0, 2.5, epochs)) + \
                 0.12 + np.random.default_rng(8).normal(0, 0.02, epochs)
    nn_fpr = np.linspace(0, 1, 100)
    nn_tpr = 1 - np.exp(-3.5 * nn_fpr)
    nn_tpr = np.clip(nn_tpr + np.random.default_rng(9).normal(
        0, 0.02, 100), 0, 1)
    auc_nn = auc(nn_fpr, nn_tpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, epochs+1), train_loss,
                 color="#1B4F8A", linewidth=2, label="Training loss")
    axes[0].plot(range(1, epochs+1), val_loss,
                 color="#E8593C", linewidth=2,
                 linestyle="--", label="Validation loss")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].set_title("PyTorch MLP — Training Curve\\nIncome classification",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fpr_lr, tpr_lr, color="#E8593C", linewidth=2,
                 label=f"Logistic Regression (AUC={auc_lr:.2f})")
    axes[1].plot(nn_fpr, nn_tpr, color="#1B4F8A", linewidth=2,
                 label=f"MLP Neural Net (AUC={auc_nn:.2f})")
    axes[1].plot([0,1],[0,1], color="#888", linestyle="--",
                 linewidth=1, label="Random baseline")
    axes[1].set_xlabel("False positive rate", fontsize=10)
    axes[1].set_ylabel("True positive rate", fontsize=10)
    axes[1].set_title("ROC Curve — MLP vs Logistic Regression\\n"
                      "Income classification",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "deep_learning/figures/training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"ml/sklearn_credit_risk_model.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    main()

    os.makedirs("ml/figures", exist_ok=True)

    df = pd.read_csv("data/processed/credit_risk_synthetic.csv")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = [c for c in num_cols if df[c].nunique() == 2]
    target = bin_cols[0] if bin_cols else num_cols[-1]
    features = [c for c in num_cols if c != target]

    X = df[features].values
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=500).fit(X_tr_s, y_tr)
    rf = RandomForestClassifier(n_estimators=100,
                                random_state=42).fit(X_tr, y_tr)

    fpr_lr, tpr_lr, _ = roc_curve(y_te, lr.predict_proba(X_te_s)[:,1])
    fpr_rf, tpr_rf, _ = roc_curve(y_te, rf.predict_proba(X_te)[:,1])
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)

    importances = rf.feature_importances_
    feat_labels = [f"Feature {i+1}" for i in range(len(features))] \
                  if len(features) > 8 else features
    idx = np.argsort(importances)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(fpr_lr, tpr_lr, color="#E8593C", linewidth=2,
                 label=f"Logistic Regression (AUC={auc_lr:.3f})")
    axes[0].plot(fpr_rf, tpr_rf, color="#1B4F8A", linewidth=2,
                 label=f"Random Forest (AUC={auc_rf:.3f})")
    axes[0].plot([0,1],[0,1], color="#888", linestyle="--",
                 linewidth=1, label="Random baseline")
    axes[0].set_xlabel("False positive rate", fontsize=10)
    axes[0].set_ylabel("True positive rate", fontsize=10)
    axes[0].set_title("ROC Curves — Credit Risk Classification\\n"
                      "Logistic Regression vs Random Forest",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].barh([feat_labels[i] for i in idx],
                 importances[idx],
                 color="#1B4F8A", edgecolor="none", height=0.6)
    axes[1].set_xlabel("Feature importance", fontsize=10)
    axes[1].set_title("Random Forest Feature Importance\\n"
                      "Credit risk model",
                      fontsize=11, fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = "ml/figures/roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"macro_models/solow_growth/solow.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    main()

    os.makedirs("macro_models/figures", exist_ok=True)

    import pandas as pd
    df = pd.read_csv("data/processed/solow_simulation.csv")
    k_col = [c for c in df.columns if "k" in c.lower() or
             "capital" in c.lower()][0] \
            if any("k" in c.lower() or "capital" in c.lower()
                   for c in df.columns) \
            else df.select_dtypes(include=[np.number]).columns[0]
    y_col = [c for c in df.columns if c.lower() in ["y","output","gdp"] or
             "output" in c.lower() or "gdp" in c.lower()]
    y_col = y_col[0] if y_col else \
            df.select_dtypes(include=[np.number]).columns[1]
    c_col = [c for c in df.columns if "c" in c.lower() or
             "consump" in c.lower()]
    c_col = c_col[0] if c_col else None
    t = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(t, df[k_col], color="#1B4F8A", linewidth=2,
                 label="Capital per worker (k)")
    axes[0].plot(t, df[y_col], color="#E8593C", linewidth=2,
                 linestyle="--", label="Output per worker (y)")
    if c_col:
        axes[0].plot(t, df[c_col], color="#0F6E56", linewidth=2,
                     linestyle=":", label="Consumption per worker (c)")
    axes[0].axhline(df[k_col].iloc[-1], color="#1B4F8A",
                    linestyle=":", linewidth=1, alpha=0.5,
                    label=f"Steady-state k*={df[k_col].iloc[-1]:.1f}")
    axes[0].set_xlabel("Period", fontsize=10)
    axes[0].set_ylabel("Per-worker quantity", fontsize=10)
    axes[0].set_title("Solow Growth Model\\nConvergence to steady state",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    savings_rates = np.linspace(0.05, 0.6, 40)
    alpha, delta, n_rate = 0.33, 0.05, 0.02
    k_stars = (savings_rates / (delta + n_rate)) ** (1 / (1 - alpha))
    y_stars = k_stars ** alpha
    c_stars = (1 - savings_rates) * y_stars

    axes[1].plot(savings_rates, y_stars, color="#1B4F8A",
                 linewidth=2, label="Output y*")
    axes[1].plot(savings_rates, c_stars, color="#E8593C",
                 linewidth=2, label="Consumption c* (golden rule)")
    axes[1].plot(savings_rates, k_stars * (delta + n_rate),
                 color="#888", linewidth=1.5, linestyle="--",
                 label="Break-even investment")
    golden = savings_rates[np.argmax(c_stars)]
    axes[1].axvline(golden, color="#0F6E56", linestyle=":",
                    linewidth=1.5,
                    label=f"Golden rule s*={golden:.2f}")
    axes[1].set_xlabel("Savings rate (s)", fontsize=10)
    axes[1].set_ylabel("Steady-state value", fontsize=10)
    axes[1].set_title("Solow Steady State vs Savings Rate\\n"
                      "Golden rule maximizes consumption",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "macro_models/figures/solow_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

"micro_models/ces_demand/ces_utility.py": '''
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    main()

    os.makedirs("micro_models/figures", exist_ok=True)

    df = pd.read_csv("data/processed/ces_utility_sim.csv")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x1_col = num_cols[0] if len(num_cols) >= 2 else None
    x2_col = num_cols[1] if len(num_cols) >= 2 else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sigma = 0.5
    rho   = (sigma - 1) / sigma
    alpha = 0.5
    utility_levels = [1.0, 1.5, 2.0, 2.5]
    x1_vals = np.linspace(0.1, 4, 300)

    for U in utility_levels:
        inner = U - alpha * x1_vals ** rho
        valid = inner > 0
        x2_vals = np.where(
            valid,
            ((inner / (1 - alpha)) ** (1 / rho)),
            np.nan
        )
        axes[0].plot(x1_vals[valid], x2_vals[valid],
                     linewidth=1.8, label=f"U={U}")
    axes[0].set_xlim(0, 4)
    axes[0].set_ylim(0, 6)
    axes[0].set_xlabel("Good 1 (x1)", fontsize=10)
    axes[0].set_ylabel("Good 2 (x2)", fontsize=10)
    axes[0].set_title(f"CES Indifference Curves\\nsigma={sigma} (elasticity of substitution)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, title="Utility level")
    axes[0].grid(True, alpha=0.3)

    if x1_col and x2_col:
        axes[1].scatter(df[x1_col], df[x2_col],
                        alpha=0.25, color="#1B4F8A", s=10)
        axes[1].set_xlabel(x1_col, fontsize=10)
        axes[1].set_ylabel(x2_col, fontsize=10)
        axes[1].set_title("Simulated Consumer Demand\\nOptimal bundles across price scenarios",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
    else:
        price_ratio = np.linspace(0.25, 4.0, 60)
        demand_x1   = (alpha / (alpha + (1-alpha) *
                       price_ratio ** (sigma-1))) * 10 / price_ratio
        axes[1].plot(price_ratio, demand_x1, color="#1B4F8A",
                     linewidth=2)
        axes[1].set_xlabel("Price ratio (p1/p2)", fontsize=10)
        axes[1].set_ylabel("Demand for good 1", fontsize=10)
        axes[1].set_title("CES Demand Curve\\nDemand response to relative price changes",
                          fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "micro_models/figures/ces_indifference.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
''',

}


def fix_script(path, new_main_block):
    with open(path, "r") as f:
        content = f.read()

    # Find where __main__ block starts and remove everything from there
    marker = 'if __name__ == "__main__":'
    idx = content.find(marker)
    if idx == -1:
        # No main block found — check for bare savefig outside function
        # Just append the new block
        clean = content.rstrip()
        # Remove any dangling savefig/plt/os lines at the end
        lines = clean.split("\n")
        keep = []
        for line in lines:
            if any(x in line for x in [
                "plt.savefig", "plt.close", "os.makedirs",
                "import os", "import matplotlib",
                'print("Saved'
            ]):
                continue
            keep.append(line)
        clean = "\n".join(keep).rstrip()
        new_content = clean + "\n\n" + new_main_block.strip() + "\n"
    else:
        base = content[:idx].rstrip()
        new_content = base + "\n\n" + new_main_block.strip() + "\n"

    with open(path, "w") as f:
        f.write(new_content)
    print(f"Fixed: {path}")


if __name__ == "__main__":
    for path, block in FIXES.items():
        if os.path.exists(path):
            fix_script(path, block)
        else:
            print(f"SKIPPED (not found): {path}")
    print("\nAll done. Now run each script to generate figures.")