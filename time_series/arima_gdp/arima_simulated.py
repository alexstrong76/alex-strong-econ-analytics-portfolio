#!/usr/bin/env python
"""
arima_simulated.py

Simulate GDP growth as an AR(1) process and build a GDP index series.

Output:
    data/processed/gdp_growth_synthetic.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    print("Running arima_simulated.py ...")

    rng = np.random.default_rng(2025)
    T = 160  # e.g., 40 years of quarterly data

    phi = 0.5
    mu = 0.005
    sigma = 0.01

    eps = rng.normal(0, sigma, size=T)
    g = np.zeros(T)
    g[0] = mu + eps[0]

    for t in range(1, T):
        g[t] = mu + phi * g[t - 1] + eps[t]

    gdp_index = 100 * np.exp(np.cumsum(g))

    dates = pd.period_range("1985Q1", periods=T, freq="Q")

    df = pd.DataFrame(
        {
            "period": dates.astype(str),
            "g": g,
            "gdp_index": gdp_index,
        }
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gdp_growth_synthetic.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved GDP growth data to: {out_path.resolve()}")
    print(df.head())

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
    ax.set_title("ARIMA GDP Forecast\nTraining fit and out-of-sample forecast",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "time_series/figures/arima_forecast.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
