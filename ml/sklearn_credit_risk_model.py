#!/usr/bin/env python
"""
sklearn_credit_risk_model.py

Simulate a small credit risk dataset and fit scikit-learn classification models.

Output:
    data/processed/credit_risk_synthetic.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    print("Running sklearn_credit_risk_model.py ...")

    rng = np.random.default_rng(2025)
    n = 3000

    income = rng.lognormal(mean=10.5, sigma=0.5, size=n)
    utilization = rng.uniform(0, 1, size=n)
    age = rng.integers(21, 75, size=n)

    # Latent probability of default
    logit = (
        -3.0
        + 2.5 * utilization
        - 0.00001 * income
        - 0.02 * (age - 40)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p_default, size=n)

    df = pd.DataFrame(
        {
            "income": income,
            "utilization": utilization,
            "age": age,
            "default": default,
        }
    )

    X = df[["income", "utilization", "age"]].to_numpy()
    y = df["default"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2025, stratify=y
    )

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    proba_logreg = logreg.predict_proba(X_test)[:, 1]
    auc_logreg = roc_auc_score(y_test, proba_logreg)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=2025
    )
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, proba_rf)

    print("Logistic Regression AUC:", round(auc_logreg, 3))
    print("Random Forest AUC:", round(auc_rf, 3))
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf.predict(X_test)))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "credit_risk_synthetic.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved credit risk dataset to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
