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
    feat_labels = [f"Feature {i+1}" for i in range(len(features))]                   if len(features) > 8 else features
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
    axes[0].set_title("ROC Curves — Credit Risk Classification\n"
                      "Logistic Regression vs Random Forest",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].barh([feat_labels[i] for i in idx],
                 importances[idx],
                 color="#1B4F8A", edgecolor="none", height=0.6)
    axes[1].set_xlabel("Feature importance", fontsize=10)
    axes[1].set_title("Random Forest Feature Importance\n"
                      "Credit risk model",
                      fontsize=11, fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = "ml/figures/roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
