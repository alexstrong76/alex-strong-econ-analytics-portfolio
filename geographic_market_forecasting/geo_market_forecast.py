import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────
df = pd.read_csv("data/raw/bls_metro_employment.csv")
print(f"Loaded {len(df)} rows across {df['metro'].nunique()} metros")
print("\nMetros in dataset:")
print(df.groupby("metro")["year"].count().to_string())

all_metros = sorted(df["metro"].unique())

# ── 2. FEATURE ENGINEERING ───────────────────────────────
def build_features(data, metro_list):
    rows = []
    for metro in metro_list:
        sub = (data[data["metro"] == metro]
               .sort_values("year")
               .reset_index(drop=True))
        if len(sub) < 3:
            print(f"  Skipping {metro} — only {len(sub)} years of data")
            continue
        for i in range(2, len(sub)):
            rows.append({
                "metro":            metro,
                "year":             sub.loc[i, "year"],
                "emp":              sub.loc[i, "employment_thousands"],
                "emp_lag1":         sub.loc[i-1, "employment_thousands"],
                "emp_lag2":         sub.loc[i-2, "employment_thousands"],
                "yoy_growth_lag1": (sub.loc[i-1, "employment_thousands"] /
                                    sub.loc[i-2, "employment_thousands"] - 1),
                "trend":            i,
            })
    return pd.DataFrame(rows)

# ── 3. DYNAMIC TRAIN / HOLDOUT SPLIT ─────────────────────
# Use last 20% of metros as holdout — works regardless of how many came back
n_holdout   = max(2, len(all_metros) // 5)
HOLDOUT_METROS = all_metros[-n_holdout:]
TRAIN_METROS   = [m for m in all_metros if m not in HOLDOUT_METROS]

print(f"\nTraining metros  ({len(TRAIN_METROS)}): {TRAIN_METROS}")
print(f"Holdout metros   ({len(HOLDOUT_METROS)}): {HOLDOUT_METROS}")

train_df   = build_features(df, TRAIN_METROS)
holdout_df = build_features(df, HOLDOUT_METROS)

print(f"\nTraining rows : {len(train_df)}")
print(f"Holdout rows  : {len(holdout_df)}")

if len(holdout_df) == 0:
    print("\nERROR: No holdout data built. Check CSV contents above.")
    raise SystemExit(1)

FEATURES = ["emp_lag1", "emp_lag2", "yoy_growth_lag1", "trend"]
TARGET   = "emp"

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_hold  = holdout_df[FEATURES].values
y_hold  = holdout_df[TARGET].values

# ── 4. FIT MODEL ─────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_hold_s  = scaler.transform(X_hold)

model = Ridge(alpha=10.0)
model.fit(X_train_s, y_train)

# ── 5. EVALUATE ──────────────────────────────────────────
holdout_df = holdout_df.copy()
holdout_df["predicted"] = model.predict(X_hold_s)

mape_by_metro = {}
for metro in HOLDOUT_METROS:
    sub  = holdout_df[holdout_df["metro"] == metro]
    if len(sub) == 0:
        continue
    mape = mean_absolute_percentage_error(sub["emp"], sub["predicted"]) * 100
    mape_by_metro[metro] = round(mape, 2)

overall_mape = mean_absolute_percentage_error(
    holdout_df["emp"], holdout_df["predicted"]
) * 100

print("\n── Holdout MAPE by metro ──────────────────────────")
for metro, mape in mape_by_metro.items():
    bar = "+" if mape <= 3 else "~"
    print(f"  [{bar}] {metro:<35} MAPE: {mape:.2f}%")
print(f"\n  Overall holdout MAPE: {overall_mape:.2f}%")

# ── 6. CHART 1: Predicted vs Actual ──────────────────────
n       = len(HOLDOUT_METROS)
ncols   = min(3, n)
nrows   = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
axes = np.array(axes).flatten()

for idx, metro in enumerate(HOLDOUT_METROS):
    sub = holdout_df[holdout_df["metro"] == metro].sort_values("year")
    ax  = axes[idx]
    ax.plot(sub["year"], sub["emp"],
            color="#1B4F8A", linewidth=2,
            label="Actual", marker="o", markersize=4)
    ax.plot(sub["year"], sub["predicted"],
            color="#E8593C", linewidth=2, linestyle="--",
            label="Predicted", marker="s", markersize=4)
    ax.set_title(metro, fontsize=10, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Employment (000s)", fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    if metro in mape_by_metro:
        ax.annotate(f"MAPE: {mape_by_metro[metro]:.1f}%",
                    xy=(0.05, 0.88), xycoords="axes fraction",
                    fontsize=8, color="#555")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Geographic Market Expansion Forecast — Holdout Validation\n"
    f"Ridge Regression | Trained on {len(TRAIN_METROS)} metros, "
    f"evaluated on {len(HOLDOUT_METROS)} unseen metros",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figures/holdout_forecast_by_metro.png", dpi=150,
            bbox_inches="tight")
print("\nSaved: figures/holdout_forecast_by_metro.png")

# ── 7. CHART 2: MAPE bar chart ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, max(3, len(mape_by_metro)*0.6)))
colors = ["#1B4F8A" if v <= 3 else "#E8593C"
          for v in mape_by_metro.values()]
bars = ax2.barh(list(mape_by_metro.keys()),
                list(mape_by_metro.values()),
                color=colors, edgecolor="none", height=0.5)
ax2.axvline(overall_mape, color="#888", linestyle="--", linewidth=1,
            label=f"Overall MAPE: {overall_mape:.1f}%")
ax2.set_xlabel("MAPE (%)", fontsize=10)
ax2.set_title("Forecast Error by Holdout Metro",
              fontsize=12, fontweight="bold")
for bar, val in zip(bars, mape_by_metro.values()):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/mape_by_metro.png", dpi=150, bbox_inches="tight")
print("Saved: figures/mape_by_metro.png")

# ── 8. CHART 3: Feature coefficients ─────────────────────
coef_df = pd.DataFrame({
    "feature":     FEATURES,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=True)

fig3, ax3 = plt.subplots(figsize=(7, 4))
colors3 = ["#1B4F8A" if c >= 0 else "#E8593C"
           for c in coef_df["coefficient"]]
ax3.barh(coef_df["feature"], coef_df["coefficient"],
         color=colors3, edgecolor="none", height=0.4)
ax3.axvline(0, color="#333", linewidth=0.8)
ax3.set_title("Ridge Regression Coefficients (standardized features)",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Coefficient value", fontsize=10)
ax3.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/feature_coefficients.png", dpi=150,
            bbox_inches="tight")
print("Saved: figures/feature_coefficients.png")

# ── 9. SAVE RESULTS ──────────────────────────────────────
holdout_df[["metro","year","emp","predicted"]].to_csv(
    "data/processed/holdout_forecast_results.csv", index=False
)
print("Saved: data/processed/holdout_forecast_results.csv")
print("\nAll done.")