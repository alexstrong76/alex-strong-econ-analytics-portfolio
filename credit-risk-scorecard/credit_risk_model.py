import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(7)
N = 15000

# ── 1. SIMULATE CREDIT BUREAU DATA ──────────────────────────────────────────
age              = np.random.normal(42, 12, N).clip(18, 80).astype(int)
income_k         = np.random.lognormal(4.0, 0.55, N).clip(15, 500)
loan_amount_k    = np.random.lognormal(3.2, 0.7, N).clip(1, 100)
dti              = (loan_amount_k * 12 / (income_k * 1000) * 100).clip(1, 60)
fico             = np.random.normal(690, 75, N).clip(300, 850).astype(int)
revolving_util   = np.random.beta(2.5, 5, N) * 100
num_open_accts   = np.random.poisson(5.5, N).clip(1, 20)
num_derog        = np.random.poisson(0.4, N).clip(0, 8)
months_employed  = np.random.exponential(48, N).clip(0, 360).astype(int)
num_inq_6mo      = np.random.poisson(1.0, N).clip(0, 10)
home_owner       = np.random.binomial(1, 0.58, N)
loan_purpose     = np.random.choice(
    ["debt_consolidation","home_improvement","auto","medical","other"],
    N, p=[0.38, 0.22, 0.18, 0.10, 0.12])

log_odds = (
    -2.0
    - 0.80 * (fico - 650) / 100
    + 0.25 * (dti - 25) / 10
    + 0.20 * (revolving_util - 30) / 20
    + 0.60 * num_derog
    + 0.25 * num_inq_6mo
    - 0.15 * (income_k - 60) / 30
    - 0.10 * np.log1p(months_employed)
    + 0.20 * (loan_amount_k - 15) / 15
)
default_prob = 1 / (1 + np.exp(-log_odds))
default      = np.random.binomial(1, default_prob)
print(f"Dataset: {N:,} applicants | Default rate: {default.mean():.1%}")

df = pd.DataFrame({
    "age": age, "income_k": income_k.round(1), "loan_amount_k": loan_amount_k.round(1),
    "dti": dti.round(1), "fico": fico, "revolving_util": revolving_util.round(1),
    "num_open_accts": num_open_accts, "num_derog": num_derog,
    "months_employed": months_employed, "num_inq_6mo": num_inq_6mo,
    "home_owner": home_owner, "loan_purpose": loan_purpose,
    "default_prob_true": default_prob.round(4), "default": default,
})

# ── 2. WEIGHT OF EVIDENCE ────────────────────────────────────────────────────
def woe_bin(feature, target, n_bins=10):
    df_tmp = pd.DataFrame({"x": feature, "y": target})
    df_tmp["bin"] = pd.qcut(df_tmp["x"], q=n_bins, duplicates="drop")
    g = df_tmp.groupby("bin")["y"].agg(["sum","count"])
    g.columns = ["bad","total"]
    g["good"]      = g["total"] - g["bad"]
    total_bad, total_good = g["bad"].sum(), g["good"].sum()
    g["bad_rate"]  = g["bad"]  / total_bad
    g["good_rate"] = g["good"] / total_good
    g["woe"] = np.log(g["good_rate"].clip(1e-4) / g["bad_rate"].clip(1e-4))
    g["iv"]  = (g["good_rate"] - g["bad_rate"]) * g["woe"]
    return g, g["iv"].sum()

woe_features = ["fico", "dti", "revolving_util", "num_derog", "income_k", "num_inq_6mo"]
iv_results, woe_tables = {}, {}
for feat in woe_features:
    tbl, iv = woe_bin(df[feat], df["default"])
    iv_results[feat] = iv
    woe_tables[feat] = tbl

# ── 3. MODEL TRAINING ────────────────────────────────────────────────────────
feature_cols = ["fico","dti","revolving_util","num_derog","income_k",
                "loan_amount_k","num_inq_6mo","months_employed",
                "home_owner","num_open_accts","age"]
X = df[feature_cols].values
y = df["default"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_sc, y_train)
y_prob = lr.predict_proba(X_te_sc)[:, 1]
y_pred = (y_prob > 0.5).astype(int)
auc_test = roc_auc_score(y_test, y_prob)
cv_auc   = cross_val_score(lr, X_tr_sc, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring="roc_auc")

# ── 4. SCORECARD SCALING ─────────────────────────────────────────────────────
BASE_SCORE, PDO, BASE_ODDS = 600, 20, 1/9
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

def prob_to_score(prob):
    odds  = (1 - prob) / prob.clip(1e-6)
    score = offset + factor * np.log(odds.clip(1e-6))
    return score.clip(300, 850).round().astype(int)

scores_all = prob_to_score(lr.predict_proba(scaler.transform(X))[:, 1])
df["credit_score"] = scores_all
bins_   = [0, 500, 580, 640, 700, 750, 800, 900]
labels_ = ["<500","500-579","580-639","640-699","700-749","750-799","800+"]
df["score_band"] = pd.cut(df["credit_score"], bins=bins_, labels=labels_)
band_stats = df.groupby("score_band", observed=True).agg(
    count=("default","count"), defaults=("default","sum"),
    default_rate=("default","mean"), avg_score=("credit_score","mean")).round(3)

coef_df = pd.DataFrame({
    "feature": feature_cols, "coefficient": lr.coef_[0],
    "abs_coef": np.abs(lr.coef_[0])
}).sort_values("abs_coef", ascending=True)

# ── 5. VISUALIZATIONS ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#F8FAFC")
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

GREEN="#1a7f4b"; BLUE="#2563EB"; RED="#DC2626"; GREY="#6B7280"; LGREY="#F3F4F6"; DARK="#111827"

def ax_style(ax):
    ax.set_facecolor(LGREY); ax.grid(True, alpha=0.4, lw=0.8)
    ax.spines[["top","right"]].set_visible(False)

fpr, tpr, _ = roc_curve(y_test, y_prob)
ks_stat = max(tpr - fpr)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp_ = cm.ravel()
precision = tp_/(tp_+fp) if (tp_+fp) > 0 else 0
recall    = tp_/(tp_+fn) if (tp_+fn) > 0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

# ROC
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"Logistic Reg (AUC={auc_test:.3f})")
ax1.plot([0,1],[0,1], color=GREY, lw=1.5, ls="--", label="Random")
ax1.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
ax1.set_title(f"ROC Curve\nCV AUC: {cv_auc.mean():.3f}±{cv_auc.std():.3f}", fontsize=12, fontweight="bold")
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend(fontsize=9); ax_style(ax1)

# Score distribution
ax2 = fig.add_subplot(gs[0,1])
ax2.hist(df[df["default"]==0]["credit_score"], bins=50, alpha=0.6, color=GREEN,
         label="Non-Default", density=True, edgecolor="none")
ax2.hist(df[df["default"]==1]["credit_score"], bins=50, alpha=0.6, color=RED,
         label="Default", density=True, edgecolor="none")
ax2.set_title("Score Distribution:\nDefault vs Non-Default", fontsize=12, fontweight="bold")
ax2.set_xlabel("Credit Score"); ax2.set_ylabel("Density")
ax2.legend(fontsize=9); ax_style(ax2)
ax2.text(0.02, 0.96, f"KS = {ks_stat:.3f}", transform=ax2.transAxes, fontsize=9,
         va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Feature importance
ax3 = fig.add_subplot(gs[0,2])
colors_c = [RED if c > 0 else GREEN for c in coef_df["coefficient"]]
ax3.barh(coef_df["feature"], coef_df["coefficient"], color=colors_c, alpha=0.85, edgecolor="white")
ax3.axvline(0, color="black", lw=1.2)
ax3.set_title("Logistic Regression Coefficients\n(+ = higher default risk)", fontsize=12, fontweight="bold")
flabels = {"fico":"FICO Score","dti":"Debt-to-Income","revolving_util":"Revolving Util %",
           "num_derog":"# Derog Marks","income_k":"Income ($k)","loan_amount_k":"Loan Amount",
           "num_inq_6mo":"# Inquiries (6mo)","months_employed":"Months Employed",
           "home_owner":"Homeowner","num_open_accts":"# Open Accounts","age":"Age"}
ax3.set_yticklabels([flabels.get(f,f) for f in coef_df["feature"]], fontsize=9); ax_style(ax3)

# WoE FICO
ax4 = fig.add_subplot(gs[1,0])
wf = woe_tables["fico"]
ax4.bar(range(len(wf)), wf["woe"],
        color=[GREEN if w>0 else RED for w in wf["woe"]], alpha=0.8, edgecolor="white")
ax4.set_title(f"WoE: FICO Score\nIV={iv_results['fico']:.3f} (Strong Predictor)", fontsize=12, fontweight="bold")
ax4.set_xlabel("FICO Bin (low→high)"); ax4.set_ylabel("WoE"); ax4.axhline(0,color="black",lw=1); ax_style(ax4)
ax4.set_xticks(range(len(wf)))
ax4.set_xticklabels([str(b).split(",")[0].replace("(","").replace("[","")[:6]
                     for b in wf.index], rotation=45, fontsize=7)

# WoE DTI
ax5 = fig.add_subplot(gs[1,1])
wd = woe_tables["dti"]
ax5.bar(range(len(wd)), wd["woe"],
        color=[GREEN if w>0 else RED for w in wd["woe"]], alpha=0.8, edgecolor="white")
ax5.set_title(f"WoE: Debt-to-Income\nIV={iv_results['dti']:.3f}", fontsize=12, fontweight="bold")
ax5.set_xlabel("DTI Bin (low→high)"); ax5.set_ylabel("WoE"); ax5.axhline(0,color="black",lw=1); ax_style(ax5)
ax5.set_xticks(range(len(wd)))
ax5.set_xticklabels([str(b).split(",")[0].replace("(","").replace("[","")[:5]
                     for b in wd.index], rotation=45, fontsize=7)

# Default rate by band
ax6 = fig.add_subplot(gs[1,2])
bands = band_stats.index.astype(str); dr = band_stats["default_rate"]*100
ax6.bar(bands, dr, color=[GREEN if d<8 else (BLUE if d<18 else RED) for d in dr],
        alpha=0.85, edgecolor="white")
ax6.set_title("Default Rate by Score Band\n(Scorecard Discrimination)", fontsize=12, fontweight="bold")
ax6.set_xlabel("Score Band"); ax6.set_ylabel("Default Rate (%)")
ax6.set_xticklabels(bands, rotation=35, fontsize=9); ax_style(ax6)
for x, d in enumerate(dr):
    ax6.text(x, d+0.3, f"{d:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

# IV Summary
ax7 = fig.add_subplot(gs[2,0])
iv_s = pd.Series(iv_results).sort_values(ascending=True)
ax7.barh(iv_s.index, iv_s.values,
         color=[GREEN if v>0.3 else (BLUE if v>0.1 else GREY) for v in iv_s], alpha=0.85, edgecolor="white")
ax7.axvline(0.1, color=BLUE,  lw=1.5, ls="--", alpha=0.6, label="Weak (0.1)")
ax7.axvline(0.3, color=GREEN, lw=1.5, ls="--", alpha=0.6, label="Strong (0.3)")
ax7.set_title("Information Value by Feature\n(Predictive Power)", fontsize=12, fontweight="bold")
ax7.set_xlabel("Information Value"); ax7.legend(fontsize=8); ax_style(ax7)
iv_lbl = {"fico":"FICO Score","dti":"Debt-to-Income","revolving_util":"Revolving Util %",
          "num_derog":"# Derog Marks","income_k":"Income ($k)","num_inq_6mo":"# Inquiries"}
ax7.set_yticklabels([iv_lbl.get(f,f) for f in iv_s.index], fontsize=9)

# Calibration
ax8 = fig.add_subplot(gs[2,1])
pt, pp = calibration_curve(y_test, y_prob, n_bins=12)
ax8.plot(pp, pt, "s-", color=BLUE, lw=2, ms=7, label="Model")
ax8.plot([0,1],[0,1], color=GREY, lw=1.5, ls="--", label="Perfect")
ax8.fill_between(pp, pt, pp, alpha=0.15, color=BLUE)
ax8.set_title("Calibration Curve\n(Predicted vs Actual Default Rate)", fontsize=12, fontweight="bold")
ax8.set_xlabel("Predicted Probability"); ax8.set_ylabel("Observed Default Rate")
ax8.legend(fontsize=9); ax_style(ax8)

# Summary box
ax9 = fig.add_subplot(gs[2,2])
ax9.axis("off")
summary_lines = [
    ("SCORECARD SUMMARY", True, DARK, 12),("", False, DARK, 4),
    (f"N = {N:,} applicants", False, DARK, 9.5),
    (f"Default Rate: {default.mean():.1%}", False, DARK, 9.5),("", False, DARK, 4),
    ("Discrimination", True, BLUE, 10),
    (f"Test AUC:    {auc_test:.4f}", False, GREEN, 9.5),
    (f"CV AUC:      {cv_auc.mean():.4f}±{cv_auc.std():.4f}", False, GREEN, 9.5),
    (f"KS Stat:     {ks_stat:.4f}", False, GREEN, 9.5),("", False, DARK, 4),
    ("Classification (0.5 cutoff)", True, BLUE, 10),
    (f"Precision:   {precision:.3f}", False, DARK, 9.5),
    (f"Recall:      {recall:.3f}", False, DARK, 9.5),
    (f"F1 Score:    {f1:.3f}", False, DARK, 9.5),("", False, DARK, 4),
    ("Top Predictors (IV)", True, BLUE, 10),
    (f"FICO Score:  IV = {iv_results['fico']:.3f}", False, GREEN, 9.5),
    (f"Derog Marks: IV = {iv_results['num_derog']:.3f}", False, GREEN, 9.5),
    (f"DTI Ratio:   IV = {iv_results['dti']:.3f}", False, GREEN, 9.5),
]
y_pos = 0.99
for text, bold, color, size in summary_lines:
    ax9.text(0.04, y_pos, text, transform=ax9.transAxes, fontsize=size,
             fontweight="bold" if bold else "normal", color=color, va="top")
    y_pos -= size*0.013 + 0.012
ax9.set_facecolor("#EEF2FF")
for spine in ax9.spines.values():
    spine.set_edgecolor(BLUE); spine.set_linewidth(1.5)

fig.suptitle("Consumer Credit Risk Scorecard: Logistic Regression + WoE Analysis\n"
             "FICO-Style Scoring Model  |  N=15,000 Simulated Applicants",
             fontsize=14, fontweight="bold", y=0.995, color=DARK)

plt.savefig("credit_risk_scorecard.png", dpi=150, bbox_inches="tight")
plt.close(); print("Dashboard saved.")

df[["age","income_k","loan_amount_k","dti","fico","revolving_util","num_derog",
    "months_employed","num_inq_6mo","home_owner","loan_purpose",
    "credit_score","default"]].to_csv("credit_applicants.csv", index=False)
band_stats.to_csv("scorecard_band_performance.csv")
print(f"AUC={auc_test:.4f} | KS={ks_stat:.4f} | CV={cv_auc.mean():.4f}±{cv_auc.std():.4f}")
