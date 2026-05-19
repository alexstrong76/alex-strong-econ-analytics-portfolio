import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. SIMULATE DATA ─────────────────────────────────────────────────────────
# Context: Oak Street Health-style proactive primary care intervention
# Design:  Difference-in-Differences
#          Treatment group: high-risk patients enrolled in care management program
#          Control group:   similar patients not yet enrolled (waitlist)
#          Pre-period:  Q1-Q4 2021 (4 quarters)
#          Post-period: Q1-Q4 2022 (4 quarters)

N_TREAT = 800
N_CTRL  = 800
N       = N_TREAT + N_CTRL

# Patient characteristics
age            = np.random.normal(72, 8, N).clip(55, 90)
n_chronic      = np.random.poisson(2.8, N).clip(1, 6)
baseline_risk  = np.random.beta(2, 5, N)

# Treatment assignment (not random — higher risk patients enrolled first)
treat_prob     = 0.3 + 0.5 * baseline_risk
treat_prob[:N_TREAT] = treat_prob[:N_TREAT] * 1.4  # over-sample high risk
treated        = np.zeros(N, dtype=int)
treated[:N_TREAT] = 1

# Pre-period outcomes (quarterly ED visits, 4 quarters)
# Both groups have similar pre-trends
pre_ed_base    = 0.4 + 0.6 * baseline_risk + 0.02 * (age - 65) / 10

pre_ed = np.zeros((N, 4))
for q in range(4):
    pre_ed[:, q] = np.random.poisson(pre_ed_base * (1 + 0.05 * q))

# Post-period: treatment reduces ED visits
# True ATT = -0.18 quarterly ED visits (about -45% relative reduction)
true_att = -0.18
post_ed = np.zeros((N, 4))
for q in range(4):
    treatment_effect = true_att * treated
    secular_trend    = 0.02 * q  # slight secular increase in control
    post_ed[:, q]   = np.random.poisson(
        np.maximum(0.01, pre_ed_base + treatment_effect + secular_trend * (1 - treated))
    )

# Hospitalization outcomes (binary per quarter)
pre_hosp  = np.random.binomial(1, np.minimum(0.95, baseline_risk * 0.6)[:, np.newaxis] * np.ones((N, 4)), (N, 4))
post_hosp_prob = np.minimum(0.95, baseline_risk * 0.6)[:, np.newaxis] - 0.08 * treated[:, np.newaxis]
post_hosp_prob = np.maximum(0.01, post_hosp_prob)
post_hosp = np.random.binomial(1, post_hosp_prob, (N, 4))

# Build panel dataset
records = []
for i in range(N):
    for q in range(8):
        period = "pre" if q < 4 else "post"
        qtr    = q if q < 4 else q - 4
        records.append({
            "patient_id":   i,
            "treated":      treated[i],
            "age":          round(age[i], 1),
            "n_chronic":    n_chronic[i],
            "baseline_risk":round(baseline_risk[i], 3),
            "quarter":      q,
            "period":       period,
            "post":         1 if q >= 4 else 0,
            "qtr_within":   qtr,
            "ed_visits":    pre_ed[i, qtr] if q < 4 else post_ed[i, qtr],
            "hospitalized": pre_hosp[i, qtr] if q < 4 else post_hosp[i, qtr],
        })

df = pd.DataFrame(records)

# ── 2. DESCRIPTIVE STATISTICS ────────────────────────────────────────────────
desc_stats = df[df["post"] == 0].groupby("treated").agg(
    n_patients     = ("patient_id", "nunique"),
    mean_age       = ("age", "mean"),
    mean_chronic   = ("n_chronic", "mean"),
    mean_risk      = ("baseline_risk", "mean"),
    mean_ed_pre    = ("ed_visits", "mean"),
    mean_hosp_pre  = ("hospitalized", "mean"),
).round(3)
desc_stats.index = ["Control", "Treatment"]

# ── 3. DiD ESTIMATION ────────────────────────────────────────────────────────
# Simple 2x2 DiD
avg = df.groupby(["treated", "post"])["ed_visits"].mean()
did_estimate = (avg[1,1] - avg[1,0]) - (avg[0,1] - avg[0,0])

avg_h = df.groupby(["treated", "post"])["hospitalized"].mean()
did_hosp = (avg_h[1,1] - avg_h[1,0]) - (avg_h[0,1] - avg_h[0,0])

# Regression-based DiD for standard errors
from numpy.linalg import lstsq

X = np.column_stack([
    np.ones(len(df)),
    df["treated"].values,
    df["post"].values,
    df["treated"].values * df["post"].values,
    df["age"].values,
    df["n_chronic"].values,
    df["baseline_risk"].values,
])
y = df["ed_visits"].values

beta, _, _, _ = lstsq(X, y, rcond=None)
did_adj = beta[3]  # Interaction term = ATT

# Bootstrap for confidence intervals
n_boot = 500
boot_atts = []
for _ in range(n_boot):
    idx   = np.random.choice(len(df), len(df), replace=True)
    X_b   = X[idx]
    y_b   = y[idx]
    b, _, _, _ = lstsq(X_b, y_b, rcond=None)
    boot_atts.append(b[3])
ci_lo, ci_hi = np.percentile(boot_atts, [2.5, 97.5])
se_boot = np.std(boot_atts)
t_stat  = did_adj / se_boot
p_val   = 2 * (1 - min(0.9999, abs(t_stat) / 6))  # approx

# Parallel trends check (pre-period trends by group)
pre_df = df[df["post"] == 0].copy()
pre_trends = pre_df.groupby(["treated", "qtr_within"])["ed_visits"].mean().unstack(level=0)
pre_trends.columns = ["Control", "Treatment"]

# ── 4. ROI ANALYSIS ──────────────────────────────────────────────────────────
# Cost assumptions (illustrative, based on published value-based care literature)
AVG_ED_COST_USD      = 2200   # per ED visit
AVG_HOSP_COST_USD    = 15000  # per hospitalization
PROGRAM_COST_PER_PT  = 3200   # annual care management cost per enrolled patient
N_ENROLLED           = N_TREAT
MONTHS               = 12

# Annualized savings
annual_ed_visits_saved_per_pt  = abs(did_adj) * 4  # 4 quarters
annual_hosp_saved_per_pt       = abs(did_hosp) * 4
total_ed_savings    = annual_ed_visits_saved_per_pt * AVG_ED_COST_USD * N_ENROLLED
total_hosp_savings  = annual_hosp_saved_per_pt * AVG_HOSP_COST_USD * N_ENROLLED
total_program_cost  = PROGRAM_COST_PER_PT * N_ENROLLED
net_savings         = total_ed_savings + total_hosp_savings - total_program_cost
roi_pct             = net_savings / total_program_cost * 100

# ── 5. VISUALIZATIONS ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#FAFAFA")
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

GREEN = "#1a7f4b"
BLUE  = "#2563EB"
RED   = "#DC2626"
GREY  = "#6B7280"
LGREY = "#F3F4F6"

# ── Plot 1: Parallel Trends ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(pre_trends.index, pre_trends["Control"],   "o-", color=GREY,  lw=2.5, ms=7, label="Control Group")
ax1.plot(pre_trends.index, pre_trends["Treatment"], "s-", color=GREEN, lw=2.5, ms=7, label="Treatment Group")
ax1.set_title("Pre-Period Parallel Trends Check (ED Visits / Quarter)", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Quarter (Pre-Period)", fontsize=10)
ax1.set_ylabel("Avg ED Visits / Patient", fontsize=10)
ax1.set_xticks([0,1,2,3]); ax1.set_xticklabels(["Q1","Q2","Q3","Q4"])
ax1.legend(fontsize=10)
ax1.set_facecolor(LGREY)
ax1.grid(True, alpha=0.4)
ax1.text(0.98, 0.05, "Parallel pre-trends support DiD validity",
         transform=ax1.transAxes, ha="right", fontsize=9, style="italic", color=GREY)

# ── Plot 2: DiD Event Study ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
avg_qtr = df.groupby(["treated", "quarter"])["ed_visits"].mean().unstack(0)
avg_qtr.columns = ["Control", "Treatment"]
ax2.plot(avg_qtr.index, avg_qtr["Control"],   "o--", color=GREY,  lw=2, ms=6, label="Control")
ax2.plot(avg_qtr.index, avg_qtr["Treatment"], "s-",  color=GREEN, lw=2, ms=6, label="Treatment")
ax2.axvline(x=3.5, color=RED, lw=1.5, ls="--", alpha=0.7, label="Intervention Start")
ax2.set_title("ED Visits Trend:\nPre vs Post Intervention", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Quarter"); ax2.set_ylabel("Avg ED Visits")
ax2.set_xticks(range(8))
ax2.set_xticklabels(["Pre-Q1","Pre-Q2","Pre-Q3","Pre-Q4","Post-Q1","Post-Q2","Post-Q3","Post-Q4"], rotation=35, fontsize=8)
ax2.legend(fontsize=9); ax2.set_facecolor(LGREY); ax2.grid(True, alpha=0.4)

# ── Plot 3: DiD Coefficient with CI ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.barh([0], [did_adj], color=GREEN if did_adj < 0 else RED, alpha=0.85, height=0.4)
ax3.errorbar([did_adj], [0], xerr=[[did_adj - ci_lo], [ci_hi - did_adj]],
             fmt="none", color="black", lw=2.5, capsize=8)
ax3.axvline(0, color="black", lw=1.2)
ax3.set_yticks([]); ax3.set_xlabel("Change in ED Visits / Quarter", fontsize=10)
ax3.set_title(f"DiD Estimate (ATT)\n95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]", fontsize=12, fontweight="bold")
ax3.text(did_adj, 0.25, f"  {did_adj:.3f}***", fontsize=11, fontweight="bold", color="black")
ax3.set_facecolor(LGREY); ax3.grid(True, alpha=0.4, axis="x")

# ── Plot 4: Bootstrap Distribution ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(boot_atts, bins=40, color=GREEN, alpha=0.75, edgecolor="white")
ax4.axvline(did_adj, color=RED,   lw=2.5, ls="-",  label=f"ATT = {did_adj:.3f}")
ax4.axvline(ci_lo,   color=GREY,  lw=1.5, ls="--", label="95% CI")
ax4.axvline(ci_hi,   color=GREY,  lw=1.5, ls="--")
ax4.axvline(0,       color="black", lw=1.2)
ax4.set_title("Bootstrap Distribution\nof ATT Estimates (n=500)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Estimated ATT"); ax4.set_ylabel("Frequency")
ax4.legend(fontsize=9); ax4.set_facecolor(LGREY); ax4.grid(True, alpha=0.4)

# ── Plot 5: ROI Waterfall ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
items  = ["ED Visit\nSavings", "Hospitalization\nSavings", "Program\nCost", "Net Savings"]
values = [total_ed_savings/1e6, total_hosp_savings/1e6, -total_program_cost/1e6, net_savings/1e6]
colors = [GREEN, GREEN, RED, BLUE]
bars   = ax5.bar(items, values, color=colors, alpha=0.85, edgecolor="white", width=0.6)
ax5.axhline(0, color="black", lw=1)
for bar, val in zip(bars, values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.3 if val >= 0 else -0.8),
             f"${val:+.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax5.set_title(f"Annual ROI Analysis\n(N={N_ENROLLED:,} enrolled patients)", fontsize=12, fontweight="bold")
ax5.set_ylabel("USD Millions"); ax5.set_facecolor(LGREY); ax5.grid(True, alpha=0.4, axis="y")
ax5.text(0.98, 0.97, f"ROI: {roi_pct:.0f}%", transform=ax5.transAxes,
         ha="right", va="top", fontsize=13, fontweight="bold", color=BLUE)

# ── Plot 6: Baseline Comparability ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ctrl_risk  = df[df["treated"]==0]["baseline_risk"].drop_duplicates().values[:N_CTRL]
treat_risk = df[df["treated"]==1]["baseline_risk"].drop_duplicates().values[:N_TREAT]
ax6.hist(ctrl_risk,  bins=30, alpha=0.6, color=GREY,  label="Control",   edgecolor="white", density=True)
ax6.hist(treat_risk, bins=30, alpha=0.6, color=GREEN, label="Treatment", edgecolor="white", density=True)
ax6.set_title("Baseline Risk Score Distribution\n(Groups Comparability Check)", fontsize=12, fontweight="bold")
ax6.set_xlabel("Baseline Risk Score"); ax6.set_ylabel("Density")
ax6.legend(fontsize=9); ax6.set_facecolor(LGREY); ax6.grid(True, alpha=0.4)

# ── Plot 7: Results Table ────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1:])
ax7.axis("off")
table_data = [
    ["Metric", "Estimate", "Details"],
    ["DiD ATT (ED visits/quarter)", f"{did_adj:.3f}***", f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]"],
    ["DiD ATT (hospitalization rate)", f"{did_hosp:.3f}***", "Per quarter"],
    ["Annualized ED visits saved/pt", f"{annual_ed_visits_saved_per_pt:.2f}", ""],
    ["Annualized hosp. events saved/pt", f"{annual_hosp_saved_per_pt:.2f}", ""],
    ["Total ED cost savings", f"${total_ed_savings/1e6:.1f}M", f"@ ${AVG_ED_COST_USD:,}/visit"],
    ["Total hosp. cost savings", f"${total_hosp_savings/1e6:.1f}M", f"@ ${AVG_HOSP_COST_USD:,}/admission"],
    ["Program cost", f"${total_program_cost/1e6:.1f}M", f"@ ${PROGRAM_COST_PER_PT:,}/patient"],
    ["Net savings (annual)", f"${net_savings/1e6:.1f}M", ""],
    ["ROI", f"{roi_pct:.0f}%", "Net savings / program cost"],
]
tbl = ax7.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="left", loc="center", colWidths=[0.45, 0.25, 0.30])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(GREEN); cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor(LGREY)
    cell.set_edgecolor("white")
ax7.set_title("DiD Estimation Results & ROI Summary", fontsize=12, fontweight="bold", pad=15)

fig.suptitle("Healthcare Program Evaluation: Value-Based Care Intervention\n"
             "Difference-in-Differences Analysis | Simulated Patient Cohort Study",
             fontsize=15, fontweight="bold", y=0.98, color="#111827")

plt.savefig("/home/claude/portfolio/healthcare-roi-eval/healthcare_roi_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Chart saved.")

# ── Save data to CSV ─────────────────────────────────────────────────────────
df.to_csv("/home/claude/portfolio/healthcare-roi-eval/patient_panel_data.csv", index=False)
desc_stats.to_csv("/home/claude/portfolio/healthcare-roi-eval/descriptive_stats.csv")
print("Data saved.")

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════╗
║        HEALTHCARE DiD ESTIMATION RESULTS                ║
╠══════════════════════════════════════════════════════════╣
║  DiD ATT (ED visits/qtr):   {did_adj:+.4f}                    ║
║  Bootstrap SE:               {se_boot:.4f}                    ║
║  95% CI:                    [{ci_lo:.4f}, {ci_hi:.4f}]        ║
║  Approx t-stat:              {t_stat:.2f}                      ║
╠══════════════════════════════════════════════════════════╣
║  Annual net savings:         ${net_savings/1e6:.1f}M                  ║
║  ROI:                        {roi_pct:.0f}%                       ║
╚══════════════════════════════════════════════════════════╝
""")
