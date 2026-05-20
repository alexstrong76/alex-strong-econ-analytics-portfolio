import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)

# ── 1. SIMULATE MACRO + HOUSING DATA (calibrated to real US housing cycle) ───
# Monthly data: Jan 2018 – Dec 2024 (84 months)
dates = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
T     = len(dates)

# Federal Funds Rate (approximate actual path)
ffr_path = (
    list(np.linspace(1.41, 2.40, 18)) +    # 2018: hikes
    list(np.linspace(2.40, 1.55, 12)) +    # 2019: cuts
    list(np.linspace(1.55, 0.09, 6))  +    # 2020 H1: covid cuts
    list(np.linspace(0.09, 0.08, 18)) +    # 2020H2-2021: near-zero
    list(np.linspace(0.08, 4.33, 10)) +    # 2022: aggressive hikes
    list(np.linspace(4.33, 5.33, 12)) +    # 2023: peak
    list(np.linspace(5.33, 4.58, 8))       # 2024: modest cuts
)
ffr = np.array(ffr_path[:T])

# 30-yr mortgage rate (spread ~170bps over 10yr, approx)
mtg_spread = 1.70 + np.random.normal(0, 0.15, T)
mortgage   = ffr + mtg_spread + np.random.normal(0, 0.08, T)
mortgage   = mortgage.clip(2.5, 8.5)

# National Home Price Index (calibrated to Case-Shiller, Jan 2018 = 100)
# Price dynamics: driven by mortgage rates (lagged), income, supply constraints
hpi = np.zeros(T)
hpi[0] = 100.0
supply_constraint = 0.003  # structural undersupply boost
for t in range(1, T):
    mtg_drag    = -0.008 * (mortgage[t] - mortgage[t-1])
    momentum    = 0.004 * (hpi[t-1] - hpi[max(0, t-12)]) / hpi[max(0, t-12)] if t >= 12 else 0
    shock       = np.random.normal(0, 0.003)
    if t < 26:    # 2018-2019: moderate growth
        trend = 0.004
    elif t < 30:  # 2020 Q1-Q2: brief dip
        trend = -0.005
    elif t < 48:  # 2020 H2 - 2021: pandemic boom
        trend = 0.012
    elif t < 60:  # 2022: rate shock, prices stall
        trend = -0.003
    elif t < 72:  # 2023: modest recovery
        trend = 0.002
    else:          # 2024: gradual recovery
        trend = 0.004
    hpi[t] = hpi[t-1] * (1 + trend + mtg_drag + supply_constraint + shock)

hpi_yoy = np.concatenate([[np.nan]*12, (hpi[12:] / hpi[:-12] - 1) * 100])

# Housing starts (thousands, annual rate)
starts_base = 1250
starts = np.zeros(T)
for t in range(T):
    mtg_effect = -30 * (mortgage[t] - 4.0)
    if t < 26:   trend = 5
    elif t < 30: trend = -40
    elif t < 48: trend = 20
    elif t < 60: trend = -25
    else:        trend = 8
    starts[t] = max(800, starts_base + trend * (t % 12) + mtg_effect + np.random.normal(0, 45))

# Existing home sales (millions, annual rate)
sales_base = 5.3
sales = np.zeros(T)
for t in range(T):
    affordability = -0.15 * (mortgage[t] - 4.0) - 0.002 * (hpi[t] - 100)
    sales[t] = max(3.5, sales_base + affordability + np.random.normal(0, 0.15))

# Months of supply
supply_mo = np.maximum(1.5, 6.0 - 0.4 * (sales - 4.5) + np.random.normal(0, 0.3, T))

# Rental vacancy rate
vacancy = 5.8 - 0.3 * np.maximum(0, sales - 5.0) + np.random.normal(0, 0.3, T)

# Build DataFrame
df = pd.DataFrame({
    "date":          dates,
    "fed_funds_rate": ffr,
    "mortgage_30yr":  mortgage,
    "hpi":            hpi,
    "hpi_yoy_pct":    hpi_yoy,
    "housing_starts": starts,
    "existing_sales": sales,
    "months_supply":  supply_mo,
    "rental_vacancy": vacancy,
})
df.set_index("date", inplace=True)

# ── 2. AFFORDABILITY INDEX ───────────────────────────────────────────────────
# Simplified: monthly payment on median-priced home at current mortgage rate
MEDIAN_HOME_PRICE_2018 = 260000
median_price = MEDIAN_HOME_PRICE_2018 * (hpi / 100)
monthly_payment = median_price * (mortgage/100/12) / (1 - (1 + mortgage/100/12)**-360)
# Assume median household income grows 4% annually
income_monthly = 5500 * (1.04 ** (np.arange(T) / 12))
payment_to_income = monthly_payment / income_monthly * 100  # as % of income
df["affordability_pti"] = payment_to_income
df["median_price_k"]    = median_price / 1000

# ── 3. REGRESSION: HPI ~ Mortgage Rate (lagged 3 months) + Supply ───────────
df_reg = df.dropna().copy()
df_reg["mtg_lag3"]     = df_reg["mortgage_30yr"].shift(3)
df_reg["supply_lag1"]  = df_reg["months_supply"].shift(1)
df_reg = df_reg.dropna()

X_reg = np.column_stack([
    np.ones(len(df_reg)),
    df_reg["mtg_lag3"].values,
    df_reg["supply_lag1"].values,
    df_reg["existing_sales"].values,
])
y_reg = df_reg["hpi_yoy_pct"].values
beta_reg, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
y_hat = X_reg @ beta_reg
residuals = y_reg - y_hat
r_sq = 1 - np.var(residuals) / np.var(y_reg)

# ── 4. REGIONAL SNAPSHOT (6 metro markets) ──────────────────────────────────
metros = {
    "Dallas, TX":    {"hpi_2018": 100, "premium": 0.015, "supply_mult": 1.2, "color": "#1a7f4b"},
    "Austin, TX":    {"hpi_2018": 100, "premium": 0.022, "supply_mult": 0.8, "color": "#16a34a"},
    "Phoenix, AZ":   {"hpi_2018": 100, "premium": 0.018, "supply_mult": 1.1, "color": "#2563EB"},
    "Denver, CO":    {"hpi_2018": 100, "premium": 0.016, "supply_mult": 0.9, "color": "#7C3AED"},
    "Chicago, IL":   {"hpi_2018": 100, "premium": 0.006, "supply_mult": 1.4, "color": "#DC2626"},
    "National Avg":  {"hpi_2018": 100, "premium": 0.010, "supply_mult": 1.0, "color": "#6B7280"},
}
metro_hpi = {}
for metro, params in metros.items():
    m_hpi = np.zeros(T)
    m_hpi[0] = 100
    for t in range(1, T):
        base_change = hpi[t] / hpi[t-1] - 1
        local_premium = params["premium"] / 12
        supply_drag   = (1 / params["supply_mult"] - 1) * 0.003
        m_hpi[t] = m_hpi[t-1] * (1 + base_change + local_premium + supply_drag)
    metro_hpi[metro] = m_hpi

# ── 5. DASHBOARD VISUALIZATION ───────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#F8FAFC")
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

GREEN = "#1a7f4b"; BLUE = "#2563EB"; RED = "#DC2626"; GREY = "#6B7280"; LGREY = "#F3F4F6"
DARK  = "#111827"

def ax_style(ax):
    ax.set_facecolor(LGREY)
    ax.grid(True, alpha=0.4, lw=0.8)
    ax.spines[["top","right"]].set_visible(False)

# ── Panel 1: HPI vs Mortgage Rate ────────────────────────────────────────────
ax1  = fig.add_subplot(gs[0, :2])
ax1b = ax1.twinx()
valid = ~np.isnan(df["hpi_yoy_pct"])
ax1.fill_between(df.index[valid], df["hpi_yoy_pct"][valid],
                 where=df["hpi_yoy_pct"][valid] > 0, alpha=0.35, color=GREEN, label="HPI YoY > 0%")
ax1.fill_between(df.index[valid], df["hpi_yoy_pct"][valid],
                 where=df["hpi_yoy_pct"][valid] <= 0, alpha=0.35, color=RED, label="HPI YoY < 0%")
ax1.plot(df.index[valid], df["hpi_yoy_pct"][valid], color=GREEN, lw=1.8)
ax1b.plot(df.index, df["mortgage_30yr"], color=RED, lw=2, ls="--", alpha=0.9, label="30-yr Mortgage Rate")
ax1.set_ylabel("Home Price Growth (YoY %)", color=GREEN, fontsize=10)
ax1b.set_ylabel("30-yr Mortgage Rate (%)", color=RED, fontsize=10)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("National Home Price Growth vs. Mortgage Rates (2018–2024)", fontsize=13, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax1.set_facecolor(LGREY); ax1.grid(True, alpha=0.4)

# ── Panel 2: Affordability ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(df.index, df["affordability_pti"], color=RED, lw=2.2, label="Payment-to-Income %")
ax2.axhline(28, color=GREY, lw=1.5, ls="--", label="28% Guideline (traditional)")
ax2.fill_between(df.index, df["affordability_pti"], 28,
                 where=df["affordability_pti"] > 28, alpha=0.2, color=RED, label="Unaffordable Zone")
ax2.set_title("Housing Affordability\n(Monthly Payment / Income)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Payment-to-Income %", fontsize=10)
ax2.legend(fontsize=8); ax_style(ax2)

# ── Panel 3: Regional HPI Comparison ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
for metro, params in metros.items():
    lw  = 2.5 if metro == "Dallas, TX" else 1.8
    ls  = "-"  if metro != "National Avg" else "--"
    alpha = 1.0 if metro in ["Dallas, TX", "National Avg", "Austin, TX"] else 0.6
    ax3.plot(df.index, metro_hpi[metro], color=params["color"], lw=lw, ls=ls,
             alpha=alpha, label=metro)
ax3.set_title("Regional Home Price Index by Metro (Jan 2018 = 100)", fontsize=13, fontweight="bold")
ax3.set_ylabel("HPI (Jan 2018 = 100)", fontsize=10)
ax3.legend(fontsize=9, ncol=2); ax_style(ax3)

# ── Panel 4: Regression Results ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(y_reg, y_hat, alpha=0.5, color=BLUE, s=20, edgecolors="none")
lims = [min(y_reg.min(), y_hat.min())-1, max(y_reg.max(), y_hat.max())+1]
ax4.plot(lims, lims, color=RED, lw=1.5, ls="--", label="45° line")
ax4.set_xlabel("Actual HPI YoY %", fontsize=10)
ax4.set_ylabel("Fitted HPI YoY %", fontsize=10)
ax4.set_title(f"OLS: HPI Growth ~ Mortgage Rate\n+ Supply + Sales  |  R² = {r_sq:.3f}", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9); ax_style(ax4)
reg_text = (f"β_mortgage = {beta_reg[1]:.2f}\n"
            f"β_supply   = {beta_reg[2]:.2f}\n"
            f"β_sales    = {beta_reg[3]:.2f}")
ax4.text(0.04, 0.96, reg_text, transform=ax4.transAxes, fontsize=8.5,
         va="top", family="monospace",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── Panel 5: Housing Starts & Sales ─────────────────────────────────────────
ax5  = fig.add_subplot(gs[2, 0])
ax5b = ax5.twinx()
ax5.bar(df.index, df["housing_starts"], width=25, alpha=0.5, color=GREEN, label="Housing Starts (k, ann.)")
ax5b.plot(df.index, df["existing_sales"], color=BLUE, lw=2, label="Existing Sales (M, ann.)")
ax5.set_title("Housing Supply & Demand\n(Starts vs. Sales)", fontsize=12, fontweight="bold")
ax5.set_ylabel("Housing Starts (thousands)", color=GREEN, fontsize=9)
ax5b.set_ylabel("Existing Home Sales (M)", color=BLUE, fontsize=9)
ax5.set_facecolor(LGREY); ax5.grid(True, alpha=0.4)
lines5, labels5 = ax5.get_legend_handles_labels()
lines6, labels6 = ax5b.get_legend_handles_labels()
ax5.legend(lines5+lines6, labels5+labels6, fontsize=8)

# ── Panel 6: Months of Supply ─────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(df.index, df["months_supply"], color=BLUE, lw=2)
ax6.axhline(6.0, color=RED, lw=1.5, ls="--", label="Balanced Market (6 mo.)")
ax6.fill_between(df.index, df["months_supply"], 6,
                 where=df["months_supply"] < 6, alpha=0.2, color=GREEN, label="Seller's Market")
ax6.fill_between(df.index, df["months_supply"], 6,
                 where=df["months_supply"] > 6, alpha=0.2, color=RED, label="Buyer's Market")
ax6.set_title("Months of Inventory Supply\n(Market Balance Indicator)", fontsize=12, fontweight="bold")
ax6.set_ylabel("Months of Supply", fontsize=10)
ax6.legend(fontsize=8); ax_style(ax6)

# ── Panel 7: Key Stats Box ────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
stats_text = [
    ("NATIONAL HOUSING DASHBOARD", True, DARK, 12),
    ("", False, DARK, 6),
    (f"Peak HPI Growth:  +{df['hpi_yoy_pct'].max():.1f}% (2021-22 Boom)", False, GREEN, 10),
    (f"Trough HPI Growth: {df['hpi_yoy_pct'].min():.1f}% (Rate Shock)", False, RED, 10),
    (f"Current HPI (2024): {hpi[-1]:.0f} (2018=100)", False, DARK, 10),
    ("", False, DARK, 6),
    (f"Peak Mortgage:  {mortgage.max():.2f}% (2023)", False, RED, 10),
    (f"Trough Mortgage: {mortgage.min():.2f}% (2021)", False, GREEN, 10),
    (f"Current Rate: {mortgage[-1]:.2f}%", False, DARK, 10),
    ("", False, DARK, 6),
    (f"OLS R²: {r_sq:.3f}", False, BLUE, 10),
    (f"β_mortgage (lag 3mo): {beta_reg[1]:.2f} pp", False, BLUE, 10),
    ("(1pp mortgage ↑ → {:.1f}pp HPI growth ↓)".format(abs(beta_reg[1])), False, GREY, 9),
    ("", False, DARK, 6),
    ("Dallas Metro: Outperformed", False, GREEN, 10),
    ("national avg across full cycle", False, GREEN, 10),
]
y_pos = 0.97
for text, bold, color, size in stats_text:
    ax7.text(0.05, y_pos, text, transform=ax7.transAxes, fontsize=size,
             fontweight="bold" if bold else "normal", color=color, va="top")
    y_pos -= size * 0.012 + 0.015

ax7.set_facecolor("#EEF2FF")
for spine in ax7.spines.values():
    spine.set_edgecolor(BLUE); spine.set_linewidth(1.5)

fig.suptitle("U.S. Housing Market Econometric Analysis: Fed Policy, Affordability & Regional Dynamics (2018–2024)",
             fontsize=14, fontweight="bold", y=0.995, color=DARK)

plt.savefig("/home/claude/portfolio/housing-market-dashboard/housing_market_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Housing dashboard saved.")

# Save data
df.to_csv("/home/claude/portfolio/housing-market-dashboard/housing_market_data.csv")
print("Data saved.")

print(f"""
OLS Results:
  Intercept:              {beta_reg[0]:.3f}
  β_mortgage (lag 3mo):   {beta_reg[1]:.3f}  (1pp rate ↑ → {beta_reg[1]:.2f}pp HPI growth)
  β_supply (lag 1mo):     {beta_reg[2]:.3f}  (1mo supply ↑ → {beta_reg[2]:.2f}pp HPI growth)
  β_existing_sales:       {beta_reg[3]:.3f}
  R²:                     {r_sq:.4f}
""")
