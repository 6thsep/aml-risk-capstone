"""
QM640 Capstone: Notebook 02: Exploratory Data Analysis
Produces all figures for the interim report and prints descriptive stats.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(ROOT, "data", "processed")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(os.path.join(PROC, "merged_dataset_2023.csv"))
print(f"Dataset loaded: {df.shape[0]} countries, {df.shape[1]} columns\n")

PRED_COLS = [
    "corruption_control", "rule_of_law", "govt_effectiveness",
    "regulatory_quality", "cpi_score", "bank_branches_per100k",
    "financial_account_pct", "gdp_log"
]
LABELS = {
    "basel_score":            "Basel AML Score",
    "corruption_control":     "WGI Control of Corruption",
    "rule_of_law":            "WGI Rule of Law",
    "govt_effectiveness":     "WGI Govt Effectiveness",
    "regulatory_quality":     "WGI Regulatory Quality",
    "cpi_score":              "TI CPI Score",
    "bank_branches_per100k":  "Bank Branches / 100k",
    "financial_account_pct":  "Financial Account %",
    "gdp_log":                "GDP per Capita (log)",
}

sns.set_theme(style="whitegrid", palette="muted", font="Times New Roman")
plt.rcParams.update({"font.family": "DejaVu Serif", "axes.titlesize": 11,
                     "axes.labelsize": 10, "figure.dpi": 150})

# ── TABLE 3: Descriptive statistics ──────────────────────────────────────────
stats_cols = ["basel_score"] + PRED_COLS
desc = df[stats_cols].describe().T[["mean","std","min","max"]]
desc["skewness"] = df[stats_cols].skew()
desc.index = [LABELS[c] for c in desc.index]
desc = desc.round(3)
print("=== TABLE 3: Descriptive Statistics ===")
print(desc.to_string())
desc.to_csv(os.path.join(FIGURES, "table3_descriptive_stats.csv"))

# ── FIGURE 1: Distribution of Basel AML Index scores ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["basel_score"], bins=20, kde=True, ax=ax,
             color="#2c7bb6", edgecolor="white", alpha=0.8)
ax.axvline(df["basel_score"].mean(), color="#d7191c", linestyle="--",
           linewidth=1.5, label=f"Mean = {df['basel_score'].mean():.2f}")
ax.set_xlabel("Basel AML Index Score (0 = Low Risk, 10 = High Risk)")
ax.set_ylabel("Frequency")
ax.set_title(
    f"Figure 1. Distribution of Basel AML Index 2023 Scores (n = {len(df)})\n"
    f"Mean = {df['basel_score'].mean():.2f}, SD = {df['basel_score'].std():.2f}, "
    f"Skewness = {df['basel_score'].skew():.2f}"
)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure1_basel_distribution.png"), bbox_inches="tight")
plt.close()
print("\nFigure 1 saved.")
print(f"  Basel AML: mean={df['basel_score'].mean():.3f}, "
      f"SD={df['basel_score'].std():.3f}, skew={df['basel_score'].skew():.3f}")

# ── FIGURE 2: Correlation heatmap ────────────────────────────────────────────
corr_cols = ["basel_score"] + PRED_COLS
corr_labels = [LABELS[c] for c in corr_cols]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr, dtype=bool)  # show full matrix
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
            xticklabels=corr_labels, yticklabels=corr_labels,
            ax=ax, linewidths=0.5, annot_kws={"size": 8},
            vmin=-1, vmax=1, center=0)
ax.set_title("Figure 2. Pearson Correlation Matrix: All Analytical Variables",
             fontsize=11, pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure2_correlation_heatmap.png"), bbox_inches="tight")
plt.close()
print("Figure 2 saved.")

# Print top correlations with Basel
print("\nCorrelations with Basel AML Score:")
corr_with_basel = corr["basel_score"].drop("basel_score").sort_values()
for var, val in corr_with_basel.items():
    print(f"  {LABELS.get(var, var)}: r = {val:.3f}")

# Highest inter-predictor correlation
corr_pred = df[PRED_COLS].corr()
corr_pred_arr = corr_pred.values.copy()
np.fill_diagonal(corr_pred_arr, np.nan)
corr_pred_masked = pd.DataFrame(corr_pred_arr, index=corr_pred.index, columns=corr_pred.columns)
max_corr = corr_pred_masked.stack().abs().idxmax()
print(f"\nHighest inter-predictor correlation: {max_corr[0]} vs {max_corr[1]}: "
      f"r = {corr_pred.loc[max_corr[0], max_corr[1]]:.3f}")

# ── FIGURE 3: Scatter plots Basel score vs 4 key predictors ──────────────────
scatter_vars = [
    ("corruption_control", "WGI Control of Corruption"),
    ("cpi_score",          "TI CPI Score (0=Most Corrupt)"),
    ("gdp_log",            "GDP per Capita (log USD)"),
    ("bank_branches_per100k", "Bank Branches per 100k Adults"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
for i, (var, label) in enumerate(scatter_vars):
    ax = axes[i]
    ax.scatter(df[var], df["basel_score"], alpha=0.5, s=30,
               color="#2c7bb6", edgecolors="white", linewidths=0.3)
    # OLS regression line
    m, b = np.polyfit(df[var].dropna(), df.loc[df[var].notna(), "basel_score"], 1)
    x_range = np.linspace(df[var].min(), df[var].max(), 100)
    ax.plot(x_range, m * x_range + b, color="#d7191c", linewidth=1.5, label=f"OLS slope")
    r = df[["basel_score", var]].corr().iloc[0, 1]
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Basel AML Score", fontsize=9)
    ax.set_title(f"r = {r:.3f}", fontsize=9)
    ax.tick_params(labelsize=8)

fig.suptitle(
    "Figure 3. Basel AML Index 2023 Score vs. Key Predictors\n"
    "OLS regression line shown in red. Negative r values confirm higher governance quality associates with lower AML risk.",
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure3_scatter_plots.png"), bbox_inches="tight")
plt.close()
print("Figure 3 saved.")

print("\nEDA complete. All figures saved to:", FIGURES)
