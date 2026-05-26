"""
QM640 Capstone: Notebook 06: RQ3: Moderation Regression
Tests whether corruption moderates the relationship between financial access and AML risk.
Produces Table 6 (model comparison) and Figure 8 (simple slopes plot).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(ROOT, "data", "processed")
FIGURES = os.path.join(ROOT, "outputs", "figures")
RESULTS = os.path.join(ROOT, "outputs", "results")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Serif", "figure.dpi": 150,
                     "axes.titlesize": 11, "axes.labelsize": 10})

df = pd.read_csv(os.path.join(PROC, "merged_dataset_2023.csv"))
n  = len(df)
print(f"Dataset: n = {n}")

# ── Variables ─────────────────────────────────────────────────────────────────
# Corruption operationalised as WGI Control of Corruption (higher = less corrupt)
# Financial access operationalised as financial_account_pct
# Note: higher WGI corruption control = LOWER AML risk
# So we reverse-code it as "corruption_level" = -corruption_control for intuition
# Actually we keep as-is and interpret directions from coefficients

y      = df["basel_score"].values
corrupt = df["corruption_control"].values
fin_acc = df["financial_account_pct"].values
gdp    = df["gdp_log"].values

# Mean-centre both predictors before interaction (standard moderation practice)
corrupt_c = corrupt - corrupt.mean()
fin_acc_c = fin_acc - fin_acc.mean()
interaction = corrupt_c * fin_acc_c

print(f"\nCorruption control: mean={corrupt.mean():.3f}, SD={corrupt.std():.3f}")
print(f"Financial access:   mean={fin_acc.mean():.3f}, SD={fin_acc.std():.3f}")

# ── Model 1: Main effects only ────────────────────────────────────────────────
X1 = np.column_stack([np.ones(n), corrupt_c, fin_acc_c, gdp])
coef1, res1, _, _ = np.linalg.lstsq(X1, y, rcond=None)
y_pred1 = X1 @ coef1
ss_res1 = np.sum((y - y_pred1)**2)
ss_tot  = np.sum((y - y.mean())**2)
r2_m1   = 1 - ss_res1 / ss_tot
adj_r2_m1 = 1 - (1 - r2_m1) * (n - 1) / (n - 4)
rmse_m1 = np.sqrt(ss_res1 / n)

# t-tests for Model 1 coefficients
k1 = X1.shape[1]
mse1 = ss_res1 / (n - k1)
XtX_inv1 = np.linalg.inv(X1.T @ X1)
se1 = np.sqrt(np.diag(mse1 * XtX_inv1))
t1  = coef1 / se1
p1  = 2 * (1 - stats.t.cdf(np.abs(t1), df=n - k1))

print(f"\n=== Model 1: Main Effects ===")
print(f"  R² = {r2_m1:.4f},  Adj R² = {adj_r2_m1:.4f},  RMSE = {rmse_m1:.4f}")
labels_m1 = ["Intercept", "Corruption Control (centred)", "Financial Access (centred)", "GDP log"]
for lab, c, se, t, p in zip(labels_m1, coef1, se1, t1, p1):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {lab:<35}: beta={c:+.4f}, SE={se:.4f}, t={t:+.3f}, p={p:.4f} {sig}")

# ── Model 2: Main effects + interaction ──────────────────────────────────────
X2 = np.column_stack([np.ones(n), corrupt_c, fin_acc_c, interaction, gdp])
coef2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
y_pred2 = X2 @ coef2
ss_res2 = np.sum((y - y_pred2)**2)
r2_m2   = 1 - ss_res2 / ss_tot
adj_r2_m2 = 1 - (1 - r2_m2) * (n - 1) / (n - 5)
rmse_m2 = np.sqrt(ss_res2 / n)

k2  = X2.shape[1]
mse2 = ss_res2 / (n - k2)
XtX_inv2 = np.linalg.inv(X2.T @ X2)
se2 = np.sqrt(np.diag(mse2 * XtX_inv2))
t2  = coef2 / se2
p2  = 2 * (1 - stats.t.cdf(np.abs(t2), df=n - k2))

print(f"\n=== Model 2: Main Effects + Interaction ===")
print(f"  R² = {r2_m2:.4f},  Adj R² = {adj_r2_m2:.4f},  RMSE = {rmse_m2:.4f}")
labels_m2 = ["Intercept", "Corruption Control (centred)", "Financial Access (centred)",
             "Corruption x Financial Access", "GDP log"]
for lab, c, se, t, p in zip(labels_m2, coef2, se2, t2, p2):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {lab:<35}: beta={c:+.4f}, SE={se:.4f}, t={t:+.3f}, p={p:.4f} {sig}")

# ── F-test for R² change ──────────────────────────────────────────────────────
delta_r2   = r2_m2 - r2_m1
f_stat     = (delta_r2 / 1) / ((1 - r2_m2) / (n - k2))
f_pval     = 1 - stats.f.cdf(f_stat, dfn=1, dfd=n - k2)
print(f"\n=== F-test: R² change ===")
print(f"  Delta R² = {delta_r2:.4f}")
print(f"  F({1}, {n-k2}) = {f_stat:.3f}, p = {f_pval:.4f}")
moderation_sig = "SIGNIFICANT" if f_pval < 0.05 else "NOT SIGNIFICANT"
print(f"  Interaction is {moderation_sig} at alpha = 0.05")

# ── TABLE 6: Model comparison ─────────────────────────────────────────────────
table6 = pd.DataFrame({
    "": ["R²", "Adjusted R²", "RMSE", "N", "Predictors"],
    "Model 1 (Main Effects)":       [f"{r2_m1:.4f}", f"{adj_r2_m1:.4f}", f"{rmse_m1:.4f}", str(n), "Corruption, Financial Access, GDP"],
    "Model 2 (+ Interaction)":      [f"{r2_m2:.4f}", f"{adj_r2_m2:.4f}", f"{rmse_m2:.4f}", str(n), "Corruption x Financial Access added"],
    "Delta R² / F-test p":          [f"{delta_r2:.4f}", "", f"F={f_stat:.3f}", f"p={f_pval:.4f}", moderation_sig],
})
table6.to_csv(os.path.join(RESULTS, "table6_moderation.csv"), index=False)
print("\nTable 6 saved.")

# ── FIGURE 8: Simple slopes plot ──────────────────────────────────────────────
# Plot Basel AML score vs Financial Access at high (+1SD), mean, and low (-1SD) corruption
fig, ax = plt.subplots(figsize=(8, 5))

fin_range = np.linspace(fin_acc_c.min(), fin_acc_c.max(), 100)
gdp_mean  = (gdp - gdp.mean()).mean()   # 0 (centred)

for level, label, color, ls in [
    (+corrupt.std(), "High Corruption (+1 SD)", "#d7191c", "-"),
    (0,              "Mean Corruption",          "#756bb1", "--"),
    (-corrupt.std(), "Low Corruption (-1 SD)",   "#2c7bb6", ":"),
]:
    b0, b1, b2, b3, b4 = coef2
    predicted = b0 + b1 * level + b2 * fin_range + b3 * (level * fin_range) + b4 * gdp_mean
    ax.plot(fin_range + fin_acc.mean(), predicted, color=color,
            linestyle=ls, linewidth=2, label=label)

ax.set_xlabel("Financial Account Ownership (% Adults)", fontsize=10)
ax.set_ylabel("Basel AML Index Score (Predicted)", fontsize=10)
ax.set_title(
    f"Figure 8. Simple Slopes: Effect of Financial Access on AML Risk\n"
    f"at Three Levels of Corruption Control (n = {n})\n"
    f"Interaction beta = {coef2[3]:+.4f}, F({1},{n-k2}) = {f_stat:.3f}, p = {f_pval:.4f}",
    fontsize=9
)
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.set_ylabel("Basel AML Index Score (Predicted)\n← Lower score = less risk")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure8_simple_slopes.png"), bbox_inches="tight")
plt.close()
print("Figure 8 saved.")
print("\nRQ3 complete.")
