"""
QM640 Capstone: Notebook 04: RQ1: LASSO Feature Importance
Produces Table 4 (OLS vs LASSO) and Figure 5 (feature importance bar chart).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.linear_model    import LinearRegression, LassoCV, Lasso
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from scipy                   import stats

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(ROOT, "data", "processed")
FIGURES = os.path.join(ROOT, "outputs", "figures")
RESULTS = os.path.join(ROOT, "outputs", "results")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(os.path.join(PROC, "merged_dataset_2023.csv"))

PRED_COLS = [
    "corruption_control", "rule_of_law", "govt_effectiveness",
    "regulatory_quality", "cpi_score", "bank_branches_per100k",
    "financial_account_pct", "gdp_log"
]
LABELS = {
    "corruption_control":     "WGI Control of Corruption",
    "rule_of_law":            "WGI Rule of Law",
    "govt_effectiveness":     "WGI Govt Effectiveness",
    "regulatory_quality":     "WGI Regulatory Quality",
    "cpi_score":              "TI CPI Score",
    "bank_branches_per100k":  "Bank Branches / 100k",
    "financial_account_pct":  "Financial Account %",
    "gdp_log":                "GDP per Capita (log)",
}

plt.rcParams.update({"font.family": "DejaVu Serif", "figure.dpi": 150,
                     "axes.titlesize": 11, "axes.labelsize": 10})

X = df[PRED_COLS].values
y = df["basel_score"].values

scaler  = StandardScaler()
X_std   = scaler.fit_transform(X)

# ── OLS Baseline ─────────────────────────────────────────────────────────────
ols = LinearRegression()
ols.fit(X_std, y)
y_pred_ols = ols.predict(X_std)

ols_r2   = r2_score(y, y_pred_ols)
ols_adjr2 = 1 - (1 - ols_r2) * (len(y) - 1) / (len(y) - len(PRED_COLS) - 1)
ols_rmse  = np.sqrt(mean_squared_error(y, y_pred_ols))
ols_mae   = mean_absolute_error(y, y_pred_ols)

# OLS cross-validated RMSE
ols_cv = cross_val_score(ols, X_std, y, cv=10, scoring="neg_root_mean_squared_error")
ols_cv_rmse = -ols_cv.mean()

print("=== OLS Baseline ===")
print(f"  R²       : {ols_r2:.4f}")
print(f"  Adj R²   : {ols_adjr2:.4f}")
print(f"  RMSE     : {ols_rmse:.4f}")
print(f"  MAE      : {ols_mae:.4f}")
print(f"  10-fold CV RMSE: {ols_cv_rmse:.4f}")

# Breusch-Pagan test (heteroskedasticity)
residuals = y - y_pred_ols
kt = stats.kendalltau(y_pred_ols, residuals**2)
bp_pval = kt.pvalue
sw_stat, sw_pval = stats.shapiro(residuals[:50])  # Shapiro-Wilk (use first 50 for power)
print(f"\n  Shapiro-Wilk (residual normality) p = {sw_pval:.4f} "
      f"({'NORMAL' if sw_pval > 0.05 else 'NON-NORMAL'})")

# VIF
from numpy.linalg import inv
corr_X = np.corrcoef(X_std.T)
vif = np.diag(inv(corr_X))
print("\n  VIF values:")
for col, v in zip(PRED_COLS, vif):
    flag = " *** HIGH" if v > 10 else ""
    print(f"    {LABELS[col]}: {v:.2f}{flag}")

# ── LASSO with 10-fold CV ─────────────────────────────────────────────────────
lasso_cv = LassoCV(cv=10, random_state=42, max_iter=20000, n_alphas=200)
lasso_cv.fit(X_std, y)
best_alpha = lasso_cv.alpha_

lasso = Lasso(alpha=best_alpha, max_iter=20000, random_state=42)
lasso.fit(X_std, y)
y_pred_lasso = lasso.predict(X_std)

lasso_r2    = r2_score(y, y_pred_lasso)
lasso_adjr2 = 1 - (1 - lasso_r2) * (len(y) - 1) / (len(y) - len(PRED_COLS) - 1)
lasso_rmse  = np.sqrt(mean_squared_error(y, y_pred_lasso))
lasso_mae   = mean_absolute_error(y, y_pred_lasso)
lasso_cv_rmse = -cross_val_score(lasso, X_std, y, cv=10,
                                  scoring="neg_root_mean_squared_error").mean()

nonzero = sum(lasso.coef_ != 0)
print(f"\n=== LASSO (lambda = {best_alpha:.4f}) ===")
print(f"  Non-zero predictors: {nonzero} of {len(PRED_COLS)}")
print(f"  R²       : {lasso_r2:.4f}")
print(f"  Adj R²   : {lasso_adjr2:.4f}")
print(f"  RMSE     : {lasso_rmse:.4f}")
print(f"  MAE      : {lasso_mae:.4f}")
print(f"  10-fold CV RMSE: {lasso_cv_rmse:.4f}")

print("\n  LASSO Coefficients (standardised):")
coef_df = pd.DataFrame({
    "predictor": PRED_COLS,
    "label":     [LABELS[c] for c in PRED_COLS],
    "coef":      lasso.coef_,
    "abs_coef":  np.abs(lasso.coef_),
}).sort_values("abs_coef", ascending=False)
print(coef_df[["label","coef"]].to_string(index=False))
coef_df.to_csv(os.path.join(RESULTS, "rq1_lasso_coefficients.csv"), index=False)

# ── TABLE 4: Model comparison ────────────────────────────────────────────────
table4 = pd.DataFrame({
    "Metric":        ["R²", "Adjusted R²", "RMSE (in-sample)", "MAE (in-sample)", "10-fold CV RMSE"],
    "OLS Baseline":  [f"{ols_r2:.4f}", f"{ols_adjr2:.4f}", f"{ols_rmse:.4f}",
                      f"{ols_mae:.4f}", f"{ols_cv_rmse:.4f}"],
    "LASSO":         [f"{lasso_r2:.4f}", f"{lasso_adjr2:.4f}", f"{lasso_rmse:.4f}",
                      f"{lasso_mae:.4f}", f"{lasso_cv_rmse:.4f}"],
})
print("\n=== TABLE 4: OLS vs LASSO ===")
print(table4.to_string(index=False))
table4.to_csv(os.path.join(RESULTS, "table4_ols_vs_lasso.csv"), index=False)

# ── FIGURE 5: LASSO Feature Importance ───────────────────────────────────────
active = coef_df[coef_df["abs_coef"] > 0].copy()
colors = ["#d7191c" if c > 0 else "#2c7bb6" for c in active["coef"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(active["label"], active["coef"], color=colors, edgecolor="white",
               height=0.6)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardised LASSO Coefficient", fontsize=10)
ax.set_title(
    f"Figure 5. LASSO Feature Importance (lambda = {best_alpha:.4f})\n"
    f"Red = higher coefficient raises AML risk. Blue = lowers AML risk. "
    f"{nonzero} of {len(PRED_COLS)} predictors retained after regularisation.",
    fontsize=9
)
ax.tick_params(labelsize=9)
for bar, val in zip(bars, active["coef"]):
    ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure5_lasso_importance.png"), bbox_inches="tight")
plt.close()
print("\nFigure 5 saved.")
print("RQ1 complete.")
