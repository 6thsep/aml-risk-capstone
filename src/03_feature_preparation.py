"""
QM640 Capstone: Notebook 03: Feature Preparation & Multicollinearity Check
Validates sample size, checks VIF, ranks predictor correlations with outcome,
and documents the tertile thresholds used for risk class labels.
Produces table3b_vif.csv and table3c_correlations.csv.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "outputs", "results")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(os.path.join(PROC, "merged_dataset_2023.csv"))
print(f"Dataset: {df.shape[0]} countries, {df.shape[1]} columns\n")

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

# ── 1. Sample Size Validation (Green's Rule) ─────────────────────────────────
n = len(df)
k = len(PRED_COLS)
green_threshold = 50 + 8 * k
print("=== Sample Size Validation ===")
print(f"  Predictors (k): {k}")
print(f"  Green's Rule threshold: n ≥ 50 + 8×{k} = {green_threshold}")
print(f"  Available: n = {n}")
print(f"  Status: {'✅ SUFFICIENT' if n >= green_threshold else '❌ INSUFFICIENT'}\n")

# ── 2. VIF: Variance Inflation Factor ───────────────────────────────────────
scaler = StandardScaler()
X_std  = scaler.fit_transform(df[PRED_COLS].values)

corr_X = np.corrcoef(X_std.T)
vif    = np.diag(inv(corr_X))

print("=== VIF Check (multicollinearity) ===")
vif_rows = []
for col, v in zip(PRED_COLS, vif):
    flag = "HIGH ⚠️" if v > 10 else "OK"
    print(f"  {LABELS[col]:<35}: VIF = {v:.2f}  [{flag}]")
    vif_rows.append({"Predictor": LABELS[col], "VIF": round(v, 3),
                     "Status": "HIGH" if v > 10 else "OK"})

vif_df = pd.DataFrame(vif_rows)
vif_df.to_csv(os.path.join(RESULTS, "table3b_vif.csv"), index=False)
print(f"\n  Max VIF: {vif.max():.2f}: {'multicollinearity concern, LASSO handles via regularisation' if vif.max() > 10 else 'no severe multicollinearity'}")
print("table3b_vif.csv saved.\n")

# ── 3. Predictor–Outcome Correlations ────────────────────────────────────────
print("=== Pearson Correlations with Basel AML Score ===")
corr_rows = []
for col in PRED_COLS:
    r = df[col].corr(df["basel_score"])
    corr_rows.append({"Predictor": LABELS[col], "r": round(r, 4)})
    direction = "↑ higher score" if r > 0 else "↓ lower score"
    print(f"  {LABELS[col]:<35}: r = {r:+.4f}  ({direction} = {'more' if r > 0 else 'less'} AML risk)")

corr_df = pd.DataFrame(corr_rows).sort_values("r", key=abs, ascending=False)
corr_df.to_csv(os.path.join(RESULTS, "table3c_correlations.csv"), index=False)
print("\ntable3c_correlations.csv saved.\n")

# ── 4. Risk Class Tertile Thresholds ─────────────────────────────────────────
t33 = df["basel_score"].quantile(0.333)
t67 = df["basel_score"].quantile(0.667)
df["risk_class"] = pd.cut(df["basel_score"],
                           bins=[-np.inf, t33, t67, np.inf],
                           labels=["Low", "Medium", "High"])

print("=== Risk Class Tertile Thresholds ===")
print(f"  Low    : Basel score < {t33:.3f}")
print(f"  Medium : {t33:.3f} ≤ Basel score < {t67:.3f}")
print(f"  High   : Basel score ≥ {t67:.3f}")
print()
print("  Class distribution:")
for cls in ["Low", "Medium", "High"]:
    n_cls = (df["risk_class"] == cls).sum()
    print(f"    {cls:<8}: n = {n_cls:3d}  ({100 * n_cls / n:.1f}%)")

thresholds_df = pd.DataFrame([
    {"Class": "Low",    "Upper Bound": f"< {t33:.3f}"},
    {"Class": "Medium", "Upper Bound": f"{t33:.3f} – {t67:.3f}"},
    {"Class": "High",   "Upper Bound": f"≥ {t67:.3f}"},
])
thresholds_df.to_csv(os.path.join(RESULTS, "table3d_risk_thresholds.csv"), index=False)
print("\ntable3d_risk_thresholds.csv saved.")

print("\nNotebook 03 complete.")
