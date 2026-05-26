"""
QM640 Capstone: Notebook 05: RQ2: Multi-Class AML Risk Classification
Produces Table 5 (model comparison) and Figures 6-7 (confusion matrix, RF importance).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (accuracy_score, f1_score, precision_score,
                                      recall_score, confusion_matrix,
                                      roc_auc_score, classification_report)
from xgboost import XGBClassifier

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(ROOT, "data", "processed")
FIGURES = os.path.join(ROOT, "outputs", "figures")
RESULTS = os.path.join(ROOT, "outputs", "results")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Serif", "figure.dpi": 150,
                     "axes.titlesize": 11, "axes.labelsize": 10})

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

# ── Bin Basel score into 3 risk classes (tertile-based for balanced classes) ──
t33 = df["basel_score"].quantile(0.333)
t67 = df["basel_score"].quantile(0.667)
print(f"Tertile thresholds: Low < {t33:.2f}, Medium {t33:.2f}-{t67:.2f}, High > {t67:.2f}")
df["risk_class"] = pd.cut(df["basel_score"],
                           bins=[-np.inf, t33, t67, np.inf],
                           labels=[0, 1, 2]).astype(int)
CLASS_THRESHOLDS = (t33, t67)
CLASS_NAMES = ["Low", "Medium", "High"]

print("=== Class Distribution ===")
for i, name in enumerate(CLASS_NAMES):
    n = (df["risk_class"] == i).sum()
    print(f"  {name}: n={n} ({100*n/len(df):.1f}%)")

X = df[PRED_COLS].values
y = df["risk_class"].values

scaler  = StandardScaler()
X_std   = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   min_samples_split=5, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                          random_state=42, eval_metric="mlogloss",
                                          verbosity=0),
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Cross-val F1 on training set
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf,
                             scoring="f1_macro").mean()

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="macro")
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    auc   = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    cm    = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    prec_per  = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_test, y_pred, average=None, zero_division=0)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "cv_f1": cv_f1, "acc": acc, "f1": f1, "prec": prec,
        "rec": rec, "auc": auc, "cm": cm,
        "prec_per": prec_per, "rec_per": rec_per,
    }
    print(f"\n{name}:")
    print(f"  5-fold CV F1 (macro): {cv_f1:.4f}")
    print(f"  Test Accuracy: {acc:.4f}  F1 (macro): {f1:.4f}  AUC: {auc:.4f}")
    print(f"  Per-class Precision: {[f'{p:.3f}' for p in prec_per]}")
    print(f"  Per-class Recall:    {[f'{r:.3f}' for r in rec_per]}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

# ── TABLE 5: Model comparison ─────────────────────────────────────────────────
rows = []
for name, r in results.items():
    row = {
        "Model":         name,
        "CV F1 (macro)": f"{r['cv_f1']:.4f}",
        "Test Accuracy": f"{r['acc']:.4f}",
        "F1 (macro)":    f"{r['f1']:.4f}",
        "AUC-ROC":       f"{r['auc']:.4f}",
        "Prec Low":      f"{r['prec_per'][0]:.3f}",
        "Prec Med":      f"{r['prec_per'][1]:.3f}",
        "Prec High":     f"{r['prec_per'][2]:.3f}",
        "Rec Low":       f"{r['rec_per'][0]:.3f}",
        "Rec Med":       f"{r['rec_per'][1]:.3f}",
        "Rec High":      f"{r['rec_per'][2]:.3f}",
    }
    rows.append(row)

table5 = pd.DataFrame(rows)
print("\n=== TABLE 5: Classification Performance ===")
print(table5.to_string(index=False))
table5.to_csv(os.path.join(RESULTS, "table5_classification_performance.csv"), index=False)

# Best model by F1
best_name = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

# ── FIGURE 6: Confusion matrices (all 3 models) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm_pct = r["cm"].astype(float) / r["cm"].sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar=False, annot_kws={"size": 10})
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(f"{name}\nF1={r['f1']:.3f}, AUC={r['auc']:.3f}", fontsize=9)

fig.suptitle(
    "Figure 6. Confusion Matrices: AML Risk Tier Classification (Test Set, % of Actual Class)\n"
    f"Test set: n={len(y_test)} countries. Low = Basel 0-3.33, Medium = 3.33-6.67, High = 6.67-10.",
    fontsize=9, y=1.05
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure6_confusion_matrices.png"), bbox_inches="tight")
plt.close()
print("Figure 6 saved.")

# ── FIGURE 7: Random Forest Feature Importance ───────────────────────────────
rf = results["Random Forest"]["model"]
importances = rf.feature_importances_
fi_df = pd.DataFrame({
    "label":      [LABELS[c] for c in PRED_COLS],
    "importance": importances
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(fi_df["label"], fi_df["importance"],
               color="#2c7bb6", edgecolor="white", height=0.6)
ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=10)
ax.set_title(
    "Figure 7. Random Forest Feature Importance\n"
    "Higher values indicate greater contribution to correct classification of AML risk tiers.",
    fontsize=9
)
for bar, val in zip(bars, fi_df["importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure7_rf_importance.png"), bbox_inches="tight")
plt.close()
print("Figure 7 saved.")

# ── FIGURE 4: Multi-class ROC curves (One-vs-Rest) ───────────────────────────
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sk_auc

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
CLASS_COLORS = ["#2c7bb6", "#756bb1", "#d7191c"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, r) in zip(axes, results.items()):
    y_prob = r["y_proba"]
    for cls_i, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, cls_i], y_prob[:, cls_i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{cls_name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(f"{name}", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)

macro_auc = results["Random Forest"]["auc"]
fig.suptitle(
    f"Figure 4. One-vs-Rest ROC Curves: AML Risk Tier Classification (Test Set, n={len(y_test)})\n"
    f"Best model (Random Forest) macro AUC = {macro_auc:.3f}",
    fontsize=9, y=1.04
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure4_roc_curves.png"), bbox_inches="tight")
plt.close()
print("Figure 4 saved.")

print("\nRQ2 complete.")
