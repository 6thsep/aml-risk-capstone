"""
QM640 Capstone: Notebook 07: RQ4: Country Typology Discovery via Clustering
K-Means + Hierarchical clustering on all 8 predictors.
Produces Figure 9 (elbow + silhouette), Figure 10 (cluster radar chart),
Figure 11 (dendrogram) and Table 7 (cluster profiles).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from sklearn.preprocessing      import StandardScaler
from sklearn.cluster            import KMeans, AgglomerativeClustering
from sklearn.metrics            import silhouette_score
from scipy.cluster.hierarchy    import dendrogram, linkage
from scipy.spatial.distance     import pdist, squareform

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
    "corruption_control":     "WGI Corruption Control",
    "rule_of_law":            "WGI Rule of Law",
    "govt_effectiveness":     "WGI Govt Effectiveness",
    "regulatory_quality":     "WGI Regulatory Quality",
    "cpi_score":              "TI CPI Score",
    "bank_branches_per100k":  "Bank Branches/100k",
    "financial_account_pct":  "Financial Account %",
    "gdp_log":                "GDP per Capita (log)",
}

scaler  = StandardScaler()
X_std   = scaler.fit_transform(df[PRED_COLS].values)

# ── FIGURE 9: Elbow plot + Silhouette scores ──────────────────────────────────
wcss       = []
sil_scores = []
K_range    = range(2, 7)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_std)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_std, labels))
    print(f"k={k}: WCSS={km.inertia_:.1f}, Silhouette={sil_scores[-1]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(list(K_range), wcss, marker='o', color="#2c7bb6", linewidth=2)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Within-Cluster Sum of Squares")
ax1.set_title("Elbow Plot")
ax1.set_xticks(list(K_range))

ax2.plot(list(K_range), sil_scores, marker='s', color="#d7191c", linewidth=2)
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Average Silhouette Score")
ax2.set_title("Silhouette Analysis")
ax2.set_xticks(list(K_range))

best_k    = list(K_range)[np.argmax(sil_scores)]
best_sil  = max(sil_scores)
ax2.axvline(best_k, color="gray", linestyle="--", linewidth=1, label=f"Optimal k={best_k}")
ax2.legend(fontsize=9)

fig.suptitle(
    f"Figure 9. K-Means Cluster Selection: Elbow Plot and Silhouette Analysis\n"
    f"Optimal k = {best_k} (silhouette = {best_sil:.4f})",
    fontsize=9, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure9_cluster_selection.png"), bbox_inches="tight")
plt.close()
print(f"\nOptimal k = {best_k}, Silhouette = {best_sil:.4f}")
print("Figure 9 saved.")

# ── Final K-Means with optimal k ─────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = km_final.fit_predict(X_std)

print(f"\nCluster sizes:")
for c in range(best_k):
    n_c = (df["cluster"] == c).sum()
    print(f"  Cluster {c+1}: n={n_c} ({100*n_c/len(df):.1f}%)")

# Cluster profiles (raw means)
profile = df.groupby("cluster")[PRED_COLS + ["basel_score"]].mean().round(3)
profile.index = [f"Cluster {i+1}" for i in range(best_k)]
print(f"\nCluster profiles (raw means):")
print(profile.T.to_string())
profile.to_csv(os.path.join(RESULTS, "table7_cluster_profiles.csv"))

# Name clusters based on Basel score ranking
cluster_basel = df.groupby("cluster")["basel_score"].mean().sort_values()
cluster_names = {}
tier_labels = ["Low-Risk Governance Leaders", "Moderate-Risk Mixed Profiles", "High-Risk Fragile States"]
if best_k == 3:
    for rank, (cid, _) in enumerate(cluster_basel.items()):
        cluster_names[cid] = tier_labels[rank]
elif best_k == 4:
    tier_labels4 = ["Low-Risk Governance Leaders", "Medium-Low Risk",
                    "Medium-High Risk", "High-Risk Fragile States"]
    for rank, (cid, _) in enumerate(cluster_basel.items()):
        cluster_names[cid] = tier_labels4[rank]
else:
    for rank, (cid, _) in enumerate(cluster_basel.items()):
        cluster_names[cid] = f"Cluster {cid+1}"

df["cluster_name"] = df["cluster"].map(cluster_names)
print("\nCluster labels assigned:")
for cid, name in cluster_names.items():
    n_c = (df["cluster"] == cid).sum()
    avg = df.loc[df["cluster"] == cid, "basel_score"].mean()
    print(f"  {name}: n={n_c}, avg Basel score={avg:.2f}")

# Sample countries per cluster
print("\nSample countries per cluster:")
for cid, name in cluster_names.items():
    countries = df.loc[df["cluster"] == cid, "country_name"].head(5).tolist()
    print(f"  {name}: {', '.join(countries)}")

# ── FIGURE 10: Radar chart of cluster profiles ───────────────────────────────
# Normalise profiles to 0-1 range for radar
profile_raw = df.groupby("cluster")[PRED_COLS].mean()
profile_norm = (profile_raw - profile_raw.min()) / (profile_raw.max() - profile_raw.min())

categories = [LABELS[c] for c in PRED_COLS]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

COLORS = ["#2c7bb6", "#abdda4", "#d7191c", "#756bb1"][:best_k]

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
for idx, (cid, name) in enumerate(cluster_names.items()):
    values = profile_norm.loc[cid, PRED_COLS].tolist()
    values += values[:1]
    ax.plot(angles, values, color=COLORS[idx], linewidth=2, label=name)
    ax.fill(angles, values, color=COLORS[idx], alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=7)
ax.set_title(
    f"Figure 10. Cluster Profiles: Radar Chart (k = {best_k})\n"
    "Values normalised 0-1. Higher = stronger governance / financial access.",
    fontsize=9, pad=20
)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure10_cluster_radar.png"), bbox_inches="tight")
plt.close()
print("Figure 10 saved.")

# ── FIGURE 11: Dendrogram (Hierarchical clustering, Ward linkage) ─────────────
Z = linkage(X_std, method="ward")

fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(
    Z, ax=ax,
    labels=df["country_name"].tolist(),
    leaf_rotation=90,
    leaf_font_size=5,
    color_threshold=Z[-best_k+1, 2],
    above_threshold_color="gray"
)
ax.set_title(
    f"Figure 11. Hierarchical Clustering Dendrogram (Ward Linkage, n = {len(df)} countries)\n"
    f"Horizontal dashed line shows the cut for k = {best_k} clusters.",
    fontsize=9
)
ax.axhline(Z[-best_k+1, 2], color="red", linestyle="--", linewidth=1.2,
           label=f"Cut for k={best_k}")
ax.set_xlabel("Country", fontsize=9)
ax.set_ylabel("Ward Linkage Distance", fontsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "figure11_dendrogram.png"), bbox_inches="tight")
plt.close()
print("Figure 11 saved.")

# ── TABLE 7: Full cluster profile ─────────────────────────────────────────────
named_profile = df.groupby("cluster_name")[PRED_COLS + ["basel_score"]].mean().round(3)
named_profile.index.name = "Cluster"
named_profile.to_csv(os.path.join(RESULTS, "table7_cluster_profiles_named.csv"))
print("\n=== TABLE 7: Cluster Profiles ===")
print(named_profile.T.to_string())
print(f"\nRQ4 complete. Optimal k = {best_k}, Silhouette = {best_sil:.4f}")
