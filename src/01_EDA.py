"""
Analyse Exploratoire des Données (EDA)
Projet : Maintenance Prédictive Industrielle
Variable cible : failure_within_24h (Classification Binaire)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("../outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("=" * 60)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 60)

df = pd.read_csv("../data/predictive_maintenance_v3.csv")
print(f"Shape : {df.shape}")
print(f"Colonnes : {df.columns.tolist()}")
print(df.head(3))

# ─────────────────────────────────────────────
# 2. OBESRVATION DES DONNEES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. OBESRVATION DES DONNEES")
print("=" * 60)

print("\n--- Types ---")
print(df.dtypes)

print("\n--- Statistiques ---")
print(df.describe())

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Count": missing, "Percent (%)": missing_pct})
print(missing_df[missing_df["Count"] > 0])

# ─────────────────────────────────────────────
# 3. DISTRIBUTION DE LA VARIABLE CIBLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DISTRIBUTION DE LA VARIABLE CIBLE")
print("=" * 60)

target_counts = df["failure_within_24h"].value_counts()
target_pct = (target_counts / len(df) * 100).round(2)
print(target_counts)
print(f"\nRatio déséquilibre : {target_counts[0] / target_counts[1]:.2f}:1")
print(f"Classe 0 (pas de panne) : {target_pct[0]}%")
print(f"Classe 1 (panne) : {target_pct[1]}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribution de la Variable Cible : failure_within_24h", fontsize=14, fontweight="bold")

colors = ["#0392cf", "#f37736"]
axes[0].bar(["Pas de panne (0)", "Panne dans 24h (1)"], target_counts.values, color=colors, edgecolor="black")
axes[0].set_title("Décompte")
axes[0].set_ylabel("Nombre d'observations")
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

axes[1].pie(
    target_counts.values,
    labels=["Pas de panne\n(83.5%)", "Panne dans 24h\n(14.8%)"],
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    explode=(0, 0.08),
)
axes[1].set_title("Proportions")

plt.tight_layout()
plt.savefig("../outputs/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("--> Fig. save : 01_target_distribution.png")

# ─────────────────────────────────────────────
# 4. DISTRIBUTION DES VARIABLES NUMÉRIQUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. DISTRIBUTION DES VARIABLES NUMÉRIQUES")
print("=" * 60)

numeric_features = [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp", "rul_hours"
]

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle("Distribution des Variables Numériques par Classe Cible", fontsize=14, fontweight="bold")

for i, feat in enumerate(numeric_features):
    ax_hist = axes[i // 2][i % 2 * 2]
    ax_box = axes[i // 2][i % 2 * 2 + 1]

    df[df["failure_within_24h"] == 0][feat].dropna().plot(
        kind="hist", bins=40, alpha=0.6, color="#0392cf", label="Pas de panne", ax=ax_hist
    )
    df[df["failure_within_24h"] == 1][feat].dropna().plot(
        kind="hist", bins=40, alpha=0.6, color="#f37736", label="Panne", ax=ax_hist
    )
    ax_hist.set_title(f"{feat}", fontsize=9)
    ax_hist.legend(fontsize=7)
    ax_hist.set_xlabel("")

    df.boxplot(column=feat, by="failure_within_24h", ax=ax_box)
    ax_box.set_title(f"{feat} (boxplot)", fontsize=9)
    ax_box.set_xlabel("")

plt.tight_layout()
plt.savefig("../outputs/02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig. save : 02_numeric_distributions.png")

# ─────────────────────────────────────────────
# 5. VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. VARIABLES CATÉGORIELLES")
print("=" * 60)

cat_features = ["machine_type", "operating_mode", "failure_type"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Variables Catégorielles — Taux de Panne par Modalité", fontsize=13, fontweight="bold")

for i, feat in enumerate(cat_features):
    failure_rate = df.groupby(feat)["failure_within_24h"].mean().sort_values(ascending=False)
    bars = axes[i].bar(failure_rate.index, failure_rate.values * 100, color=["#f37736", "#e67e22", "#3498db", "#0392cf", "#9b59b6"][:len(failure_rate)])
    axes[i].set_title(f"Taux de panne par {feat}", fontsize=10)
    axes[i].set_ylabel("Taux de panne (%)")
    axes[i].set_xticklabels(failure_rate.index, rotation=20, ha="right")
    for bar in bars:
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("../outputs/03_categorical_failure_rates.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig. save : 03_categorical_failure_rates.png")

# ─────────────────────────────────────────────
# 6. MATRICE DE CORRÉLATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. MATRICE DE CORRÉLATION")
print("=" * 60)

corr_cols = numeric_features + ["failure_within_24h"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    annot_kws={"size": 9}
)
ax.set_title("Matrice de Corrélation — Variables Numériques", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/04_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig. 4 save : 04_correlation_matrix.png")

# Corrélations avec la cible
print("\n--- Corrélations avec failure_within_24h ---")
target_corr = corr_matrix["failure_within_24h"].drop("failure_within_24h").sort_values(key=abs, ascending=False)
print(target_corr)

# ─────────────────────────────────────────────
# 7. ANALYSE DES VALEURS ABERRANTES (METHODE IQR)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. VALEURS ABERRANTES")
print("=" * 60)

for feat in numeric_features:
    q1 = df[feat].quantile(0.25)
    q3 = df[feat].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
    print(f"  {feat:30s} : {outliers:5d} outliers ({outliers / len(df) * 100:.2f}%)")

# ─────────────────────────────────────────────
# 8. ÉVOLUTION TEMPORELLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. ANALYSE TEMPORELLE")
print("=" * 60)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M")

monthly_failure = df.groupby("month")["failure_within_24h"].mean() * 100

fig, ax = plt.subplots(figsize=(14, 5))
monthly_failure.plot(ax=ax, color="#f37736", marker="o", linewidth=2, markersize=5)
ax.set_title("Taux de Panne Mensuel (%)", fontsize=13, fontweight="bold")
ax.set_ylabel("Taux de panne (%)")
ax.set_xlabel("Mois")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/05_temporal_failure_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig. save : 05_temporal_failure_rate.png")

print("\n" + "=" * 60)
print("EDA TERMINÉE ")
print("=" * 60)
