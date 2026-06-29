"""
04_evaluation.py — Évaluation Comparative des Modèles + Interprétabilité
Projet : Maintenance Prédictive Industrielle
Variable cible : failure_within_24h

Contenu :
  - Métriques complètes sur le test set (Accuracy, F1, ROC-AUC, Recall, PR-AUC)
  - Matrices de confusion
  - Courbes ROC et Precision-Recall
  - Optimisation du seuil de décision
  - Feature Importance (native + permutation)
  - SHAP sur le meilleur modèle
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import warnings
import os

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score, precision_score,
    confusion_matrix, classification_report, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
os.makedirs("../outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉVALUATION — Maintenance Prédictive")
print("=" * 60)

X_train = np.load("../models/X_train_proc.npy")
X_test = np.load("../models/X_test_proc.npy")
y_train = np.load("../models/y_train.npy")
y_test = np.load("../models/y_test.npy")
feature_names = joblib.load("../models/feature_names.pkl")
cv_results = joblib.load("../models/cv_results.pkl")

model_names = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP"]
models = {name: joblib.load(f"../models/{name}.pkl") for name in model_names}

print(f"Modèles chargés : {list(models.keys())}")
print(f"Test set : {X_test.shape} | Positifs : {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. MÉTRIQUES SUR LE TEST SET
# ─────────────────────────────────────────────
print("\n[1] Calcul des métriques sur le test set (seuil 0.5)...")

results = {}
proba_dict = {}

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    proba_dict[name] = y_proba

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }

results_df = pd.DataFrame(results).T.round(4)
results_df.to_csv("../outputs/model_comparison.csv")
print("\n--- Tableau comparatif (Test Set, seuil 0.5) ---")
print(results_df.to_string())

# ─────────────────────────────────────────────
# 3. MATRICES DE CONFUSION
# ─────────────────────────────────────────────
print("\n[2] Génération des matrices de confusion...")

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Matrices de Confusion — Test Set (seuil 0.5)", fontsize=13, fontweight="bold")

for i, (name, model) in enumerate(models.items()):
    y_proba = proba_dict[name]
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Prédit 0", "Prédit 1"],
                yticklabels=["Réel 0", "Réel 1"])
    axes[i].set_title(f"{name}\nF1={results[name]['F1-Score']:.3f} | Recall={results[name]['Recall']:.3f}", fontsize=9)

plt.tight_layout()
plt.savefig("../outputs/06_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 06_confusion_matrices.png")

# ─────────────────────────────────────────────
# 4. COURBES ROC ET PRECISION-RECALL
# ─────────────────────────────────────────────
print("\n[3] Courbes ROC et Precision-Recall...")

colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Courbes ROC et Precision-Recall — Test Set", fontsize=13, fontweight="bold")

for i, (name, color) in enumerate(zip(model_names, colors)):
    y_proba = proba_dict[name]
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]["ROC-AUC"]
    axes[0].plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})", lw=2)

    # PR
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = results[name]["PR-AUC"]
    axes[1].plot(rec, prec, color=color, label=f"{name} (PR-AUC={pr_auc:.3f})", lw=2)

axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
axes[0].set_xlabel("Taux de Faux Positifs")
axes[0].set_ylabel("Taux de Vrais Positifs (Recall)")
axes[0].set_title("Courbe ROC")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

baseline_pr = y_test.mean()
axes[1].axhline(y=baseline_pr, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline_pr:.3f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Courbe Precision-Recall")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../outputs/07_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 07_roc_pr_curves.png")

# ─────────────────────────────────────────────
# 5. OPTIMISATION DU SEUIL DE DÉCISION
# ─────────────────────────────────────────────
print("\n[4] Optimisation du seuil de décision (Gradient Boosting)...")

# On optimise sur le meilleur modèle identifié
best_model_name = results_df["ROC-AUC"].idxmax()
print(f"   Meilleur modèle (ROC-AUC) : {best_model_name}")

y_proba_best = proba_dict[best_model_name]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores, recall_scores, precision_scores = [], [], []

for t in thresholds:
    y_pred_t = (y_proba_best >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_t, zero_division=0))
    precision_scores.append(precision_score(y_test, y_pred_t, zero_division=0))

best_t_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_t_idx]
print(f"   Seuil optimal (F1) : {best_threshold:.2f}")
print(f"   F1 @ seuil 0.50 : {f1_score(y_test, (y_proba_best >= 0.5).astype(int)):.4f}")
print(f"   F1 @ seuil {best_threshold:.2f} : {f1_scores[best_t_idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, f1_scores, label="F1-Score", color="#e74c3c", lw=2)
ax.plot(thresholds, recall_scores, label="Recall", color="#3498db", lw=2)
ax.plot(thresholds, precision_scores, label="Precision", color="#2ecc71", lw=2)
ax.axvline(best_threshold, color="black", linestyle="--", label=f"Seuil optimal = {best_threshold:.2f}")
ax.axvline(0.5, color="gray", linestyle=":", label="Seuil default = 0.50")
ax.set_title(f"Optimisation du Seuil — {best_model_name}", fontsize=12, fontweight="bold")
ax.set_xlabel("Seuil de Décision")
ax.set_ylabel("Score")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/08_threshold_optimization.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 08_threshold_optimization.png")

# Sauvegarde du seuil optimal
joblib.dump({"best_model": best_model_name, "optimal_threshold": best_threshold}, "../models/optimal_threshold.pkl")

# ─────────────────────────────────────────────
# 6. COMPARAISON VISUELLE DES MODÈLES
# ─────────────────────────────────────────────
print("\n[5] Graphique comparatif des modèles...")

metrics_plot = ["Recall", "Precision", "F1-Score", "ROC-AUC", "PR-AUC"]
x = np.arange(len(model_names))
width = 0.15
colors_bar = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]

fig, ax = plt.subplots(figsize=(14, 6))

for j, (metric, color) in enumerate(zip(metrics_plot, colors_bar)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + j * width, vals, width, label=metric, color=color, alpha=0.85, edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("Comparaison des Modèles — Métriques sur Test Set", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/09_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 09_model_comparison.png")

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (Random Forest & Gradient Boosting)
# ─────────────────────────────────────────────
print("\n[6] Feature Importance native (RF et GB)...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Feature Importance Native (Top 15)", fontsize=13, fontweight="bold")

for ax, model_name in zip(axes, ["RandomForest", "GradientBoosting"]):
    model = models[model_name]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    top_feats = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    bars = ax.barh(top_feats[::-1], top_vals[::-1], color="#3498db", edgecolor="black", alpha=0.8)
    ax.set_title(f"{model_name}", fontsize=11)
    ax.set_xlabel("Importance")
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("../outputs/10_feature_importance_native.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 10_feature_importance_native.png")

# ─────────────────────────────────────────────
# 8. PERMUTATION IMPORTANCE (Logistic Regression)
# ─────────────────────────────────────────────
print("\n[7] Permutation Importance (LogisticRegression)...")

perm_imp = permutation_importance(
    models["LogisticRegression"], X_test, y_test,
    n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
)

perm_indices = np.argsort(perm_imp.importances_mean)[::-1][:15]
perm_feats = [feature_names[i] for i in perm_indices]
perm_vals = perm_imp.importances_mean[perm_indices]
perm_std = perm_imp.importances_std[perm_indices]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(perm_feats[::-1], perm_vals[::-1], xerr=perm_std[::-1],
        color="#e74c3c", edgecolor="black", alpha=0.8, capsize=4)
ax.set_title("Permutation Importance — Logistic Regression (Top 15)", fontsize=11, fontweight="bold")
ax.set_xlabel("Baisse ROC-AUC (moyenne ± std)")
plt.tight_layout()
plt.savefig("../outputs/11_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 11_permutation_importance.png")

# ─────────────────────────────────────────────
# 9. SHAP — MEILLEUR MODÈLE
# ─────────────────────────────────────────────
print(f"\n[8] SHAP sur le meilleur modèle : {best_model_name}...")

best_model = models[best_model_name]

# Conversion en liste Python pure — évite l'erreur d'indexation avec shap
feature_names_list = list(feature_names)

# Sous-échantillon pour SHAP (performance)
np.random.seed(42)
shap_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
X_shap = X_test[shap_idx]

if best_model_name in ["RandomForest", "GradientBoosting"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)
    sv = np.array(shap_values)
    # Selon la version de shap/sklearn : liste [class0, class1] ou array 3D (n, f, 2)
    if sv.ndim == 3:
        shap_vals = sv[:, :, 1]
    elif isinstance(shap_values, list):
        shap_vals = np.array(shap_values[1])
    else:
        shap_vals = sv
elif best_model_name == "LogisticRegression":
    explainer = shap.LinearExplainer(best_model, X_shap)
    shap_vals = np.array(explainer.shap_values(X_shap))
else:
    explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_shap, 50))
    sv = np.array(explainer.shap_values(X_shap))
    shap_vals = sv[:, :, 1] if sv.ndim == 3 else sv

# SHAP Summary Plot (beeswarm)
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_names_list, show=False, max_display=15)
plt.title(f"SHAP Summary Plot — {best_model_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/12_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 12_shap_summary.png")

# SHAP Bar Plot (importance globale)
fig = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_names_list,
                  plot_type="bar", show=False, max_display=15)
plt.title(f"SHAP Feature Importance Globale — {best_model_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/13_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 13_shap_bar.png")

# Sauvegarde des SHAP values
joblib.dump({"shap_values": shap_vals, "X_shap": X_shap, "feature_names": feature_names_list}, "../models/shap_data.pkl")

# ─────────────────────────────────────────────
# 10. RAPPORT FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RAPPORT FINAL — COMPARAISON DES MODÈLES")
print("=" * 60)

print("\nMétriques Test Set :")
print(results_df.to_string())

print(f"\nMeilleur modèle (ROC-AUC) : {best_model_name}")
print(f"  ROC-AUC  : {results[best_model_name]['ROC-AUC']:.4f}")
print(f"  PR-AUC   : {results[best_model_name]['PR-AUC']:.4f}")
print(f"  F1-Score : {results[best_model_name]['F1-Score']:.4f}")
print(f"  Recall   : {results[best_model_name]['Recall']:.4f}")
print(f"  Seuil optimal : {best_threshold:.2f}")

print("\nRésultats CV (validation croisée stratifiée 5-fold) :")
cv_df = pd.DataFrame(cv_results).T.round(4)
print(cv_df.to_string())

print("\n" + "=" * 60)
print("ÉVALUATION TERMINÉE — Tous les graphiques dans /outputs/")
print("=" * 60)
