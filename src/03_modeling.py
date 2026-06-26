"""
03_modeling.py — Entraînement des 4 Modèles de Classification
Projet : Maintenance Prédictive Industrielle
Variable cible : failure_within_24h

Modèles :
  1. Régression Logistique (baseline interprétable)
  2. Forêt Aléatoire (Random Forest)
  3. Gradient Boosting (GradientBoostingClassifier)
  4. MLP — Perceptron Multi-Couches (Deep Learning simple)

Stratégie déséquilibre : class_weight="balanced" + SMOTE
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("../models", exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ─────────────────────────────────────────────
print("=" * 60)
print("MODÉLISATION — Maintenance Prédictive (4 modèles)")
print("=" * 60)

X_train = np.load("../models/X_train_proc.npy")
X_test = np.load("../models/X_test_proc.npy")
y_train = np.load("../models/y_train.npy")
y_test = np.load("../models/y_test.npy")
feature_names = joblib.load("../models/feature_names.pkl")

print(f"X_train : {X_train.shape} | Positifs : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"X_test  : {X_test.shape}  | Positifs : {y_test.sum()}  ({y_test.mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. GESTION DU DÉSÉQUILIBRE — SMOTE
# ─────────────────────────────────────────────
print("\n[1] Application du SMOTE sur le train set...")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"   Avant SMOTE : {len(y_train)} obs | Positifs : {int(y_train.sum())}")
print(f"   Après SMOTE : {len(y_train_smote)} obs | Positifs : {int(y_train_smote.sum())}")

# ─────────────────────────────────────────────
# 3. DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────
print("\n[2] Définition des modèles...")

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, C=0.1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1, min_samples_leaf=5
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,          # L2 regularization pour éviter l'overfitting
        batch_size=256,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
}

# ─────────────────────────────────────────────
# 4. VALIDATION CROISÉE STRATIFIÉE (sur données originales)
# ─────────────────────────────────────────────
print("\n[3] Validation croisée stratifiée (5 folds) — données originales...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = ["accuracy", "f1", "roc_auc", "recall", "precision"]

cv_results = {}

for name, model in models.items():
    print(f"\n   → Cross-validation : {name}")
    t0 = time.time()
    cv = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=cv_metrics,
        return_train_score=False,
        n_jobs=-1
    )
    elapsed = time.time() - t0
    cv_results[name] = {m: cv[f"test_{m}"].mean() for m in cv_metrics}
    cv_results[name]["cv_time_s"] = round(elapsed, 2)
    print(f"     F1={cv_results[name]['f1']:.4f} | ROC-AUC={cv_results[name]['roc_auc']:.4f} | Recall={cv_results[name]['recall']:.4f} | Durée={elapsed:.1f}s")

# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT FINAL SUR SMOTE + TEST FINAL
# ─────────────────────────────────────────────
print("\n[4] Entraînement final sur SMOTE train set...")

trained_models = {}

for name, model in models.items():
    print(f"   → Entraînement : {name}...")
    t0 = time.time()
    model.fit(X_train_smote, y_train_smote)
    elapsed = time.time() - t0
    trained_models[name] = model
    print(f"     Terminé en {elapsed:.1f}s")

# ─────────────────────────────────────────────
# 6. SAUVEGARDE DES MODÈLES ET RÉSULTATS CV
# ─────────────────────────────────────────────
print("\n[5] Sauvegarde des modèles et résultats CV...")

for name, model in trained_models.items():
    path = f"../models/{name}.pkl"
    joblib.dump(model, path)
    print(f"   [OK] {path}")

joblib.dump(cv_results, "../models/cv_results.pkl")
print("   [OK] cv_results.pkl")

# Tableau récapitulatif CV
cv_df = pd.DataFrame(cv_results).T.round(4)
cv_df.to_csv("../outputs/cv_results.csv")
print("\n--- Résultats Validation Croisée ---")
print(cv_df.to_string())

print("\n" + "=" * 60)
print("MODÉLISATION TERMINÉE")
print("=" * 60)
