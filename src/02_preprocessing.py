"""
02_preprocessing.py — Pipeline de Préparation des Données
Projet : Maintenance Prédictive Industrielle
Variable cible : failure_within_24h

Ce module prépare les données de façon sécurisée (sans data leakage)
en utilisant sklearn.Pipeline et ColumnTransformer.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
print("=" * 60)
print("PREPROCESSING — Maintenance Prédictive")
print("=" * 60)

df = pd.read_csv("../data/predictive_maintenance_v3.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[1] Feature Engineering...")

# Extraction temporelle
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# Ratios physiques pertinents (signaux composites)
df["temp_over_rpm"] = df["temperature_motor"] / (df["rpm"] + 1)
df["vibration_x_current"] = df["vibration_rms"] * df["current_phase_avg"]
df["maintenance_urgency"] = df["hours_since_maintenance"] / (df["rul_hours"] + 1)

# Suppression des colonnes non-features
cols_to_drop = [
    "timestamp", "machine_id",
    "failure_type",         # cible alternative → fuite potentielle
    "estimated_repair_cost" # cible alternative → fuite potentielle
]
df = df.drop(columns=cols_to_drop)

print(f"   Colonnes après engineering : {df.columns.tolist()}")

# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST (stratifié)
# ─────────────────────────────────────────────
print("\n[2] Split Train/Test (stratifié, 80/20)...")

TARGET = "failure_within_24h"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"   Train : {X_train.shape[0]} lignes — Positifs : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"   Test  : {X_test.shape[0]} lignes — Positifs : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. DÉFINITION DES COLONNES
# ─────────────────────────────────────────────
numeric_features = [
    "vibration_rms", "temperature_motor", "current_phase_avg", "pressure_level",
    "rpm", "hours_since_maintenance", "ambient_temp", "rul_hours",
    "hour", "day_of_week", "month",
    "temp_over_rpm", "vibration_x_current", "maintenance_urgency"
]

categorical_features = ["machine_type", "operating_mode"]

# ─────────────────────────────────────────────
# 5. PIPELINE DE PRÉPROCESSING
# ─────────────────────────────────────────────
print("\n[3] Construction du Pipeline sklearn...")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ─────────────────────────────────────────────
# 6. FIT SUR TRAIN UNIQUEMENT — Transformation des deux ensembles
# ─────────────────────────────────────────────
print("\n[4] Fit du preprocessor sur train uniquement (anti data leakage)...")

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Noms de colonnes après transformation
ohe_cats = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(ohe_cats)

print(f"   Dimensions X_train_proc : {X_train_proc.shape}")
print(f"   Dimensions X_test_proc  : {X_test_proc.shape}")
print(f"   Nombre de features finales : {len(feature_names)}")

# ─────────────────────────────────────────────
# 7. SAUVEGARDE
# ─────────────────────────────────────────────
print("\n[5] Sauvegarde des artefacts...")

joblib.dump(preprocessor, "../models/preprocessor.pkl")
joblib.dump(feature_names, "../models/feature_names.pkl")

np.save("../models/X_train_proc.npy", X_train_proc)
np.save("../models/X_test_proc.npy", X_test_proc)
np.save("../models/y_train.npy", y_train.values)
np.save("../models/y_test.npy", y_test.values)

# Sauvegarde des splits bruts pour le dashboard
X_train.to_csv("../models/X_train_raw.csv", index=False)
X_test.to_csv("../models/X_test_raw.csv", index=False)
y_train.to_csv("../models/y_train_raw.csv", index=False)
y_test.to_csv("../models/y_test_raw.csv", index=False)

print("   [OK] preprocessor.pkl")
print("   [OK] feature_names.pkl")
print("   [OK] X_train_proc.npy / X_test_proc.npy")
print("   [OK] y_train.npy / y_test.npy")
print("   [OK] splits bruts CSV")

print("\n" + "=" * 60)
print("PREPROCESSING TERMINÉ")
print("=" * 60)
