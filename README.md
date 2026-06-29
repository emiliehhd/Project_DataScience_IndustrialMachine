# Maintenance Prédictive Industrielle

Ce projet implémente une plateforme de maintenance prédictive basée sur des données de capteurs industriels. L'objectif est de prédire si un équipement est susceptible de tomber en panne dans les 24 prochaines heures, en s'appuyant sur quatre algorithmes de classification et un dashboard décisionnel interactif.

---

## Problématique

**Tâche prédictive : Classification Binaire**

Variable cible : `failure_within_24h`

Intérêt métier :
- Prioriser les interventions préventives
- Réduire les arrêts non planifiés
- Diminuer les coûts de maintenance corrective
- Améliorer la disponibilité des machines (uptime)

---

## Installation

Créer et activer un environnement virtuel Python :

```bash
python -m venv venv_maintenance

# WIndows
venv_maintenance\Scripts\activate

# Linux / Mac
source venv_maintenance/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Placer le fichier de données dans `data/` :

```
data/predictive_maintenance_v3.csv
```

---

## Exécution

Les scripts doivent être lancés dans l'ordre depuis le dossier `src/` :

```bash
cd src

python 01_EDA.py
python 02_preprocessing.py
python 03_modeling.py
python 04_evaluation.py
```

Lancer le dashboard :

```bash
cd ..
streamlit run app.py
```

---

## Dashboard Streamlit

Le dashboard est organisé de la manière suivante :

---

## Dataset

Source : [Kaggle — Industrial Machine Predictive Maintenance](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance)

Fichier : `predictive_maintenance_v3.csv`

Caractéristiques :
- 24 042 enregistrements
- 15 variables (capteurs, identifiants machine, variables cibles)
- Déséquilibre des classes : 85.2% (pas de panne) / 14.8% (panne)

Variables capteurs principales : `vibration_rms`, `temperature_motor`, `current_phase_avg`, `pressure_level`, `rpm`, `hours_since_maintenance`, `ambient_temp`, `rul_hours`

---

## Structure du projet

```
projet_maintenance/
|
├── data/
|   └── predictive_maintenance_v3.csv
|
├── src/
|   ├── 1_EDA.py               Analyse exploratoire des données
|   ├── 2_preprocessing.py     Pipeline de préparation des données
|   ├── 3_modeling.py          Entraînement des 4 modèles
|   └── 4_evaluation.py        Évaluation, interprétabilité, SHAP
|
├── models/                     Artefacts sauvegardés (.pkl, .npy, .csv)
|
├── outputs/                    Graphiques et tableaux générés
|
├── app.py                      Dashboard Streamlit
├── requirements.txt            Dépendances Python
└── README.md
```

---

## Modèles implémentés

1. **Régression Logistique** — modèle baseline, interprétable, robuste au déséquilibre via `class_weight="balanced"`
2. **Random Forest** — capture les non-linéarités, résistant au bruit, importance des variables native
3. **Gradient Boosting** — optimisation séquentielle, meilleur compromis précision/rappel
4. **MLP (Perceptron Multi-Couches)** — réseau de neurones avec deux couches cachées (64 et 32 neurones), regularisation L2, early stopping

---

## Résultats sur le test set

| Modèle              | Accuracy | Recall | Precision | F1-Score | ROC-AUC | PR-AUC |
|---------------------|----------|--------|-----------|----------|---------|--------|
| LogisticRegression  | 0.9393   | 0.9382 | 0.7293    | 0.8206   | 0.9828  | 0.9241 |
| RandomForest        | 0.9971   | 0.9944 | 0.9861    | 0.9902   | 0.9999  | 0.9996 |
| GradientBoosting    | 0.9967   | 0.9916 | 0.9860    | 0.9888   | 0.9998  | 0.9992 |
| MLP                 | 0.9921   | 0.9775 | 0.9694    | 0.9734   | 0.9991  | 0.9945 |

**Modèle final recommandé : Random Forest**

Meilleur ROC-AUC (0.9999) et meilleur Recall (99.4%) sur le test set, confirmés par la validation croisée stratifiée 5-fold. Dans un contexte industriel, minimiser les faux négatifs (pannes non détectées) est la priorité absolue. Le Random Forest offre le meilleur compromis entre performance, stabilité et interprétabilité.

Seuil de décision optimisé : 0.48 (maximisation du F1-Score, contre 0.50 par défaut).

---

## Méthodologie

### Gestion du déséquilibre des classes

Le ratio de déséquilibre est de 5.75:1 (majoritaire/minoritaire). Deux stratégies complémentaires ont été appliquées :

- `class_weight="balanced"` sur tous les modèles compatibles (Logistic Regression, Random Forest)
- SMOTE (Synthetic Minority Over-sampling Technique) appliqué uniquement sur le train set pour éviter tout data leakage

### Pipeline sklearn et anti data leakage

Le préprocessing est entièrement encapsulé dans un `ColumnTransformer` composé de deux sous-pipelines :
- Variables numériques : `SimpleImputer(median)` + `StandardScaler`
- Variables catégorielles : `SimpleImputer(most_frequent)` + `OneHotEncoder`

Le pipeline est ajusté (`fit`) exclusivement sur le train set, puis appliqué (`transform`) sur le test set.

### Validation croisée

`StratifiedKFold` à 5 folds appliquée sur le train set pour préserver les proportions de classes à chaque fold. Métriques reportées : Accuracy, F1, ROC-AUC, Recall, Precision.

### Feature Engi

Trois variables dérivées ont été construites à partir des capteurs bruts :
- `temp_over_rpm` : ratio température moteur / RPM
- `vibration_x_current` : produit vibration × courant de phase
- `maintenance_urgency` : ratio heures depuis maintenance / durée de vie résiduelle

### Interprétabilité

- Feature Importance native (Random Forest, Gradient Boosting)
- Permutation Importance (Logistic Regression)
- SHAP (TreeExplainer sur le modèle final) : summary plot beeswarm + bar plot global

Variables les plus influentes identifiées par SHAP : `rul_hours`, `vibration_rms`, `temperature_motor`, `hours_since_maintenance`, `vibration_x_current`.


