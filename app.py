"""
app.py  —  Dashboard Maintenance Predictive Industrielle
Master Data Science — RNCP40875
Oriente utilisateur metier : responsable maintenance / ingenieur
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.facecolor": "#0f1117", "axes.facecolor": "#1a1f2e",
                             "axes.edgecolor": "#2d3748", "text.color": "#e2e8f0",
                             "axes.labelcolor": "#e2e8f0", "xtick.color": "#a0aec0",
                             "ytick.color": "#a0aec0", "grid.color": "#2d3748",
                             "axes.titlecolor": "#f7fafc", "figure.dpi": 130})

# ─── CHEMINS ────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA   = os.path.join(BASE, "data")

# ─── CONFIG PAGE ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maintenance Predictive",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS GLOBAL ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Police de base et fond */
html, body, [class*="css"] {
    font-size: 16px;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Sidebar — texte plus grand */
section[data-testid="stSidebar"] * {
    font-size: 15px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    padding: 6px 0 !important;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 18px !important;
}

/* Titres de page */
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: 0.03em; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

/* Carte KPI */
.kpi-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 20px 22px;
    margin-bottom: 8px;
}
.kpi-label {
    font-size: 0.85rem;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f7fafc;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #718096;
    margin-top: 4px;
}

/* Alerte haute */
.alert-danger {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 22px 24px;
    text-align: center;
}
.alert-danger .alert-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fca5a5;
    letter-spacing: 0.05em;
}
.alert-danger .alert-prob {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
}
.alert-danger .alert-sub {
    font-size: 0.95rem;
    color: #fca5a5;
    margin-top: 6px;
}

/* Alerte basse */
.alert-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 22px 24px;
    text-align: center;
}
.alert-safe .alert-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #6ee7b7;
    letter-spacing: 0.05em;
}
.alert-safe .alert-prob {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
}
.alert-safe .alert-sub {
    font-size: 0.95rem;
    color: #6ee7b7;
    margin-top: 6px;
}

/* Alerte intermediaire */
.alert-warn {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 22px 24px;
    text-align: center;
}
.alert-warn .alert-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fcd34d;
    letter-spacing: 0.05em;
}
.alert-warn .alert-prob {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
}
.alert-warn .alert-sub {
    font-size: 0.95rem;
    color: #fcd34d;
    margin-top: 6px;
}

/* Tableau de comparaison */
.model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}
.model-table th {
    background: #2d3748;
    color: #a0aec0;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    padding: 12px 14px;
    text-align: left;
}
.model-table td {
    padding: 12px 14px;
    border-bottom: 1px solid #2d3748;
    color: #e2e8f0;
}
.model-table tr:hover td { background: #1e2533; }
.model-table .best { color: #10b981; font-weight: 700; }
.model-table .model-name { font-weight: 600; color: #f7fafc; }

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #2d3748;
    margin: 24px 0;
}

/* Info box */
.info-box {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px 18px;
    font-size: 0.92rem;
    color: #a0aec0;
    line-height: 1.6;
}
.info-box strong { color: #f7fafc; }

/* Facteur SHAP */
.shap-bar-container { margin: 8px 0; }
.shap-label {
    font-size: 0.88rem;
    color: #a0aec0;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
}
.shap-bar-bg {
    background: #2d3748;
    border-radius: 4px;
    height: 14px;
    width: 100%;
}
.shap-bar-fill {
    height: 14px;
    border-radius: 4px;
    background: #f59e0b;
}

/* Sidebar header */
.sidebar-header {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 18px;
}
.sidebar-header p {
    margin: 0;
    font-size: 0.85rem !important;
    color: #a0aec0;
    line-height: 1.5;
}
.sidebar-header strong {
    color: #f7fafc;
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── CHARGEMENT DONNEES ET MODELES ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA, "predictive_maintenance_v3.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_resource
def load_all():
    model_names = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP"]
    models = {n: joblib.load(os.path.join(MODELS, f"{n}.pkl")) for n in model_names}
    preprocessor   = joblib.load(os.path.join(MODELS, "preprocessor.pkl"))
    feature_names  = joblib.load(os.path.join(MODELS, "feature_names.pkl"))
    cv_results     = joblib.load(os.path.join(MODELS, "cv_results.pkl"))
    threshold_data = joblib.load(os.path.join(MODELS, "optimal_threshold.pkl"))
    shap_data      = joblib.load(os.path.join(MODELS, "shap_data.pkl"))
    return models, preprocessor, feature_names, cv_results, threshold_data, shap_data

df = load_data()
models, preprocessor, feature_names, cv_results, threshold_data, shap_data = load_all()

BEST_MODEL    = threshold_data["best_model"]
OPT_THRESHOLD = float(threshold_data["optimal_threshold"])

# SHAP importances globales (classe 1)
shap_vals_raw = np.array(shap_data["shap_values"])
if shap_vals_raw.ndim == 3:
    shap_vals = shap_vals_raw[:, :, 1]
else:
    shap_vals = shap_vals_raw  # déjà 2D (500, 21)
shap_mean_abs = np.abs(shap_vals).mean(axis=0)
shap_idx  = np.argsort(shap_mean_abs)[::-1]
feature_names_list = list(feature_names)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <strong>Maintenance Predictive</strong><br>
        <p>Outil d'aide a la decision<br>
        Responsable maintenance / Ingenieur</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Tableau de bord",
            "Capteurs et correlations",
            "Performance des modeles",
            "Simulateur de risque",
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.85rem; color:#718096; line-height:1.7;">
        <strong style="color:#a0aec0;">Dataset</strong><br>
        24 042 observations<br>
        4 types de machines<br>
        8 capteurs industriels<br><br>
        <strong style="color:#a0aec0;">Modele actif</strong><br>
        <span style="color:#f59e0b;">{BEST_MODEL}</span><br>
        Seuil de decision : {OPT_THRESHOLD:.2f}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ════════════════════════════════════════════════════════════════════════════
if page == "Tableau de bord":
    st.markdown("## Tableau de Bord Operationnel")
    st.markdown(
        "<p style='color:#718096; font-size:1rem; margin-top:-10px;'>"
        "Vue d'ensemble de l'etat du parc machine et des risques de panne dans les 24 prochaines heures."
        "</p>", unsafe_allow_html=True
    )

    # ── KPI ──────────────────────────────────────────────────────────────────
    total       = len(df)
    n_pannes    = int(df["failure_within_24h"].sum())
    taux_panne  = n_pannes / total * 100
    avg_rul     = df["rul_hours"].mean()
    n_machines  = df["machine_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Equipements surveilles</div>
            <div class="kpi-value">{n_machines}</div>
            <div class="kpi-sub">machines actives dans le dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Alertes panne 24h</div>
            <div class="kpi-value">{n_pannes:,}</div>
            <div class="kpi-sub">sur {total:,} observations</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Taux de panne global</div>
            <div class="kpi-value">{taux_panne:.1f}%</div>
            <div class="kpi-sub">ratio classe 1 / total</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Duree de vie residuelle moy.</div>
            <div class="kpi-value">{avg_rul:.0f}h</div>
            <div class="kpi-sub">sur l'ensemble du parc</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    # ── Taux de panne par machine ─────────────────────────────────────────
    with col_left:
        st.markdown("### Risque par type de machine")
        fm = (df.groupby("machine_type")["failure_within_24h"].mean() * 100).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        colors_bar = ["#374151" if v < fm.max() else "#f59e0b" for v in fm.values]
        bars = ax.barh(fm.index, fm.values, color=colors_bar, height=0.55, edgecolor="none")
        for bar in bars:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center", fontsize=11, color="#e2e8f0")
        ax.set_xlabel("Taux de panne (%)", fontsize=11)
        ax.set_xlim(0, fm.max() * 1.28)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title("Taux de panne moyen par type", fontsize=12, pad=10)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Taux par mode operatoire ──────────────────────────────────────────
    with col_right:
        st.markdown("### Risque par mode operatoire")
        fo = (df.groupby("operating_mode")["failure_within_24h"].mean() * 100).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        col_mode = {"idle": "#374151", "normal": "#2563eb", "peak": "#f59e0b"}
        for label, val in fo.items():
            ax.barh(label, val, color=col_mode.get(label, "#374151"), height=0.55, edgecolor="none")
            ax.text(val + 0.2, list(fo.index).index(label),
                    f"{val:.1f}%", va="center", fontsize=11, color="#e2e8f0")
        ax.set_xlabel("Taux de panne (%)", fontsize=11)
        ax.set_xlim(0, fo.max() * 1.28)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title("Taux de panne par mode operatoire", fontsize=12, pad=10)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Evolution mensuelle ────────────────────────────────────────────────
    st.markdown("### Evolution mensuelle du taux de panne")
    df_t = df.copy()
    df_t["month_str"] = df_t["timestamp"].dt.to_period("M").astype(str)
    monthly = df_t.groupby("month_str")["failure_within_24h"].mean() * 100

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.18, color="#f59e0b")
    ax.plot(range(len(monthly)), monthly.values, color="#f59e0b", lw=2.5, marker="o", markersize=5)
    ax.axhline(monthly.mean(), color="#718096", lw=1.2, linestyle="--", label=f"Moyenne {monthly.mean():.1f}%")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Taux de panne (%)", fontsize=11)
    ax.set_title("Taux de panne mensuel — Vue temporelle", fontsize=12, pad=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Variables les plus influentes (SHAP global) ───────────────────────
    st.markdown("### Variables les plus influentes sur le risque de panne")
    st.markdown(
        "<p style='color:#718096; font-size:0.92rem;'>"
        "Importance SHAP moyenne — indique dans quelle mesure chaque capteur influence la prediction du modele."
        "</p>", unsafe_allow_html=True
    )

    top_n = 8
    top_feats = [feature_names_list[i] for i in shap_idx[:top_n]]
    top_vals  = shap_mean_abs[shap_idx[:top_n]]
    max_val   = top_vals[0]

    labels_fr = {
        "rul_hours": "Duree de vie residuelle (h)",
        "temperature_motor": "Temperature moteur (C)",
        "vibration_x_current": "Vibration x Courant (compose)",
        "temp_over_rpm": "Temperature / RPM (compose)",
        "maintenance_urgency": "Urgence maintenance (compose)",
        "rpm": "Vitesse de rotation (RPM)",
        "vibration_rms": "Vibration RMS (mm/s)",
        "current_phase_avg": "Courant de phase (A)",
        "hours_since_maintenance": "Heures depuis maintenance",
        "day_of_week": "Jour de la semaine",
        "pressure_level": "Niveau de pression (bar)",
        "ambient_temp": "Temperature ambiante (C)",
    }

    cols_shap = st.columns(2)
    for i, (feat, val) in enumerate(zip(top_feats, top_vals)):
        col = cols_shap[i % 2]
        label = labels_fr.get(feat, feat)
        pct = val / max_val * 100
        with col:
            st.markdown(f"""
            <div class="shap-bar-container">
                <div class="shap-label">
                    <span>{label}</span>
                    <span style="color:#f59e0b; font-weight:600;">{val:.4f}</span>
                </div>
                <div class="shap-bar-bg">
                    <div class="shap-bar-fill" style="width:{pct:.0f}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CAPTEURS ET CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Capteurs et correlations":
    st.markdown("## Analyse des Capteurs")
    st.markdown(
        "<p style='color:#718096; font-size:1rem; margin-top:-10px;'>"
        "Explorez comment les mesures des capteurs varient selon l'etat de l'equipement."
        "</p>", unsafe_allow_html=True
    )

    NUMERIC_COLS = [
        "vibration_rms", "temperature_motor", "current_phase_avg",
        "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp", "rul_hours"
    ]
    LABELS = {
        "vibration_rms":            "Vibration RMS (mm/s)",
        "temperature_motor":        "Temperature moteur (C)",
        "current_phase_avg":        "Courant de phase (A)",
        "pressure_level":           "Pression (bar)",
        "rpm":                      "Vitesse de rotation (RPM)",
        "hours_since_maintenance":  "Heures depuis maintenance",
        "ambient_temp":             "Temperature ambiante (C)",
        "rul_hours":                "Duree de vie residuelle (h)",
    }

    # ── Selecteur capteur ────────────────────────────────────────────────
    st.markdown("### Distribution d'un capteur par etat machine")
    col_sel, col_fil = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox("Capteur a analyser", NUMERIC_COLS,
                                format_func=lambda x: LABELS[x])
    with col_fil:
        machine_filter = st.selectbox("Filtrer par type de machine",
                                      ["Tous"] + sorted(df["machine_type"].unique().tolist()))

    df_sel = df if machine_filter == "Tous" else df[df["machine_type"] == machine_filter]

    col_dist, col_box = st.columns(2)

    with col_dist:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        d0 = df_sel[df_sel["failure_within_24h"] == 0][selected].dropna()
        d1 = df_sel[df_sel["failure_within_24h"] == 1][selected].dropna()
        ax.hist(d0, bins=45, alpha=0.65, color="#10b981", label="Etat normal", density=True, edgecolor="none")
        ax.hist(d1, bins=45, alpha=0.65, color="#ef4444", label="Panne dans 24h", density=True, edgecolor="none")
        ax.set_xlabel(LABELS[selected], fontsize=11)
        ax.set_ylabel("Densite", fontsize=11)
        ax.set_title("Distribution par classe", fontsize=12, pad=8)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_box:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bp = ax.boxplot(
            [d0.values, d1.values],
            patch_artist=True,
            labels=["Normal", "Panne"],
            widths=0.45,
            medianprops=dict(color="#f7fafc", lw=2.5),
            whiskerprops=dict(color="#718096"),
            capprops=dict(color="#718096"),
            flierprops=dict(marker=".", color="#718096", markersize=3, alpha=0.4)
        )
        bp["boxes"][0].set_facecolor("#10b981")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#ef4444")
        bp["boxes"][1].set_alpha(0.7)
        ax.set_ylabel(LABELS[selected], fontsize=11)
        ax.set_title("Ecart interquartile", fontsize=12, pad=8)
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Interpretation automatique
    med_normal = d0.median()
    med_panne  = d1.median()
    delta_pct  = (med_panne - med_normal) / (med_normal + 1e-9) * 100
    direction  = "superieur" if delta_pct > 0 else "inferieur"
    st.markdown(f"""
    <div class="info-box">
        <strong>Lecture :</strong> La valeur mediane de <em>{LABELS[selected]}</em>
        est <strong>{abs(delta_pct):.1f}% {direction}</strong> sur les equipements
        en panne ({med_panne:.2f}) par rapport aux equipements normaux ({med_normal:.2f}).
        {"Un niveau eleve est donc un signal d'alerte pour ce capteur." if delta_pct > 0
         else "Un niveau faible est donc un signal d'alerte pour ce capteur."}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Matrice de correlation ────────────────────────────────────────────
    st.markdown("### Matrice de correlation")
    st.markdown(
        "<p style='color:#718096; font-size:0.92rem;'>"
        "Correlation de Pearson entre les capteurs et la variable cible."
        "</p>", unsafe_allow_html=True
    )

    corr_cols = NUMERIC_COLS + ["failure_within_24h"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 145, s=80, l=40, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                center=0, linewidths=0.6, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 10, "color": "#f7fafc"},
                cbar_kws={"shrink": 0.7})
    ax.set_xticklabels([LABELS.get(c, c) for c in corr.columns],
                       rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels([LABELS.get(c, c) for c in corr.index],
                       rotation=0, fontsize=9.5)
    ax.set_title("Matrice de correlation — capteurs et cible", fontsize=12, pad=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlations avec la cible
    st.markdown("### Correlations avec la variable cible (failure_within_24h)")
    target_corr = corr["failure_within_24h"].drop("failure_within_24h").sort_values(key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_corr = ["#ef4444" if v > 0 else "#10b981" for v in target_corr.values]
    bars = ax.barh([LABELS.get(c, c) for c in target_corr.index],
                   target_corr.values, color=colors_corr, height=0.55, edgecolor="none")
    ax.axvline(0, color="#718096", lw=1)
    ax.set_xlabel("Coefficient de correlation", fontsize=11)
    ax.set_title("Impact de chaque capteur sur le risque de panne", fontsize=12, pad=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=0.2)
    p1 = mpatches.Patch(color="#ef4444", label="Correlation positive (capteur eleve = risque eleve)")
    p2 = mpatches.Patch(color="#10b981", label="Correlation negative (capteur faible = risque eleve)")
    ax.legend(handles=[p1, p2], fontsize=10, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE DES MODELES
# ════════════════════════════════════════════════════════════════════════════
elif page == "Performance des modeles":
    st.markdown("## Comparaison des Modeles")
    st.markdown(
        "<p style='color:#718096; font-size:1rem; margin-top:-10px;'>"
        "Performance des quatre modeles de classification sur le jeu de test reel. "
        "Les metriques sont adaptees au contexte de desequilibre des classes."
        "</p>", unsafe_allow_html=True
    )

    results = {
        "Regression Logistique": {"Accuracy": 0.9393, "Recall": 0.9382, "Precision": 0.7293, "F1-Score": 0.8206, "ROC-AUC": 0.9828, "PR-AUC": 0.9241},
        "Random Forest":         {"Accuracy": 0.9971, "Recall": 0.9944, "Precision": 0.9861, "F1-Score": 0.9902, "ROC-AUC": 0.9999, "PR-AUC": 0.9996},
        "Gradient Boosting":     {"Accuracy": 0.9967, "Recall": 0.9916, "Precision": 0.9860, "F1-Score": 0.9888, "ROC-AUC": 0.9998, "PR-AUC": 0.9992},
        "MLP (Deep Learning)":   {"Accuracy": 0.9921, "Recall": 0.9775, "Precision": 0.9694, "F1-Score": 0.9734, "ROC-AUC": 0.9991, "PR-AUC": 0.9945},
    }
    res_df = pd.DataFrame(results).T

    # ── Tableau HTML ─────────────────────────────────────────────────────
    st.markdown("### Metriques sur le jeu de test")

    metrics_order = ["Recall", "Precision", "F1-Score", "ROC-AUC", "PR-AUC", "Accuracy"]
    best_per_col = {m: res_df[m].max() for m in metrics_order}

    header = "".join(f"<th>{m}</th>" for m in metrics_order)
    rows = ""
    for model_name, row in res_df.iterrows():
        is_best = (model_name == "Random Forest")
        name_cell = f'<td class="model-name">{"-> " if is_best else ""}{model_name}</td>'
        cells = "".join(
            f'<td class="{"best" if abs(row[m] - best_per_col[m]) < 1e-6 else ""}">{row[m]:.4f}</td>'
            for m in metrics_order
        )
        rows += f"<tr>{name_cell}{cells}</tr>"

    st.markdown(f"""
    <table class="model-table">
        <thead><tr><th>Modele</th>{header}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <p style="font-size:0.85rem; color:#718096; margin-top:8px;">
        En vert : meilleure valeur de la colonne — Modele recommande : Random Forest
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Graphique comparatif ──────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### Recall et ROC-AUC par modele")
        short_names = ["Log. Reg.", "Rand. Forest", "Grad. Boost.", "MLP"]
        recall_vals = [results[m]["Recall"] for m in results]
        auc_vals    = [results[m]["ROC-AUC"] for m in results]
        x = np.arange(len(short_names))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        w = 0.35
        b1 = ax.bar(x - w/2, recall_vals, width=w, label="Recall", color="#ef4444", edgecolor="none", alpha=0.9)
        b2 = ax.bar(x + w/2, auc_vals,    width=w, label="ROC-AUC", color="#3b82f6", edgecolor="none", alpha=0.9)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9, color="#e2e8f0")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=11)
        ax.set_ylim(0.7, 1.06)
        ax.legend(fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Recall et ROC-AUC", fontsize=12, pad=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.markdown("### F1-Score et PR-AUC par modele")
        f1_vals = [results[m]["F1-Score"] for m in results]
        pr_vals = [results[m]["PR-AUC"] for m in results]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        b3 = ax.bar(x - w/2, f1_vals, width=w, label="F1-Score", color="#10b981", edgecolor="none", alpha=0.9)
        b4 = ax.bar(x + w/2, pr_vals, width=w, label="PR-AUC",   color="#f59e0b", edgecolor="none", alpha=0.9)
        for bar in list(b3) + list(b4):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9, color="#e2e8f0")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=11)
        ax.set_ylim(0.7, 1.06)
        ax.legend(fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("F1-Score et PR-AUC", fontsize=12, pad=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Explication des metriques ─────────────────────────────────────────
    st.markdown("### Pourquoi ces metriques ?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box">
            <strong>Recall (priorite n°1)</strong><br>
            Parmi toutes les vraies pannes, quelle proportion le modele a-t-il detectee ?
            Une panne non detectee (faux negatif) engendre un arret non planifie couteux.
            C'est la metrique la plus importante dans ce contexte industriel.
            <br><br>
            <strong>ROC-AUC</strong><br>
            Capacite globale du modele a discriminer les deux classes,
            independamment du seuil de decision. Robuste au desequilibre des classes.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
            <strong>PR-AUC (Precision-Recall)</strong><br>
            Particulierement adapte aux classes desequilibrees (14.8% de pannes).
            Mesure la capacite a detecter les pannes tout en limitant les fausses alertes.
            <br><br>
            <strong>Pourquoi pas seulement l'Accuracy ?</strong><br>
            Un modele qui predit systematiquement "pas de panne" obtient 85% d'accuracy
            sans rien apprendre. L'accuracy est donc trompeuse ici.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Validation croisee ────────────────────────────────────────────────
    st.markdown("### Stabilite — Validation croisee 5-fold")
    st.markdown(
        "<p style='color:#718096; font-size:0.92rem;'>"
        "Resultats sur le train set par validation croisee stratifiee (5 folds). "
        "La coherence avec les scores du test set confirme l'absence d'overfitting."
        "</p>", unsafe_allow_html=True
    )

    cv_df = pd.DataFrame(cv_results).T[["f1", "roc_auc", "recall", "accuracy"]].round(4)
    cv_df.index = ["Regression Logistique", "Random Forest", "Gradient Boosting", "MLP (Deep Learning)"]
    cv_df.columns = ["F1-Score CV", "ROC-AUC CV", "Recall CV", "Accuracy CV"]

    cv_header = "".join(f"<th>{c}</th>" for c in cv_df.columns)
    cv_rows = ""
    cv_best = {c: cv_df[c].max() for c in cv_df.columns}
    for name, row in cv_df.iterrows():
        cells = "".join(
            f'<td class="{"best" if abs(row[c] - cv_best[c]) < 1e-6 else ""}">{row[c]:.4f}</td>'
            for c in cv_df.columns
        )
        cv_rows += f"<tr><td class='model-name'>{name}</td>{cells}</tr>"

    st.markdown(f"""
    <table class="model-table">
        <thead><tr><th>Modele</th>{cv_header}</tr></thead>
        <tbody>{cv_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SIMULATEUR DE RISQUE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Simulateur de risque":
    st.markdown("## Simulateur de Risque en Temps Reel")
    st.markdown(
        "<p style='color:#718096; font-size:1rem; margin-top:-10px;'>"
        "Renseignez les parametres d'une machine pour obtenir une prediction de panne "
        "dans les 24 prochaines heures et comprendre les facteurs qui influencent cette decision."
        "</p>", unsafe_allow_html=True
    )

    # ── Formulaire ───────────────────────────────────────────────────────
    col_form, col_result = st.columns([1.1, 0.9])

    with col_form:
        st.markdown("### Parametres de la machine")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            machine_type   = st.selectbox("Type de machine",   ["CNC", "Pump", "Compressor", "Robotic Arm"])
            operating_mode = st.selectbox("Mode operatoire",   ["idle", "normal", "peak"])
        with r1c2:
            vibration_rms  = st.slider("Vibration RMS (mm/s)",       0.35, 10.0,  1.6,  0.05)
            temperature_motor = st.slider("Temperature moteur (C)",  28.0, 95.0, 51.0,  0.5)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            rpm            = st.slider("Vitesse de rotation (RPM)",  124.0, 4100.0, 1145.0, 10.0)
            pressure_level = st.slider("Pression (bar)",              10.0,  206.0,   59.0,  0.5)
        with r2c2:
            current_phase_avg = st.slider("Courant de phase (A)",    2.2,  35.0,  8.8, 0.1)
            ambient_temp      = st.slider("Temperature ambiante (C)", 8.0,  18.0, 13.0, 0.5)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            hours_since_maintenance = st.slider("Heures depuis maintenance",  0.0, 575.0, 172.0, 1.0)
        with r3c2:
            rul_hours = st.slider("Duree de vie residuelle (h)",              0.5,  98.0,  27.8, 0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        col_model, col_thresh = st.columns(2)
        with col_model:
            model_choice = st.selectbox("Modele", list(models.keys()), index=list(models.keys()).index(BEST_MODEL))
        with col_thresh:
            use_opt = st.checkbox(f"Seuil optimise ({OPT_THRESHOLD:.2f})", value=True)
        threshold = OPT_THRESHOLD if use_opt else 0.50

        predict_btn = st.button("Lancer la prediction", type="primary", use_container_width=True)

    # ── Resultats ─────────────────────────────────────────────────────────
    with col_result:
        if predict_btn:
            # Construction du vecteur d'entree
            input_df = pd.DataFrame([{
                "vibration_rms":            vibration_rms,
                "temperature_motor":        temperature_motor,
                "current_phase_avg":        current_phase_avg,
                "pressure_level":           pressure_level,
                "rpm":                      rpm,
                "hours_since_maintenance":  hours_since_maintenance,
                "ambient_temp":             ambient_temp,
                "rul_hours":                rul_hours,
                "machine_type":             machine_type,
                "operating_mode":           operating_mode,
                "hour":                     12,
                "day_of_week":              0,
                "month":                    6,
                "temp_over_rpm":            temperature_motor / (rpm + 1),
                "vibration_x_current":      vibration_rms * current_phase_avg,
                "maintenance_urgency":      hours_since_maintenance / (rul_hours + 1),
            }])

            X_in    = preprocessor.transform(input_df)
            y_proba = float(models[model_choice].predict_proba(X_in)[0, 1])
            y_pred  = int(y_proba >= threshold)

            # Niveau de risque
            if y_proba >= threshold:
                level, css_class = "RISQUE ELEVE", "alert-danger"
                color_jauge = "#ef4444"
            elif y_proba >= 0.30:
                level, css_class = "RISQUE MODERE", "alert-warn"
                color_jauge = "#f59e0b"
            else:
                level, css_class = "ETAT NORMAL", "alert-safe"
                color_jauge = "#10b981"

            # Bloc d'alerte
            action_msg = {
                "alert-danger": "Intervention preventive recommandee dans les 24h.",
                "alert-warn":   "Surveillance renforcee conseillee.",
                "alert-safe":   "Aucune intervention immediate requise.",
            }
            st.markdown(f"""
            <div class="{css_class}">
                <div class="alert-title">{level}</div>
                <div class="alert-prob">{y_proba*100:.1f}%</div>
                <div class="alert-sub">probabilite de panne dans les 24h</div>
                <div style="margin-top:10px; font-size:0.9rem; opacity:0.9;">
                    {action_msg[css_class]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Jauge horizontale
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 0.9))
            ax.barh([0], [1],           height=0.6, color="#2d3748",      edgecolor="none")
            ax.barh([0], [y_proba],     height=0.6, color=color_jauge,    edgecolor="none")
            ax.axvline(threshold, color="#f7fafc", lw=2, linestyle="--")
            ax.text(threshold + 0.01, 0.35, f"seuil {threshold:.2f}", color="#f7fafc", fontsize=9, va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.8)
            ax.axis("off")
            fig.tight_layout(pad=0.2)
            st.pyplot(fig)
            plt.close()

            st.markdown(f"""
            <p style="font-size:0.85rem; color:#718096; text-align:center;">
                Modele : {model_choice} — Seuil : {threshold:.2f}
            </p>
            """, unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Explication SHAP — contribution des variables ───────────────
            st.markdown("### Pourquoi cette prediction ?")
            st.markdown(
                "<p style='color:#718096; font-size:0.88rem;'>"
                "Les valeurs SHAP indiquent la contribution de chaque parametre saisi "
                "a la prediction finale. Une valeur positive pousse vers 'panne', negative vers 'normal'."
                "</p>", unsafe_allow_html=True
            )

            # Calcul SHAP sur l'observation saisie
            import shap as shap_lib
            m = models[model_choice]
            if model_choice in ["RandomForest", "GradientBoosting"]:
                explainer   = shap_lib.TreeExplainer(m)
                sv_instance = explainer.shap_values(X_in)
                if isinstance(sv_instance, list):
                    sv_obs = sv_instance[1][0]
                else:
                    sv_obs = np.array(sv_instance)[0, :, 1] if sv_instance.ndim == 3 else sv_instance[0]
            elif model_choice == "LogisticRegression":
                explainer   = shap_lib.LinearExplainer(m, X_in)
                sv_obs      = explainer.shap_values(X_in)[0]
            else:
                bg          = shap_lib.sample(np.load(os.path.join(MODELS, "X_train_proc.npy")), 50)
                explainer   = shap_lib.KernelExplainer(m.predict_proba, bg)
                sv_obs      = explainer.shap_values(X_in)[0][:, 1] if explainer.shap_values(X_in).ndim == 3 else explainer.shap_values(X_in)[0]

            # Top 8 features par valeur absolue
            sv_abs  = np.abs(sv_obs)
            top_idx = np.argsort(sv_abs)[::-1][:8]

            labels_fr = {
                "rul_hours":                 "Duree de vie residuelle",
                "temperature_motor":         "Temperature moteur",
                "vibration_x_current":       "Vibration x Courant",
                "temp_over_rpm":             "Temperature / RPM",
                "maintenance_urgency":       "Urgence maintenance",
                "rpm":                       "Vitesse de rotation (RPM)",
                "vibration_rms":             "Vibration RMS",
                "current_phase_avg":         "Courant de phase",
                "hours_since_maintenance":   "Heures depuis maintenance",
                "pressure_level":            "Pression",
                "ambient_temp":              "Temperature ambiante",
            }

            fig, ax = plt.subplots(figsize=(6, 3.8))
            feat_labels = [labels_fr.get(feature_names_list[i], feature_names_list[i]) for i in top_idx]
            feat_vals   = [sv_obs[i] for i in top_idx]
            colors_sv   = ["#ef4444" if v > 0 else "#10b981" for v in feat_vals]
            ax.barh(feat_labels[::-1], feat_vals[::-1], color=colors_sv[::-1], height=0.55, edgecolor="none")
            ax.axvline(0, color="#718096", lw=1)
            ax.set_xlabel("Contribution SHAP (impact sur la prediction)", fontsize=10)
            ax.set_title("Facteurs de la prediction", fontsize=11, pad=8)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="x", alpha=0.2)
            p1 = mpatches.Patch(color="#ef4444", label="Augmente le risque de panne")
            p2 = mpatches.Patch(color="#10b981", label="Reduit le risque de panne")
            ax.legend(handles=[p1, p2], fontsize=9, loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box" style="text-align:center; padding: 40px 24px;">
                <strong style="font-size:1.1rem;">Aucune prediction en cours</strong><br><br>
                Renseignez les parametres de la machine dans le formulaire 
                puis cliquez sur <em>Lancer la prediction</em>.<br><br>
                Le modele calculera la probabilite de panne et 
                expliquera les facteurs determinants de sa decision.
            </div>
            """, unsafe_allow_html=True)
