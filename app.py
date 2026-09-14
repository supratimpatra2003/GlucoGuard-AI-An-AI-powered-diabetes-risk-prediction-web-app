
import sqlite3
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="GlucoGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes (2).csv"
DB_PATH = BASE_DIR / "predictions.db"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

ZERO_AS_MISSING = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]


# ============================================================
# UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 5% 0%, rgba(79,70,229,.10), transparent 25%),
        radial-gradient(circle at 95% 5%, rgba(14,165,233,.08), transparent 25%),
        #f7f8fc;
}
.block-container { max-width: 1450px; padding-top: 1.7rem; }
.hero {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: white; padding: 30px 34px; border-radius: 24px;
    margin-bottom: 22px; box-shadow: 0 18px 55px rgba(15,23,42,.15);
}
.hero h1 { margin: 4px 0 5px; font-size: 43px; font-weight: 800; }
.hero p { color: #cbd5e1; margin: 0; }
.badge {
    display:inline-block; padding:6px 11px; border-radius:999px;
    background:#eef2ff; color:#4338ca; font-size:11px; font-weight:800;
}
.card {
    background: rgba(255,255,255,.92); border:1px solid #e5e7eb;
    border-radius:18px; padding:20px; box-shadow:0 8px 30px rgba(15,23,42,.05);
}
.high {
    background:linear-gradient(135deg,#fff1f2,#ffe4e6);
    border:1px solid #fecdd3; border-radius:22px; padding:28px;
}
.low {
    background:linear-gradient(135deg,#ecfdf5,#d1fae5);
    border:1px solid #a7f3d0; border-radius:22px; padding:28px;
}
.metric-note { color:#64748b; font-size:12px; }
.section-title { font-size:24px; font-weight:800; margin-top:8px; color:#111827 !important; }

/* Strong light-theme text overrides for Streamlit Community Cloud */
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] label,
section[data-testid="stMain"] label,
section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color:#111827 !important;
}
section[data-testid="stMain"] [data-testid="stWidgetLabel"] {
    color:#111827 !important;
}
.hero, .hero * { color:white !important; }
.hero p { color:#cbd5e1 !important; }
.hero .badge { color:#4338ca !important; }
.card, .card * { color:#111827; }
.card .metric-note { color:#64748b !important; }
.high, .high * { color:#7f1d1d; }
.low, .low * { color:#065f46; }

/* Make the primary action visually consistent */
section[data-testid="stMain"] button[kind="primary"] {
    background:linear-gradient(135deg,#4f46e5,#2563eb) !important;
    border:0 !important;
    color:white !important;
    font-weight:800 !important;
}
section[data-testid="stMain"] button[kind="primary"] p {
    color:white !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        pregnancies INTEGER,
        glucose REAL,
        blood_pressure REAL,
        skin_thickness REAL,
        insulin REAL,
        bmi REAL,
        pedigree REAL,
        age INTEGER,
        prediction INTEGER,
        probability REAL,
        model_name TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_prediction(v, prediction, probability, model_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    INSERT INTO predictions
    (created_at,pregnancies,glucose,blood_pressure,skin_thickness,
     insulin,bmi,pedigree,age,prediction,probability,model_name)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        v["Pregnancies"], v["Glucose"], v["BloodPressure"],
        v["SkinThickness"], v["Insulin"], v["BMI"],
        v["DiabetesPedigreeFunction"], v["Age"],
        int(prediction), float(probability), model_name
    ))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    result = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC", conn
    )
    conn.close()
    return result

init_db()


# ============================================================
# DATA / MODEL
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"'{DATA_PATH.name}' was not found beside app.py."
        )
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    return df

@st.cache_resource
def train_models(df):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    plain = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("prep", scaled),
            ("model", LogisticRegression(
                max_iter=2500, class_weight="balanced", C=1.0, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("prep", plain),
            ("model", RandomForestClassifier(
                n_estimators=450, max_depth=7, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", plain),
            ("model", GradientBoostingClassifier(
                n_estimators=180, learning_rate=.035, max_depth=2,
                min_samples_leaf=5, random_state=42
            ))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    fitted = {}

    for name, estimator in models.items():
        scores = cross_val_score(
            estimator, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_scores[name] = float(scores.mean())
        fitted[name] = estimator.fit(X_train, y_train)

    best_name = max(cv_scores, key=cv_scores.get)
    best = fitted[best_name]

    test_prob = best.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= .50).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_prob),
        "cm": confusion_matrix(y_test, test_pred),
        "report": classification_report(
            y_test, test_pred,
            target_names=["Outcome 0", "Outcome 1"],
            output_dict=True, zero_division=0
        )
    }

    # Permutation importance on the untouched test set.
    perm = permutation_importance(
        best, X_test, y_test,
        scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=-1
    )
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std
    }).sort_values("Importance", ascending=False)

    return {
        "best_name": best_name,
        "best_model": best,
        "models": fitted,
        "cv_scores": cv_scores,
        "metrics": metrics,
        "importance": importance,
        "X_test": X_test,
        "y_test": y_test
    }


df = load_data()
bundle = train_models(df)
best_name = bundle["best_name"]
model = bundle["best_model"]
metrics = bundle["metrics"]


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🩺 GlucoGuard AI")
    st.caption("Clinical-risk screening demo")

    page = st.radio(
        "Navigation",
        ["Risk Assessment", "Explainability", "Model Lab",
         "Prediction History", "Dataset"],
        index=0
    )

    st.divider()
    st.markdown("**Active model**")
    st.success(best_name)
    st.caption(f"Holdout ROC-AUC: {metrics['roc_auc']:.3f}")

    st.divider()
    st.caption(
        "Educational/research software. Not a medical diagnostic device."
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <span class="badge">ML • EXPLAINABILITY • ANALYTICS</span>
    <h1>GlucoGuard AI</h1>
    <p>Professional diabetes-risk screening dashboard with model evaluation,
    explainability and local prediction history.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# RISK ASSESSMENT
# ============================================================

if page == "Risk Assessment":

    st.markdown('<div class="section-title">Patient risk assessment</div>',
                unsafe_allow_html=True)
    st.caption("Enter the available measurements. Zero values for selected clinical fields are treated as missing.")

    left, right = st.columns([1.35, .85], gap="large")

    with left:
        with st.form("risk_form"):
            a, b = st.columns(2)
            with a:
                pregnancies = st.number_input("Pregnancies", 0, 20, 2)
                glucose = st.number_input("Glucose", 0.0, 300.0, 120.0, 1.0)
                bp = st.number_input("Blood Pressure", 0.0, 200.0, 70.0, 1.0)
                skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0, 1.0)
            with b:
                insulin = st.number_input("Insulin", 0.0, 900.0, 80.0, 1.0)
                bmi = st.number_input("BMI", 0.0, 80.0, 25.0, .1)
                pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, .45, .01)
                age = st.number_input("Age", 1, 120, 30)

            submitted = st.form_submit_button(
                "Analyze risk →", type="primary", use_container_width=True
            )

        result_slot = st.container()

    with right:
        st.markdown("""
        <div class="card">
        <b>What is analyzed?</b><br><br>
        Glucose • BMI • Age • Blood pressure • Pregnancies • Insulin •
        Skin thickness • Diabetes pedigree function
        <br><br>
        <span class="metric-note">
        The model learns patterns from the supplied dataset. A probability
        is an estimate, not a diagnosis.
        </span>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        values = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }

        input_df = pd.DataFrame([values])
        input_df[ZERO_AS_MISSING] = input_df[ZERO_AS_MISSING].replace(0, np.nan)

        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= .50)

        save_prediction(values, prediction, probability, best_name)
        st.session_state.last_result = {
            "values": values,
            "probability": probability,
            "prediction": prediction,
            "created_at": datetime.now().strftime("%d %b %Y, %H:%M")
        }
        st.success("Assessment completed successfully. Scroll down to view the full result.")

    with result_slot:
        if st.session_state.last_result:
            r = st.session_state.last_result
            st.divider()

            if r["prediction"] == 1:
                st.markdown(
                    f"""<div class="high">
                    <h2>⚠️ Higher predicted risk</h2>
                    <p>The selected model estimates a higher probability of Outcome = 1.</p>
                    <h1>{r['probability']*100:.1f}%</h1>
                    <p><b>Estimated probability</b></p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div class="low">
                    <h2>✓ Lower predicted risk</h2>
                    <p>The selected model estimates a lower probability of Outcome = 1.</p>
                    <h1>{r['probability']*100:.1f}%</h1>
                    <p><b>Estimated probability</b></p>
                    </div>""", unsafe_allow_html=True)

            st.progress(r["probability"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", best_name)
            c2.metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
            c3.metric("Assessment", "Higher risk" if r["prediction"] else "Lower risk")
            c4.metric("Recorded", r["created_at"])

            st.warning(
                "Important: This is a research/educational screening estimate. "
                "It should not be used to diagnose, treat, or rule out diabetes."
            )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.markdown('<div class="section-title">🔎 Model explainability</div>',
                unsafe_allow_html=True)
    st.caption(
        "Permutation importance measures how much model performance changes "
        "when a feature's values are randomly shuffled on the holdout test set."
    )

    imp = bundle["importance"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = imp.sort_values("Importance")
    ax.barh(plot_df["Feature"], plot_df["Importance"])
    ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
    ax.set_title("Global Feature Importance — Holdout Test Set")
    st.pyplot(fig, clear_figure=True)

    st.dataframe(
        imp.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Feature importance is model-specific. It does not prove that a feature "
        "causes diabetes, and it should not be interpreted as medical causality."
    )


# ============================================================
# MODEL LAB
# ============================================================

elif page == "Model Lab":

    st.markdown('<div class="section-title">🧪 Model laboratory</div>',
                unsafe_allow_html=True)

    comparison = pd.DataFrame({
        "Model": list(bundle["cv_scores"].keys()),
        "5-Fold CV ROC-AUC": list(bundle["cv_scores"].values())
    }).sort_values("5-Fold CV ROC-AUC", ascending=False)

    st.dataframe(
        comparison.style.format({"5-Fold CV ROC-AUC": "{:.3f}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()

    a,b,c,d,e = st.columns(5)
    a.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    b.metric("Precision", f"{metrics['precision']*100:.1f}%")
    c.metric("Recall", f"{metrics['recall']*100:.1f}%")
    d.metric("F1", f"{metrics['f1']*100:.1f}%")
    e.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()

    cm = metrics["cm"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(cm)
    ax.set_title("Holdout Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Outcome 0", "Outcome 1"])
    ax.set_yticklabels(["Outcome 0", "Outcome 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    st.pyplot(fig, clear_figure=True)

    report = pd.DataFrame(metrics["report"]).transpose()
    st.subheader("Classification report")
    st.dataframe(report.round(3), use_container_width=True)


# ============================================================
# HISTORY
# ============================================================

elif page == "Prediction History":

    st.markdown('<div class="section-title">🗂 Prediction history</div>',
                unsafe_allow_html=True)

    history = load_history()

    if history.empty:
        st.info("No assessments have been recorded yet.")
    else:
        a,b,c = st.columns(3)
        a.metric("Total assessments", len(history))
        b.metric("Higher-risk results", int(history["prediction"].sum()))
        c.metric("Average probability", f"{history['probability'].mean()*100:.1f}%")

        display = history.copy()
        display["created_at"] = pd.to_datetime(display["created_at"]).dt.strftime("%d %b %Y, %H:%M")
        display["prediction"] = display["prediction"].map({0:"Lower Risk",1:"Higher Risk"})
        display["probability"] = (display["probability"]*100).round(1).astype(str) + "%"

        st.dataframe(display, use_container_width=True, hide_index=True)

        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "glucoguard_prediction_history.csv",
            "text/csv"
        )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.markdown('<div class="section-title">📊 Dataset explorer</div>',
                unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    a.metric("Rows", len(df))
    b.metric("Input features", len(FEATURES))
    c.metric("Outcome 0", int((df[TARGET] == 0).sum()))
    d.metric("Outcome 1", int((df[TARGET] == 1).sum()))

    st.divider()
    st.subheader("Dataset preview")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Outcome distribution")
    counts = df[TARGET].value_counts().sort_index()
    counts.index = ["Outcome 0", "Outcome 1"]
    st.bar_chart(counts)

    st.subheader("Feature statistics")
    st.dataframe(df[FEATURES].describe().T.round(2), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "GlucoGuard AI • Python • Streamlit • scikit-learn • SQLite • "
    "Educational / research project"
)
