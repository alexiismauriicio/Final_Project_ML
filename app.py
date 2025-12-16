import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from preprocess_telco import DataFramePreparer, CustomOneHotEncoder, num_pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ================================
# 1. CONFIGURACIÓN GENERAL
# ================================
st.set_page_config(
    page_title="Telco Churn – Final Exam",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set(style="whitegrid")
st.title("📘 Telco Customer Churn – Streamlit App")

st.markdown("""
This application loads **three previously trained models** (Logistic Regression, CatBoost, and Soft Voting),
each with two versions:

- A version with **all features**
- A version with **Top 15 features** (based on feature importance)

It allows you to:
- Explore the dataset (EDA)
- Compare versions of the selected model (F1, AUC, Accuracy)
- View the confusion matrix and feature importance
- Perform an **individual prediction** for a new customer
""")


# ================================
# 2. CONSTANTES
# ================================

MODEL_LIST = [
    "Logistic Regression",
    "CatBoost",
    "Voting Soft",
]

VERSION_LIST = [
    "All features",
    "Top 15 features",
]


# ================================
# 3. FUNCIONES AUXILIARES
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


@st.cache_resource
def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")


@st.cache_resource
def load_models():
    models = {
        ("Logistic Regression", "All features"):
            joblib.load("models/logreg_all_features.pkl"),
        ("Logistic Regression", "Top 15 features"):
            joblib.load("models/logreg_top15.pkl"),

        ("CatBoost", "All features"):
            joblib.load("models/catboost_all_features.pkl"),
        ("CatBoost", "Top 15 features"):
            joblib.load("models/catboost_top15.pkl"),

        ("Voting Soft", "All features"):
            joblib.load("models/voting_soft_all_features.pkl"),
        ("Voting Soft", "Top 15 features"):
            joblib.load("models/voting_soft_top15.pkl"),
    }
    return models


def split_features_target(df):
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].copy()
    return X, y


def to_binary_yes(y):
    # Convierte 'Yes'/'No' a 1/0 para AUC
    return (y == "Yes").astype(int)


def transform_top_features(preprocessor, X):
    """Aplica el preprocesador y selecciona las TOP_FEATURES."""
    X_prep = preprocessor.transform(X)

    # Si el preprocesador devuelve array, lo convertimos a DataFrame
    if not isinstance(X_prep, pd.DataFrame):
        cols = getattr(preprocessor, "_columns", [f"f{i}" for i in range(X_prep.shape[1])])
        X_prep = pd.DataFrame(X_prep, columns=cols, index=X.index)

    X_top = X_prep[TOP_FEATURES]
    return X_top


def evaluate_model(df, model, version, preprocessor):
    X, y = split_features_target(df)

    if version == "All features":
        # El modelo ya incluye el preprocesador como Pipeline
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    else:
        # Versión Top 15: usamos preprocessor externo
        X_top = transform_top_features(preprocessor, X)
        y_pred = model.predict(X_top)
        y_proba = model.predict_proba(X_top)[:, 1] if hasattr(model, "predict_proba") else None

    y_bin = to_binary_yes(y)

    acc = accuracy_score(y, y_pred)
    f1_yes = f1_score(y, y_pred, pos_label="Yes")
    auc = roc_auc_score(y_bin, y_proba) if y_proba is not None else np.nan
    cm = confusion_matrix(y, y_pred, labels=["No", "Yes"])

    return acc, f1_yes, auc, cm

# ===== Importancia de características precomputada (desde el notebook) =====
feat_imp = joblib.load("models/feat_imp.pkl")
# Normalizamos el nombre de la columna para que la función de plot funcione
if "Importance Score" in feat_imp.columns and "Importance" not in feat_imp.columns:
    feat_imp = feat_imp.rename(columns={"Importance Score": "Importance"})

TOP_FEATURES = feat_imp.head(15)["Feature"].tolist()

def get_feature_importance():
    """
    Devuelve la importancia de características calculada en el notebook
    (feat_imp cargado desde feat_imp.pkl).
    """
    return feat_imp


def plot_feature_importance(fi_df, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(15), ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)


# ================================
# 4. CARGA GLOBAL
# ================================
df = load_data()
preprocessor = load_preprocessor()
models = load_models()

# ================================
# 5. SIDEBAR – SELECCIÓN DE MODELO
# ================================
st.sidebar.header("⚙️ Model settings")

selected_model = st.sidebar.selectbox("Model:", MODEL_LIST, index=2)  # default: Soft Voting
selected_version = st.sidebar.radio("Version:", VERSION_LIST, index=0)

st.sidebar.markdown("---")
st.sidebar.write("**Current model:**")
st.sidebar.write(f"- {selected_model} ({selected_version})")

import os

MODEL_FILES = {
    ("Logistic Regression", "All features"): "models/logreg_all_features.pkl",
    ("Logistic Regression", "Top 15 features"): "models/logreg_top15.pkl",
    ("CatBoost", "All features"): "models/catboost_all_features.pkl",
    ("CatBoost", "Top 15 features"): "models/catboost_top15.pkl",
    ("Voting Soft", "All features"): "models/voting_soft_all_features.pkl",
    ("Voting Soft", "Top 15 features"): "models/voting_soft_top15.pkl",
}

def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

# --- Model size (MB) ---
model_path = MODEL_FILES.get((selected_model, selected_version))
if model_path and os.path.exists(model_path):
    size_mb = bytes_to_mb(os.path.getsize(model_path))
    st.sidebar.write(f"**Model size:** {size_mb:.2f} MB")
else:
    st.sidebar.caption("Model size: not available (file not found).")


# ================================
# 6. TABS PRINCIPALES
# ================================
tab1, tab2, tab3 = st.tabs(["📈 EDA", "📊 Models & Metrics", "🔮 Individual prediction"])

# ========== TAB 1: EDA ==========
with tab1:
    st.subheader("Exploratory Data Analysis (EDA)")

    # ---------------- KEY METRICS ----------------
    st.markdown("### Key metrics")

    data_for_kpis = df.copy()

    # Base calculations
    total_customers = len(data_for_kpis)
    churn_counts = data_for_kpis["Churn"].value_counts()
    yes_count = churn_counts.get("Yes", 0)
    churn_rate = yes_count / total_customers if total_customers > 0 else 0.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📊 Total records",
            f"{total_customers:,}",
            help="Total number of customers in the dataset"
        )

    with col2:
        st.metric(
            "✅ Churn = Yes",
            f"{yes_count:,}",
            help="Number of customers who left the service"
        )

    with col3:
        st.metric(
            "📉 Churn rate",
            f"{churn_rate:.1%}",
            help="Percentage of customers with Churn = Yes"
        )

    # ---------------- DATASET PREVIEW ----------------
    st.markdown("### Dataset preview")
    st.dataframe(df.head())

    # ---------------- TARGET DISTRIBUTION ----------------
    st.markdown("### Target distribution (Churn)")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x="Churn", data=df, ax=ax)
    ax.set_title("Churn distribution")
    st.pyplot(fig)

    # ---------------- INTERACTIVE HISTOGRAMS ----------------
    st.markdown("### Distribution explorer")

    # Numeric
    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    # Categorical, excluding customerID so it doesn't freeze the app
    cat_cols_raw = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [
        c for c in cat_cols_raw
        if c.lower() not in ["customerid"]
    ]

    var_type = st.radio(
        "Select the variable type you want to explore:",
        ["Numeric", "Categorical"],
        horizontal=True,
    )

    if var_type == "Numeric" and len(num_cols) > 0:
        selected_num = st.selectbox(
            "Select a numeric variable:",
            num_cols,
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_num}")
        st.pyplot(fig)

    elif var_type == "Categorical" and len(cat_cols) > 0:
        selected_cat = st.selectbox(
            "Select a categorical variable:",
            cat_cols,
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x=df[selected_cat], ax=ax)
        ax.set_title(f"Distribution of {selected_cat}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif var_type == "Categorical" and len(cat_cols) == 0:
        st.info("No categorical variables available to plot (excluding customerID).")

    # ---------------- CORRELATION MATRIX ----------------
    st.markdown("### Correlation matrix (numeric variables only)")
    if len(num_cols) > 0:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation matrix")
        st.pyplot(fig)
# ========== TAB 2: MODELOS Y MÉTRICAS ==========
with tab2:
    st.subheader("Model and version comparison")

    model = models[(selected_model, selected_version)]

    acc, f1_yes, auc, cm = evaluate_model(df, model, selected_version, preprocessor)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("F1 (Yes)", f"{f1_yes:.3f}")
    c3.metric("AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

    st.markdown("### Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["True: No", "True: Yes"],
        ax=ax
    )
    ax.set_title("Confusion matrix")
    st.pyplot(fig)

    # --- All vs Top 15 comparison for the selected model ---
    st.markdown("### Comparison: All features vs Top 15")

    acc_all, f1_all, auc_all, _ = evaluate_model(
        df, models[(selected_model, "All features")], "All features", preprocessor
    )
    acc_top, f1_top, auc_top, _ = evaluate_model(
        df, models[(selected_model, "Top 15 features")], "Top 15 features", preprocessor
    )

    comp_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 (Yes)", "AUC"] * 2,
        "Version": ["All features"] * 3 + ["Top 15 features"] * 3,
        "Score": [acc_all, f1_all, auc_all, acc_top, f1_top, auc_top]
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=comp_df,
        x="Metric",
        y="Score",
        hue="Version",
        ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Performance comparison – {selected_model}")
    st.pyplot(fig)

    # --- Feature importance ---
    st.markdown("### Feature importance (Top 15)")
    fi_df = get_feature_importance()
    plot_feature_importance(fi_df, f"Feature importance – {selected_model} (Top 15)")

# ========== TAB 3: PREDICCIÓN INDIVIDUAL ==========
with tab3:
    st.subheader("Individual churn prediction")

    X_full, _ = split_features_target(df)

    st.markdown("Fill in the customer information:")

    # -------------------- SECTION 1: CUSTOMER INFO --------------------
    st.divider()
    st.markdown("### 1) Customer info")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            df["gender"].unique(),
            help="Whether the customer is a male or a female"
        )
        senior = st.selectbox(
            "SeniorCitizen",
            sorted(df["SeniorCitizen"].unique()),
            help="Whether the customer is a senior citizen or not (1, 0)"
        )
        partner = st.selectbox(
            "Partner",
            df["Partner"].unique(),
            help="Whether the customer has a partner or not (Yes, No)"
        )


    with col2:

        dependents = st.selectbox(
            "Dependents",
            df["Dependents"].unique(),
            help="Whether the customer has dependents or not (Yes, No)"
        )
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=int(df["tenure"].median()),
            help="Number of months the customer has stayed with the company"
        )

    # -------------------- SECTION 2: SUBSCRIBED SERVICES --------------------
    st.divider()
    st.markdown("### 2) Subscribed services")

    colA, colB, colC = st.columns(3)

    with colA:
        phone_service = st.selectbox(
            "PhoneService",
            df["PhoneService"].unique(),
            help="Whether the customer has a phone service or not (Yes, No)"
        )
        multiple_lines = st.selectbox(
            "MultipleLines",
            df["MultipleLines"].unique(),
            help="Whether the customer has multiple lines or not (Yes, No, No phone service)"
        )
        internet_service = st.selectbox(
            "InternetService",
            df["InternetService"].unique(),
            help="Customer’s internet service provider (DSL, Fiber optic, No)"
        )

    with colB:

        online_security = st.selectbox(
            "OnlineSecurity",
            df["OnlineSecurity"].unique(),
            help="Whether the customer has online security or not (Yes, No, No internet service)"
        )
        online_backup = st.selectbox(
            "OnlineBackup",
            df["OnlineBackup"].unique(),
            help="Whether the customer has online backup or not (Yes, No, No internet service)"
        )
        device_protection = st.selectbox(
            "DeviceProtection",
            df["DeviceProtection"].unique(),
            help="Whether the customer has device protection or not (Yes, No, No internet service)"
        )

    with colC:
        tech_support = st.selectbox(
            "TechSupport",
            df["TechSupport"].unique(),
            help="Whether the customer has tech support or not (Yes, No, No internet service)"
        )
        streaming_tv = st.selectbox(
            "StreamingTV",
            df["StreamingTV"].unique(),
            help="Whether the customer has streaming TV or not (Yes, No, No internet service)"
        )
        streaming_movies = st.selectbox(
            "StreamingMovies",
            df["StreamingMovies"].unique(),
            help="Whether the customer has streaming movies or not (Yes, No, No internet service)"
        )

    # -------------------- SECTION 3: CONTRACT & BILLING --------------------
    st.divider()
    st.markdown("### 3) Contract & billing")

    colD, colE = st.columns(2)

    with colD:
        contract = st.selectbox(
            "Contract",
            df["Contract"].unique(),
            help="The contract term of the customer (Month-to-month, One year, Two year)"
        )
        paperless = st.selectbox(
            "PaperlessBilling",
            df["PaperlessBilling"].unique(),
            help="Whether the customer has paperless billing or not (Yes, No)"
        )
        payment = st.selectbox(
            "PaymentMethod",
            df["PaymentMethod"].unique(),
            help="The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))"
        )

    with colE:
        monthly_charges = st.number_input(
            "MonthlyCharges",
            min_value=0.0,
            max_value=200.0,
            value=float(df["MonthlyCharges"].median()),
            step=1.0,
            help="The amount charged to the customer monthly"
        )
        total_charges = st.number_input(
            "TotalCharges",
            min_value=0.0,
            max_value=10000.0,
            value=float(df["TotalCharges"].median()),
            step=10.0,
            help="The total amount charged to the customer"
        )

    # -------------------- BUILD INPUT ROW --------------------
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    st.markdown("### Observation preview")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🔮 Predict churn"):
        model = models[(selected_model, selected_version)]

        # --- Run prediction depending on model version ---
        if selected_version == "All features":
            # The model is a full Pipeline (includes preprocessing)
            y_pred = model.predict(input_df)[0]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(input_df)[0, 1]
            else:
                y_proba = None
        else:
            # Top 15 version (needs explicit preprocessing)
            X_top_single = transform_top_features(preprocessor, input_df)
            y_pred = model.predict(X_top_single)[0]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_top_single)[0, 1]
            else:
                y_proba = None

        # --- Show main prediction ---
        st.success(f"**Predicted class:** {y_pred}")
        if y_proba is not None:
            st.info(f"**Churn probability = Yes:** {y_proba:.3f}")
        else:
            st.info("This model does not output probabilities (class only).")

        # --- Recommendations if churn risk is high ---
        if y_pred == "Yes":
            st.warning("⚠️ This customer has a **high churn risk**.")

            st.markdown("#### Retention recommendations")

            recs = []

            # Very flexible contract → propose more stable plan with benefits
            if contract == "Month-to-month":
                recs.append(
                    "- Offer an **upgrade to a 1- or 2-year contract** with a discount on the monthly fee "
                    "or additional benefits to increase loyalty."
                )

            # High monthly charges → review plan
            if monthly_charges > df['MonthlyCharges'].median():
                recs.append(
                    "- Review whether the current plan matches the customer's real usage and consider a "
                    "**plan or package adjustment** to reduce perceived cost."
                )

            # No security / no support → value-added services
            if online_security == "No" or tech_support == "No":
                recs.append(
                    "- Propose **value-added services** (OnlineSecurity, TechSupport) as part of a "
                    "retention campaign or promotional bundle."
                )

            # Payment method electronic check → promote automatic methods
            if payment == "Electronic check":
                recs.append(
                    "- Suggest switching to an **automatic payment method** "
                    "(Bank transfer or Credit card) with a small **loyalty incentive**."
                )

            # New customer (low tenure) → onboarding actions
            if tenure < 12:
                recs.append(
                    "- Implement an **early-stage engagement strategy** "
                    "(proactive calls, personalized emails, usage tutorials) to strengthen the relationship."
                )

            # Fallback: generic recommendation
            if not recs:
                recs.append(
                    "- Review the recent interaction history and offer a "
                    "**personalized retention plan** based on usage and complaints."
                )

            for r in recs:
                st.markdown(r)
        else:
            st.success(
                "✅ The model indicates a **low churn risk**. "
                "Keep providing good service and periodically monitor this customer."
            )
    # ===== Footer =====
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>💡 Telco Customer Churn | By Alexis Mauricio Garzón </small>
    </div>
    """, unsafe_allow_html=True)
