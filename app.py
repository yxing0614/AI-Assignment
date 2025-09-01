# app.py
# Streamlit app that mirrors Combined.ipynb training steps (features, cleaning, split) and shows metrics.
# After you verify metrics match your notebook, you can remove the metrics block if you only want prediction UI.

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

st.set_page_config(page_title="Churn — notebook-consistent", layout="centered")
st.title("Churn prediction — training follows Combined.ipynb")

# FEATURES EXACTLY as in your notebook
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Candidate filenames: app will try these in order
CANDIDATES = ["telco2.csv", "Telco.csv", "telco.csv", "data/Telco.csv", "data/telco.csv"]

# Load training CSV (prefer telco2.csv if present)
train_df = None
train_path_used = None
for p in CANDIDATES:
    if os.path.exists(p):
        try:
            train_df = pd.read_csv(p)
            train_path_used = p
            break
        except Exception as e:
            train_df = None

if train_df is None:
    st.error(
        "No training CSV found in the repo root. Put your cleaned CSV into the repository root\n"
        "named `telco2.csv` or `Telco.csv` (the app prefers telco2.csv). The file must contain the columns:\n"
        "`tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`."
    )
    st.stop()

st.sidebar.success(f"Loaded training file: {train_path_used}")

# ---------- Cleaning function (matches Combined.ipynb)
def clean_df_for_app(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # drop customerID if present (not used)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # convert TotalCharges to numeric (notebook used pd.to_numeric)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # fill missing with median (notebook used median fill)
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # map Churn text to 0/1 if needed
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(df['Churn'])
    # ensure numeric types for feature columns
    for c in FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            # fill any remaining NaNs with median (same logic as notebook)
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    return df

df = clean_df_for_app(train_df)

# Check columns presence
missing = [c for c in FEATURES + ['Churn'] if c not in df.columns]
if missing:
    st.error(f"Training CSV is missing required columns: {missing}. Confirm your cleaned file contains them.")
    st.stop()

# Prepare X,y using the exact feature list from notebook
X = df[FEATURES].copy()
y = df['Churn'].astype(int)

# Sidebar: model choice and seed (notebook used random_state=42)
st.sidebar.header("Model & settings (mirror notebook)")
model_choice = st.sidebar.selectbox("Choose model", ["RandomForest", "DecisionTree", "KNN", "SVM"])
test_size = st.sidebar.slider("Test size (to compare with notebook)", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, min_value=0, step=1))
# Note: Combined.ipynb used 0.2 and random_state=42 / stratify y

# Train/test split with stratify (matching notebook)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=random_state, stratify=y)

# Build pipeline matching notebook decisions:
def build_pipeline(name: str, rnd: int = 42):
    if name == "RandomForest":
        # Combined.ipynb used RandomForestClassifier(n_estimators=100, random_state=42) in at least one cell
        clf = RandomForestClassifier(n_estimators=100, random_state=rnd, n_jobs=-1)
        return Pipeline([("clf", clf)])
    elif name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=rnd)
        return Pipeline([("clf", clf)])
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)  # notebook used k around 5 in examples
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "SVM":
        # Notebook had SVM tuning; for parity use default RBF with probability True
        clf = SVC(kernel='rbf', probability=True, random_state=rnd)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    else:
        raise ValueError("Unknown model")

# Train the model (same logic as notebook)
pipe = build_pipeline(model_choice, rnd=random_state)
with st.spinner("Training model (same split/scale choices as notebook)..."):
    pipe.fit(X_train, y_train)

# Evaluate on the test set (show metrics so you can compare with notebook)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.subheader("Evaluation (to compare vs Combined.ipynb)")
st.write(f"Model: **{model_choice}** — Test size: **{test_size}**, Random seed: **{random_state}**")
st.metric("Accuracy", f"{acc:.4f}")
st.metric("Precision", f"{prec:.4f}")
st.metric("Recall", f"{rec:.4f}")
st.metric("F1", f"{f1:.4f}")

with st.expander("Classification report and confusion matrix (detailed)"):
    st.text(classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion matrix (rows: actual, cols: predicted):")
    st.write(cm)

# Save pipeline in session_state for single-sample predictions
st.session_state['pipe'] = pipe
st.session_state['train_columns'] = FEATURES

# ---------------- Single prediction UI (users input the 3 features) ----------------
st.markdown("---")
st.header("Single prediction — enter the 3 features")

# Defaults from training medians (makes testing easier)
defaults = {c: float(X[c].median()) for c in FEATURES}

col1, col2, col3 = st.columns(3)
with col1:
    tenure_val = st.number_input("tenure (months)", value=defaults['tenure'], min_value=0.0, step=1.0, format="%.0f")
with col2:
    monthly_val = st.number_input("MonthlyCharges", value=defaults['MonthlyCharges'], min_value=0.0, step=0.1)
with col3:
    total_val = st.number_input("TotalCharges", value=defaults['TotalCharges'], min_value=0.0, step=0.1)

if st.button("Predict single customer"):
    sample = pd.DataFrame([[tenure_val, monthly_val, total_val]], columns=FEATURES)
    # ensure numeric and fill anything missing
    for c in FEATURES:
        sample[c] = pd.to_numeric(sample[c], errors='coerce').fillna(X[c].median())
    pred = pipe.predict(sample)[0]
    prob = pipe.predict_proba(sample)[0][1] if hasattr(pipe, "predict_proba") else None

    if pred == 1:
        st.markdown("## 🔴 Predicted: Likely to churn")
    else:
        st.markdown("## 🟢 Predicted: Unlikely to churn")
    if prob is not None:
        st.write(f"Predicted probability of churn: **{prob:.3f}**")

# ---------------- Helpful debug info (optional, remove later) ----------------
with st.expander("Debug: training data summary (useful for matching notebook)"):
    st.write("Training file used:", train_path_used)
    st.write("Training shape (rows,cols):", df.shape)
    st.write("Features used:", FEATURES)
    st.write("Feature medians:")
    st.write({c: float(X[c].median()) for c in FEATURES})
    st.write("Churn value counts:")
    st.write(y.value_counts())
    st.write("First 5 rows of training features:")
    st.dataframe(X.head())
