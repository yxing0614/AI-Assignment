# app.py - Minimal UI: user types 3 features -> choose model -> predict churn
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Churn quick-predict", layout="centered")
st.title("Churn quick-predict")
st.markdown("Type the customer's features, choose a model, and click **Predict**.")

# Fixed features used for training and prediction
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# --- Try to load local training CSV automatically ---
CANDIDATES = ["Telco.csv", "telco.csv", "data/Telco.csv", "data/telco.csv", "combined.csv"]
train_df = None
for p in CANDIDATES:
    if os.path.exists(p):
        try:
            train_df = pd.read_csv(p)
            st.sidebar.success(f"Loaded training file: {p}")
            break
        except Exception:
            train_df = None

if train_df is None:
    st.sidebar.warning(
        "No local training CSV found. To enable real model predictions, add your training CSV to the repo root named `Telco.csv` (or `data/Telco.csv`) and redeploy.\n\n"
        "The CSV must include these columns: `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn` (values `Yes`/`No` or `1`/`0`)."
    )

# --- Minimal cleaning that matches typical telco data for TotalCharges & churn ---
def clean_train_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # map Churn if object
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0}).fillna(df['Churn'])
    return df

# --- Build pipeline generator ---
def build_pipeline(name: str, random_state: int = 42):
    if name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        return Pipeline([("clf", clf)])
    elif name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=random_state)
        return Pipeline([("clf", clf)])
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "SVM":
        clf = SVC(kernel='rbf', probability=True, random_state=random_state)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    else:
        raise ValueError("Unknown model")

# --- Train the model automatically on app load (if training file exists) ---
if train_df is not None:
    df = clean_train_df(train_df)
    # check required columns
    missing = [c for c in FEATURES + ['Churn'] if c not in df.columns]
    if missing:
        st.sidebar.error(f"Training file is missing columns: {missing}")
        model_ready = False
    else:
        # select features and target
        X = df[FEATURES].copy()
        # fill any remaining NaNs with median
        for c in X.columns:
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median())
        y = df['Churn'].astype(int)
        model_ready = True
else:
    model_ready = False

# Sidebar: model chooser + seed
st.sidebar.header("Model settings")
model_choice = st.sidebar.selectbox("Choose model", ["RandomForest", "DecisionTree", "KNN", "SVM"])
random_state = int(st.sidebar.number_input("Random seed", value=42, min_value=0, step=1))
st.sidebar.write("Model will be trained automatically on app startup using the local CSV.")

# Train and cache pipeline in session_state (so UI prediction is instant)
if model_ready:
    # Only (re)train if not present or model_choice/seed changed
    need_train = ('trained_pipe' not in st.session_state) or (st.session_state.get('model_choice') != model_choice) or (st.session_state.get('random_state') != random_state)
    if need_train:
        pipe = build_pipeline(model_choice, random_state=random_state)
        with st.spinner("Training model on local data..."):
            pipe.fit(X, y)
        st.session_state['trained_pipe'] = pipe
        st.session_state['model_choice'] = model_choice
        st.session_state['random_state'] = random_state
        st.sidebar.success("Model trained")
    else:
        pipe = st.session_state['trained_pipe']
else:
    pipe = None

# --- Input fields for single-customer prediction ---
st.header("Enter customer features (single prediction)")
st.write("Provide the three features below and click **Predict**.")

# default values come from training data median if available
defaults = {'tenure': 12.0, 'MonthlyCharges': 70.0, 'TotalCharges': 1000.0}
if model_ready:
    for c in FEATURES:
        defaults[c] = float(X[c].median())

col1, col2, col3 = st.columns(3)
with col1:
    tenure_val = st.number_input("tenure (months)", value=defaults['tenure'], min_value=0.0, step=1.0, format="%.0f")
with col2:
    monthly_val = st.number_input("MonthlyCharges", value=defaults['MonthlyCharges'], min_value=0.0, step=0.1)
with col3:
    total_val = st.number_input("TotalCharges", value=defaults['TotalCharges'], min_value=0.0, step=0.1)

predict_btn = st.button("Predict")

# --- Prediction logic ---
if predict_btn:
    if pipe is None:
        st.error("No trained model available. Add `Telco.csv` to the repository root (with columns tenure, MonthlyCharges, TotalCharges, Churn) and redeploy.")
    else:
        sample = pd.DataFrame([[tenure_val, monthly_val, total_val]], columns=FEATURES)
        # ensure numeric
        for c in sample.columns:
            sample[c] = pd.to_numeric(sample[c], errors='coerce').fillna(X[c].median() if model_ready else 0)
        pred = pipe.predict(sample)[0]
        prob = None
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(sample)[0][1]
        # Friendly output
        if pred == 1:
            st.markdown("## 🔴 Likely to churn")
        else:
            st.markdown("## 🟢 Unlikely to churn")
        if prob is not None:
            st.write(f"Predicted probability of churn: **{prob:.3f}**")
        st.write("Model used:", model_choice)
