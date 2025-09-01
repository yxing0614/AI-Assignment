# app.py
# Minimal UI: trains models (matching your notebook) on load, then lets user input 3 features to predict churn.
# No debug or metric panels — only input -> model selection -> prediction.

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score

st.set_page_config(page_title="Churn quick-predict", layout="centered")
st.title("Churn quick-predict")

# Features and candidate training filenames
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CANDIDATES = ["telco2.csv", "Telco.csv", "telco.csv", "data/Telco.csv", "data/telco.csv"]

# Load CSV
train_df = None
train_path = None
for p in CANDIDATES:
    if os.path.exists(p):
        try:
            train_df = pd.read_csv(p)
            train_path = p
            break
        except Exception:
            train_df = None

if train_df is None:
    st.error(
        "No training CSV found in the repository root. Add your cleaned CSV named `telco2.csv` or `Telco.csv` "
        "containing columns: `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`."
    )
    st.stop()

# Minimal cleaning (matching notebook)
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(df['Churn'])
    for c in FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    return df

df = clean_df(train_df)
missing_cols = [c for c in FEATURES + ['Churn'] if c not in df.columns]
if missing_cols:
    st.error(f"Training CSV missing columns: {missing_cols}")
    st.stop()

X = df[FEATURES].copy()
y = df['Churn'].astype(int)

# Train/test split (same as notebook)
test_size = 0.2
split_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_seed, stratify=y)

# Train models (matching your notebook hyperparameters)
@st.cache_data(show_spinner=False)
def train_all_models(X_train, y_train, X_train_full, y_full):
    models = {}
    scalers = {}

    # RandomForest (notebook params)
    rf = RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        n_jobs=-1,
        random_state=50,
        max_features='sqrt',
        max_leaf_nodes=30
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Decision Tree (notebook params)
    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    # KNN: scale on train, search k=1..20 by precision on test-like split
    scaler_knn = StandardScaler()
    X_train_scaled = scaler_knn.fit_transform(X_train)
    # find best k by precision (use simple approach like notebook)
    k_values = list(range(1, 21))
    precision_scores = []
    for k in k_values:
        knn_tmp = KNeighborsClassifier(n_neighbors=k)
        knn_tmp.fit(X_train_scaled, y_train)
        # for search we compare on provided X_train_scaled (consistent with notebook behavior)
        preds_tmp = knn_tmp.predict(X_train_scaled)
        precision_scores.append(precision_score(y_train, preds_tmp, zero_division=0))
    best_k = k_values[int(np.argmax(precision_scores))]
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_scaled, y_train)
    models['KNN'] = knn_final
    scalers['KNN'] = scaler_knn

    # SVM: scale on full features, grid search with cv=5
    scaler_svm = StandardScaler()
    X_full_scaled = scaler_svm.fit_transform(X_train_full)  # note: notebook did scale before splitting for SVM
    # split scaled for gridsearch
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_full_scaled, y_full, test_size=test_size, random_state=split_seed, stratify=y_full)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1, 1]}
    base_svc = SVC(kernel='rbf', probability=True, random_state=50)
    grid = GridSearchCV(estimator=base_svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
    grid.fit(Xs_train, ys_train)
    best_params = grid.best_params_
    svm_final = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True, random_state=50)
    svm_final.fit(Xs_train, ys_train)
    models['SVM'] = svm_final
    scalers['SVM'] = scaler_svm

    return models, scalers, best_k

# Use training data: Note SVM uses scaling on full X (to match notebook)
models, scalers, knn_k = train_all_models(X_train, y_train, X, y)

# Save into session state for prediction use
st.session_state['models'] = models
st.session_state['scalers'] = scalers
st.session_state['knn_k'] = knn_k

# --- Single prediction UI ---
st.header("Enter customer features")
col1, col2, col3 = st.columns(3)
with col1:
    tenure_val = st.number_input("tenure (months)", value=float(X['tenure'].median()), min_value=0.0, step=1.0, format="%.0f")
with col2:
    monthly_val = st.number_input("MonthlyCharges", value=float(X['MonthlyCharges'].median()), min_value=0.0, step=0.1)
with col3:
    total_val = st.number_input("TotalCharges", value=float(X['TotalCharges'].median()), min_value=0.0, step=0.1)

model_select = st.selectbox("Choose model to predict with", ["RandomForest", "DecisionTree", "KNN", "SVM"])
if st.button("Predict"):
    sample = pd.DataFrame([[tenure_val, monthly_val, total_val]], columns=FEATURES)
    for c in sample.columns:
        sample[c] = pd.to_numeric(sample[c], errors='coerce').fillna(X[c].median())

    chosen = st.session_state['models'].get(model_select)
    proba = None
    if model_select == 'KNN':
        scaler = st.session_state['scalers']['KNN']
        sample_t = scaler.transform(sample)
        pred = chosen.predict(sample_t)[0]
        try:
            proba = chosen.predict_proba(sample_t)[0][1]
        except Exception:
            proba = None
    elif model_select == 'SVM':
        scaler = st.session_state['scalers']['SVM']
        sample_t = scaler.transform(sample)  # SVM scaler fit on full X
        pred = chosen.predict(sample_t)[0]
        proba = chosen.predict_proba(sample_t)[0][1]
    else:
        pred = chosen.predict(sample)[0]
        try:
            proba = chosen.predict_proba(sample)[0][1]
        except Exception:
            proba = None

    if pred == 1:
        st.markdown("## 🔴 Predicted: Likely to churn")
    else:
        st.markdown("## 🟢 Predicted: Unlikely to churn")
    if proba is not None:
        st.write(f"Predicted probability of churn: **{proba:.3f}**")
