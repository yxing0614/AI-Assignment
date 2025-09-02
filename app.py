# app.py
# Minimal UI: trains models on cleaned dataset, then lets user input 3 features to predict churn.

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score

st.set_page_config(page_title="Churn quick-predict", layout="centered")
st.title("Churn quick-predict")

# --- Config ---
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
DATA_FILE = "Telco.csv"   # ✅ use cleaned dataset

# --- Load dataset ---
if not os.path.exists(DATA_FILE):
    st.error(
        f"Training CSV `{DATA_FILE}` not found in the repository root. "
        "Please add it with columns: tenure, MonthlyCharges, TotalCharges, and Churn."
    )
    st.stop()

df = pd.read_csv(DATA_FILE)

# --- Verify required columns ---
missing_cols = [c for c in FEATURES + ['Churn'] if c not in df.columns]
if missing_cols:
    st.error(f"Training CSV missing columns: {missing_cols}")
    st.stop()

X = df[FEATURES].copy()
y = df['Churn'].astype(int)

# Train/test split (same as notebook)
test_size = 0.2
split_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=split_seed, stratify=y
)

# --- Train models ---
@st.cache_resource(show_spinner=False)
def train_all_models(X_train, y_train, X_full, y_full):
    models = {}
    scalers = {}

    # RandomForest
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

    # Decision Tree
    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    # KNN
    scaler_knn = StandardScaler()
    X_train_scaled = scaler_knn.fit_transform(X_train)
    k_values = list(range(1, 21))
    precision_scores = []
    for k in k_values:
        knn_tmp = KNeighborsClassifier(n_neighbors=k)
        knn_tmp.fit(X_train_scaled, y_train)
        preds_tmp = knn_tmp.predict(X_train_scaled)
        precision_scores.append(precision_score(y_train, preds_tmp, zero_division=0))
    best_k = k_values[int(np.argmax(precision_scores))]
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_scaled, y_train)
    models['KNN'] = knn_final
    scalers['KNN'] = scaler_knn

    # SVM (fixed best params)
    scaler_svm = StandardScaler()
    X_scaled = scaler_svm.fit_transform(X_full)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_scaled, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    svm_final = SVC(kernel='rbf', C=100, gamma=1, probability=True, random_state=50)
    svm_final.fit(Xs_train, ys_train)
    models['SVM'] = svm_final
    scalers['SVM'] = scaler_svm

    return models, scalers, best_k

models, scalers, knn_k = train_all_models(X_train, y_train, X, y)

# --- UI for single prediction ---
st.header("Enter customer features")
col1, col2, col3 = st.columns(3)
with col1:
    tenure_val = st.number_input(
        "tenure (months)", value=float(X['tenure'].median()), min_value=0.0, step=1.0, format="%.0f"
    )
with col2:
    monthly_val = st.number_input(
        "MonthlyCharges", value=float(X['MonthlyCharges'].median()), min_value=0.0, step=0.1
    )
with col3:
    total_val = st.number_input(
        "TotalCharges", value=float(X['TotalCharges'].median()), min_value=0.0, step=0.1
    )

model_select = st.selectbox("Choose model to predict with", ["RandomForest", "DecisionTree", "KNN", "SVM"])

if st.button("Predict"):
    sample = pd.DataFrame([[tenure_val, monthly_val, total_val]], columns=FEATURES)

    chosen = models[model_select]
    proba = None

    if model_select == 'KNN':
        scaler = scalers['KNN']
        sample_t = scaler.transform(sample)
        pred = chosen.predict(sample_t)[0]
        try:
            proba = chosen.predict_proba(sample_t)[0][1]
        except Exception:
            proba = None
    elif model_select == 'SVM':
        scaler = scalers['SVM']
        sample_t = scaler.transform(sample)
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
