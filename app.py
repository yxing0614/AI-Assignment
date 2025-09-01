# app.py
# Streamlit app that reproduces the training code from your Combined.ipynb for:
# RandomForest, DecisionTree, KNN (k search by precision), and SVM (GridSearchCV),
# using features = ['tenure','MonthlyCharges','TotalCharges'] and the same splits/seeds.

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

st.set_page_config(page_title="Churn - Notebook-matched models", layout="wide")
st.title("Churn models — replicate Combined.ipynb exactly")

# Exact features used in your notebook
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Candidate CSV filenames (prefer telco2.csv)
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
    st.error("No training CSV found (expected telco2.csv or Telco.csv in repo root). Add it and rerun.")
    st.stop()

st.sidebar.success(f"Loaded training file: {train_path}")

# ---------------- cleaning to match notebook ----------------
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

# Verify required columns exist
missing_cols = [c for c in FEATURES + ['Churn'] if c not in df.columns]
if missing_cols:
    st.error(f"Training CSV is missing required columns: {missing_cols}")
    st.stop()

# Prepare X,y
X = df[FEATURES].copy()
y = df['Churn'].astype(int)

# Sidebar controls (keep same split/seed as your notebook)
st.sidebar.header("Settings (match notebook)")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, 0.05)
split_seed = int(st.sidebar.number_input("Split random_state", value=42, min_value=0, step=1))
st.sidebar.write("Models are trained using the same splits/seeds as your notebook.")

# Split (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=split_seed, stratify=y)

# ---------------- Train RandomForest (exact notebook params) ----------------
st.header("Random Forest (notebook params)")
with st.spinner("Training RandomForest..."):
    rf = RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        n_jobs=-1,
        random_state=50,   # note: your notebook used 50 inside RF
        max_features='sqrt',
        max_leaf_nodes=30
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:,1] if hasattr(rf, "predict_proba") else None

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, zero_division=0)
rf_rec = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{rf_acc:.4f}")
col2.metric("Precision", f"{rf_prec:.4f}")
col3.metric("Recall", f"{rf_rec:.4f}")
col4.metric("F1", f"{rf_f1:.4f}")

with st.expander("RandomForest: classification report & confusion matrix"):
    st.text(classification_report(y_test, rf_pred, zero_division=0))
    st.write("Confusion matrix:")
    st.write(confusion_matrix(y_test, rf_pred))
st.write("Note: RF used random_state=50 (internal) while the split used random_state={}".format(split_seed))

# Cache RF in session
if 'models' not in st.session_state:
    st.session_state['models'] = {}
st.session_state['models']['RandomForest'] = rf

# ---------------- Train Decision Tree (exact notebook params) ----------------
st.header("Decision Tree (notebook params)")
with st.spinner("Training DecisionTree..."):
    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:,1] if hasattr(dt, "predict_proba") else None

dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, zero_division=0)
dt_rec = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)

dcol1, dcol2, dcol3, dcol4 = st.columns(4)
dcol1.metric("Accuracy", f"{dt_acc:.4f}")
dcol2.metric("Precision", f"{dt_prec:.4f}")
dcol3.metric("Recall", f"{dt_rec:.4f}")
dcol4.metric("F1", f"{dt_f1:.4f}")

with st.expander("DecisionTree: classification report & confusion matrix"):
    st.text(classification_report(y_test, dt_pred, zero_division=0))
    st.write("Confusion matrix:")
    st.write(confusion_matrix(y_test, dt_pred))

st.session_state['models']['DecisionTree'] = dt

# ---------------- Train KNN (k search by precision on test set) ----------------
st.header("K-Nearest Neighbors (search k=1..20 by precision on test set)")
with st.spinner("Searching best k and training KNN..."):
    scaler_knn = StandardScaler()
    X_train_scaled = scaler_knn.fit_transform(X_train)
    X_test_scaled = scaler_knn.transform(X_test)

    k_values = list(range(1, 21))
    precision_scores = []
    for k in k_values:
        knn_tmp = KNeighborsClassifier(n_neighbors=k)
        knn_tmp.fit(X_train_scaled, y_train)
        y_tmp = knn_tmp.predict(X_test_scaled)
        prec_tmp = precision_score(y_test, y_tmp, zero_division=0)
        precision_scores.append(prec_tmp)

    best_k = k_values[int(np.argmax(precision_scores))]
    # train final knn with best_k
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_scaled, y_train)

    knn_pred = knn_final.predict(X_test_scaled)
    knn_proba = None
    try:
        knn_proba = knn_final.predict_proba(X_test_scaled)[:,1]
    except Exception:
        knn_proba = None

knn_acc = accuracy_score(y_test, knn_pred)
knn_prec = precision_score(y_test, knn_pred, zero_division=0)
knn_rec = recall_score(y_test, knn_pred, zero_division=0)
knn_f1 = f1_score(y_test, knn_pred, zero_division=0)

kcol1, kcol2, kcol3, kcol4 = st.columns(4)
kcol1.metric("Accuracy", f"{knn_acc:.4f}")
kcol2.metric("Precision", f"{knn_prec:.4f}")
kcol3.metric("Recall", f"{knn_rec:.4f}")
kcol4.metric("F1", f"{knn_f1:.4f}")

with st.expander("KNN: details"):
    st.write("Best k by precision on test set:", int(best_k))
    st.line_chart(pd.DataFrame({'k': k_values, 'precision': precision_scores}).set_index('k'))
    st.text(classification_report(y_test, knn_pred, zero_division=0))
    st.write("Confusion matrix:")
    st.write(confusion_matrix(y_test, knn_pred))

st.session_state['models']['KNN'] = knn_final
# store scaler used for KNN
if 'scalers' not in st.session_state:
    st.session_state['scalers'] = {}
st.session_state['scalers']['KNN'] = scaler_knn

# ---------------- Train SVM (GridSearchCV on scaled data) ----------------
st.header("SVM (GridSearchCV then final model)")
with st.spinner("Running GridSearchCV for SVM and training final SVM..."):
    scaler_svm = StandardScaler()
    X_scaled = scaler_svm.fit_transform(X)   # notebook did scaling before splitting for SVM
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_scaled, y, test_size=float(test_size), random_state=split_seed, stratify=y)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1]
    }
    base_svc = SVC(kernel='rbf', probability=True, random_state=50)
    grid = GridSearchCV(estimator=base_svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
    grid.fit(Xs_train, ys_train)

    best_params = grid.best_params_
    best_score_cv = grid.best_score_

    # train final SVM with best params (mirror notebook)
    svm_final = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True, random_state=50)
    svm_final.fit(Xs_train, ys_train)
    svm_pred = svm_final.predict(Xs_test)
    svm_proba = svm_final.predict_proba(Xs_test)[:,1]

svm_acc = accuracy_score(ys_test, svm_pred)
svm_prec = precision_score(ys_test, svm_pred, zero_division=0)
svm_rec = recall_score(ys_test, svm_pred, zero_division=0)
svm_f1 = f1_score(ys_test, svm_pred, zero_division=0)

scol1, scol2, scol3, scol4 = st.columns(4)
scol1.metric("Accuracy", f"{svm_acc:.4f}")
scol2.metric("Precision", f"{svm_prec:.4f}")
scol3.metric("Recall", f"{svm_rec:.4f}")
scol4.metric("F1", f"{svm_f1:.4f}")

with st.expander("SVM: details"):
    st.write("Best params (GridSearchCV):", best_params)
    st.write("Best CV accuracy:", float(best_score_cv))
    st.text(classification_report(ys_test, svm_pred, zero_division=0))
    st.write("Confusion matrix:")
    st.write(confusion_matrix(ys_test, svm_pred))

st.session_state['models']['SVM'] = svm_final
st.session_state['scalers']['SVM'] = scaler_svm

# ---------------- Single prediction UI ----------------
st.markdown("---")
st.header("Single prediction — enter the features and choose a model")

colA, colB, colC = st.columns(3)
with colA:
    tenure_val = st.number_input("tenure (months)", value=float(X['tenure'].median()), min_value=0.0, step=1.0, format="%.0f")
with colB:
    monthly_val = st.number_input("MonthlyCharges", value=float(X['MonthlyCharges'].median()), min_value=0.0, step=0.1)
with colC:
    total_val = st.number_input("TotalCharges", value=float(X['TotalCharges'].median()), min_value=0.0, step=0.1)

model_select = st.selectbox("Model for prediction", ["RandomForest", "DecisionTree", "KNN", "SVM"])
if st.button("Predict"):
    sample = pd.DataFrame([[tenure_val, monthly_val, total_val]], columns=FEATURES)
    # Ensure numeric
    for c in sample.columns:
        sample[c] = pd.to_numeric(sample[c], errors='coerce').fillna(X[c].median())

    # Get model & handle scaling where needed
    chosen = st.session_state['models'].get(model_select)
    if chosen is None:
        st.error("Model not trained/available.")
    else:
        if model_select == 'KNN':
            scaler = st.session_state['scalers']['KNN']
            sample_trans = scaler.transform(sample)
            pred = chosen.predict(sample_trans)[0]
            proba = None
            try:
                proba = chosen.predict_proba(sample_trans)[0][1]
            except Exception:
                proba = None
        elif model_select == 'SVM':
            scaler = st.session_state['scalers']['SVM']
            sample_trans = scaler.transform(sample)  # SVM scaler was fit on whole X
            pred = chosen.predict(sample_trans)[0]
            proba = chosen.predict_proba(sample_trans)[0][1]
        else:
            pred = chosen.predict(sample)[0]
            proba = None
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

# ---------------- Debug / info box ----------------
with st.expander("Debug: training data & notes"):
    st.write("Training file used:", train_path)
    st.write("Training shape:", df.shape)
    st.write("Feature medians:", {c: float(X[c].median()) for c in FEATURES})
    st.write("Churn counts:", y.value_counts())
    st.write("Note: KNN best_k was chosen by evaluating precision on the test set (this mirrors your notebook).")
    st.write("Note: SVM GridSearchCV was run with cv=5 and scoring='accuracy' (mirror of notebook).")
