# app.py - Train on fixed features and allow user-keyed single prediction
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

st.set_page_config(page_title="Churn — 3-feature models", layout="wide")
st.title("Churn prediction — Train on `tenure`, `MonthlyCharges`, `TotalCharges`")

# --- Fixed feature list
FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Sidebar controls
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, min_value=0, step=1))
model_choice = st.sidebar.selectbox("Model", ["RandomForest", "DecisionTree", "KNN", "SVM"])
train_btn = st.sidebar.button("Train model (on chosen features)")

# Helper: cleaning that matches typical telco cleaning for TotalCharges
def clean_minimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # Coerce TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # fill by median of column
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df

# Build pipeline
def build_pipeline(name: str):
    if name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        pipe = Pipeline([('clf', clf)])
    elif name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=random_state)
        pipe = Pipeline([('clf', clf)])
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    elif name == "SVM":
        clf = SVC(kernel='rbf', probability=True, random_state=random_state)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    else:
        raise ValueError("Unknown model")
    return pipe

# Upload CSV
uploaded = st.file_uploader("Upload CSV for training (must contain column 'Churn')", type=["csv"])
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Raw data preview")
    st.dataframe(df.head(8))
else:
    st.info("Upload a CSV with 'Churn' column to train. Use the same dataset you used in the notebook for the most comparable results.")

# If dataset provided, prepare X,y using only FEATURES
if df is not None:
    df = clean_minimal(df)
    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        st.error(f"Missing required feature columns: {missing_features}. Please upload a dataset that includes these columns.")
    elif 'Churn' not in df.columns:
        st.error("Dataset must contain 'Churn' column (values 'Yes'/'No' or 1/0).")
    else:
        # Map Churn if necessary
        if df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes':1, 'No':0}).fillna(df['Churn'])
        df['Churn'] = df['Churn'].astype(int)

        # Select features
        X = df[FEATURES].copy()
        # If any NaNs remain in features, fill with median (column-wise)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        y = df['Churn']

        st.write(f"Training with features: {FEATURES}")
        st.write("Feature sample (first 8 rows):")
        st.dataframe(X.head(8))

        # Train when requested
        if train_btn:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size),
                                                                stratify=y, random_state=random_state)
            pipe = build_pipeline(model_choice)
            pipe.fit(X_train, y_train)

            # Save trained pipeline and columns
            st.session_state['pipe'] = pipe
            st.session_state['train_columns'] = list(X.columns)

            # Evaluate
            y_pred = pipe.predict(X_test)
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:,1]
            else:
                try:
                    scores = pipe.decision_function(X_test)
                    ranks = pd.Series(scores).rank(pct=True).values
                    y_proba = ranks
                except Exception:
                    y_proba = None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

            st.subheader("Evaluation on test set")
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{acc:.3f}")
            cols[1].metric("Precision", f"{prec:.3f}")
            cols[2].metric("Recall", f"{rec:.3f}")
            cols[3].metric("F1", f"{f1:.3f}")
            cols[4].metric("ROC AUC", f"{roc:.3f}" if not np.isnan(roc) else "N/A")

            st.write("Confusion matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            with st.expander("Classification report"):
                st.text(classification_report(y_test, y_pred, zero_division=0))

            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                st.line_chart(roc_df.set_index("FPR"))

# ---------------- Single prediction UI ----------------
st.markdown("---")
st.header("Single prediction — enter feature values")

if 'pipe' not in st.session_state:
    st.info("Train a model first (in the sidebar) to enable single predictions.")
else:
    # default values from the training data if available
    defaults = {}
    if df is not None:
        for c in FEATURES:
            defaults[c] = float(df[c].median())
    else:
        defaults = {'tenure': 12.0, 'MonthlyCharges': 70.0, 'TotalCharges': 1000.0}

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure_in = st.number_input("tenure (months)", value=float(defaults['tenure']), min_value=0.0, step=1.0)
    with col2:
        monthly_in = st.number_input("MonthlyCharges", value=float(defaults['MonthlyCharges']), min_value=0.0, step=0.1)
    with col3:
        total_in = st.number_input("TotalCharges", value=float(defaults['TotalCharges']), min_value=0.0, step=0.1)

    if st.button("Predict single customer"):
        sample = pd.DataFrame([[tenure_in, monthly_in, total_in]], columns=FEATURES)
        # Any missing columns already handled; ensure dtype numeric
        for c in FEATURES:
            sample[c] = pd.to_numeric(sample[c], errors='coerce').fillna(df[c].median() if df is not None else 0)
        pipe = st.session_state['pipe']
        pred = pipe.predict(sample)[0]
        prob = None
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(sample)[0][1]
        st.write("Prediction:", "🔴 Churn (1)" if pred == 1 else "🟢 No churn (0)")
        if prob is not None:
            st.write(f"Predicted probability of churn: {prob:.3f}")

# ---------------- Batch prediction download (optional) ----------------
st.markdown("---")
st.header("Batch prediction (upload CSV without 'Churn')")

batch_file = st.file_uploader("Upload CSV for batch prediction (no 'Churn')", type=["csv"], key="batch")
if batch_file is not None:
    if 'pipe' not in st.session_state:
        st.warning("Train a model first.")
    else:
        df_batch = pd.read_csv(batch_file)
        st.write("Preview of uploaded batch data:")
        st.dataframe(df_batch.head(6))

        # Clean and prepare features
        df_batch = clean_minimal(df_batch)
        if 'Churn' in df_batch.columns:
            df_batch = df_batch.drop(columns=['Churn'])
        # Ensure columns exist
        for c in FEATURES:
            if c not in df_batch.columns:
                df_batch[c] = 0
        X_batch = df_batch[FEATURES].copy()
        for c in X_batch.columns:
            X_batch[c] = pd.to_numeric(X_batch[c], errors='coerce').fillna(X_batch[c].median())

        pipe = st.session_state['pipe']
        preds = pipe.predict(X_batch)
        out = df_batch.copy()
        out['Churn_Pred'] = preds
        if hasattr(pipe, "predict_proba"):
            out['Churn_Prob'] = pipe.predict_proba(X_batch)[:,1]

        st.success(f"{len(out)} rows predicted.")
        st.dataframe(out.head(10))
        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime='text/csv')
