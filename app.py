# app.py - simple beginner-friendly Streamlit app to train 4 models
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="4-model trainer", layout="wide")
st.title("Train 4 Models — RandomForest / DecisionTree / KNN / SVM")

st.markdown("""
Upload a CSV with a `Churn` column (values `Yes`/`No` or `1`/`0`).  
After training you can upload another CSV (without `Churn`) to get batch predictions.
""")

uploaded = st.file_uploader("Upload CSV for training (must contain 'Churn')", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Data preview")
    st.dataframe(df.head(10))

    if "Churn" not in df.columns:
        st.error("Your CSV must include a 'Churn' column (Yes/No or 1/0).")
    else:
        # Basic conversion: Yes/No -> 1/0
        df = df.copy()
        if df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes':1,'No':0}).fillna(df['Churn'])
        df['Churn'] = df['Churn'].astype(int)

        # Keep only numeric features for this simple version
        X = df.drop(columns=["Churn"])
        X = X.select_dtypes(include=['number']).fillna(0)
        y = df["Churn"]

        st.write("Using numeric columns only (fillna->0). Columns used:", list(X.columns))

        # UI controls
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        model_choice = st.selectbox("Choose model", ["RandomForest","DecisionTree","KNN","SVM"])
        train_button = st.button("Train model")

        if train_button:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

            # For KNN and SVM we scale features
            if model_choice in ["KNN","SVM"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                # sklearn estimators accept numpy arrays / DataFrames directly
                X_train = X_train.values
                X_test = X_test.values

            if model_choice == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_choice == "DecisionTree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_choice == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
            else:
                model = SVC(kernel='rbf', probability=True, random_state=42)

            with st.spinner("Training..."):
                model.fit(X_train, y_train)

            # Evaluate
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.subheader("Results")
            st.write(f"Accuracy: {acc:.3f}")
            st.text(classification_report(y_test, preds, zero_division=0))
            cm = confusion_matrix(y_test, preds)
            st.write("Confusion matrix:")
            st.write(cm)

            # Save trained model and columns to session_state for prediction
            st.session_state['model'] = model
            st.session_state['train_columns'] = list(X.columns)
            if model_choice in ["KNN","SVM"]:
                st.session_state['scaler'] = scaler
            else:
                st.session_state['scaler'] = None
            st.success("Model trained and stored in session (you can now use the Predict section).")

# ----------------- Prediction / Batch scoring -----------------
st.markdown("---")
st.header("Predict (batch) — upload CSV without 'Churn'")

pred_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred")
if pred_file is not None:
    if 'model' not in st.session_state:
        st.warning("Train a model first (above) or refresh with a trained model in session.")
    else:
        df_new = pd.read_csv(pred_file)
        st.subheader("New data preview")
        st.dataframe(df_new.head(5))

        # Prepare numeric-only features like training
        X_new = df_new.copy()
        # Drop churn if present
        if 'Churn' in X_new.columns:
            X_new = X_new.drop(columns=['Churn'])
        X_new = X_new.select_dtypes(include=['number']).fillna(0)

        # Align columns: add missing cols with 0
        train_cols = st.session_state['train_columns']
        for c in train_cols:
            if c not in X_new.columns:
                X_new[c] = 0
        # Keep only training column order
        X_new = X_new[train_cols]

        scaler = st.session_state.get('scaler', None)
        model = st.session_state['model']
        if scaler is not None:
            X_in = scaler.transform(X_new)
        else:
            X_in = X_new.values

        preds = model.predict(X_in)
        out = df_new.copy()
        out['Churn_Pred'] = preds
        st.subheader("Predictions (first 10 rows)")
        st.dataframe(out.head(10))

        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime='text/csv')
