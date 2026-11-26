import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.set_page_config(page_title="Smart Phishing Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e8eef8; }
[data-testid="stSidebar"] { background: #111b2d; }
.metric-card { background: #15243a; padding: 18px; border-radius: 12px; border: 1px solid #263c59; }
h1, h2, h3 { color: #eaf2ff; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Smart Phishing Website Detection System")
st.caption("Machine-learning prototype • Classifies supplied website features; it does not scan live URLs.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Train & Compare Models", "Predict from Features", "About"])
    st.divider()
    st.caption("Upload a labelled CSV dataset on the training page to calculate real results.")

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "feature_names" not in st.session_state:
    st.session_state.feature_names = []
if "results" not in st.session_state:
    st.session_state.results = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None

if page == "Overview":
    st.subheader("Project dashboard")
    a, b, c = st.columns(3)
    a.metric("Classification", "Phishing / Legitimate")
    b.metric("Candidate models", "4")
    c.metric("Evaluation", "80 / 20 split")
    st.markdown("### Workflow")
    st.write("1. Upload a labelled dataset.")
    st.write("2. Select the target/class column.")
    st.write("3. Train and compare four classifiers.")
    st.write("4. Review evaluation metrics and confusion matrices.")
    st.write("5. Enter feature values to make a prediction with a trained model.")
    st.info("No performance results are displayed until models are trained on an uploaded dataset.")

elif page == "Train & Compare Models":
    st.subheader("Dataset upload and model evaluation")
    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.dataset = df
            st.write("Dataset preview", df.head())
            st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")
            target = st.selectbox("Choose the target/class column", options=list(df.columns))
            if st.button("Train and evaluate models", type="primary"):
                X = df.drop(columns=[target])
                y = df[target]
                # Convert numeric-looking columns and reject non-numeric data for this feature-based prototype.
                X = X.apply(pd.to_numeric, errors="coerce")
                X = X.replace([np.inf, -np.inf], np.nan)
                keep_cols = [col for col in X.columns if not X[col].isna().all()]
                X = X[keep_cols]
                X = X.fillna(X.median(numeric_only=True))
                if X.empty or y.nunique() < 2:
                    st.error("Dataset must contain numeric input features and at least two target classes.")
                else:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                    except ValueError:
                        st.error("Could not perform stratified split. Ensure each class has enough samples.")
                        st.stop()
                    models = {
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=2000),
                        "Support Vector Machine": SVC()
                    }
                    results, fitted = [], {}
                    progress = st.progress(0)
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        results.append({
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, pred),
                            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0)
                        })
                        fitted[name] = model
                        progress.progress((i+1)/len(models))
                    st.session_state.trained_models = fitted
                    st.session_state.feature_names = list(X.columns)
                    st.session_state.results = pd.DataFrame(results)
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.success("Training and evaluation completed using the uploaded data.")
            if st.session_state.results is not None:
                results = st.session_state.results
                st.markdown("### Model comparison")
                st.dataframe(results.style.format({c: "{:.4f}" for c in results.columns if c != "Model"}), use_container_width=True)
                st.bar_chart(results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]])
                st.markdown("### Confusion matrices")
                for name, model in st.session_state.trained_models.items():
                    pred = model.predict(st.session_state.X_test)
                    cm = confusion_matrix(st.session_state.y_test, pred)
                    st.write(f"**{name}**")
                    fig, ax = plt.subplots()
                    ax.imshow(cm)
                    ax.set_title(f"{name} confusion matrix")
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("True label")
                    for (i, j), val in np.ndenumerate(cm):
                        ax.text(j, i, str(val), ha="center", va="center")
                    st.pyplot(fig)
                    plt.close(fig)
        except Exception as e:
            st.error(f"Could not read/process the CSV: {e}")
    else:
        st.warning("Upload a CSV file to begin. The system does not generate or invent scores.")

elif page == "Predict from Features":
    st.subheader("Prediction")
    models = st.session_state.trained_models
    if not models:
        st.warning("Train models first using a labelled CSV dataset on the Train & Compare Models page.")
    else:
        selected = st.selectbox("Select trained model", list(models.keys()))
        st.write("Enter values for each feature. Values must follow the feature encoding used in your dataset.")
        vals = {}
        cols = st.columns(3)
        for i, feature in enumerate(st.session_state.feature_names):
            vals[feature] = cols[i % 3].number_input(feature, value=0.0, format="%.4f", key=f"feature_{i}")
        if st.button("Classify website features", type="primary"):
            row = pd.DataFrame([vals], columns=st.session_state.feature_names)
            prediction = models[selected].predict(row)[0]
            st.success(f"Prediction: **{prediction}**")
            st.caption("This prediction is based on supplied feature values, not a live website scan.")

else:
    st.subheader("About this project")
    st.write("""
    This prototype demonstrates machine-learning classification of website-related feature records.
    It compares Decision Tree, Random Forest, Logistic Regression, and Support Vector Machine
    classifiers using an 80/20 stratified train-test split when possible.
    """)
    st.markdown("**Metrics:** accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrix.")
    st.warning("Limitation: the application predicts from supplied features. It does not extract features from or scan a live website URL.")
