import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Adult Income Classifier", layout="wide")

st.title("Adult Income Classification App")
st.write("Compare multiple Machine Learning models on the Adult Census Dataset")

# ======================================================
# LOAD TRAINED MODELS
# ======================================================
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("models/Logistic_Regression.pkl"),
        "Decision Tree": joblib.load("models/Decision_Tree.pkl"),
        "KNN": joblib.load("models/KNN.pkl"),
        "Naive Bayes": joblib.load("models/Naive_Bayes.pkl"),
        "Random Forest": joblib.load("models/Random_Forest.pkl"),
        "XGBoost": joblib.load("models/XGBoost.pkl"),
    }

models = load_models()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# ======================================================
# METRIC FUNCTION
# ======================================================
def evaluate(y_true, y_pred, y_prob):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# ======================================================
# MAIN LOGIC
# ======================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    if "income" not in df.columns:
        st.error("CSV must contain 'income' column")
        st.stop()

    X = df.drop("income", axis=1)
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    model = models[model_name]

    # predictions
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred

    # ==================================================
    # METRICS DISPLAY
    # ==================================================
    st.subheader("Evaluation Metrics")

    metrics = evaluate(y, y_pred, y_prob)

    col1, col2, col3 = st.columns(3)

    for i, (k, v) in enumerate(metrics.items()):
        if i % 3 == 0:
            col1.metric(k, f"{v:.4f}")
        elif i % 3 == 1:
            col2.metric(k, f"{v:.4f}")
        else:
            col3.metric(k, f"{v:.4f}")

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ==================================================
    # CLASSIFICATION REPORT
    # ==================================================
    st.subheader("Classification Report")

    report = classification_report(y, y_pred)
    st.text(report)

else:
    st.info("Upload a CSV test file to begin.")
