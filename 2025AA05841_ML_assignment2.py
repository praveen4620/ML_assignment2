

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# =================================================
# LOAD + PREPROCESS DATA
# =================================================
df = pd.read_csv("adult.csv")

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

y = df["income"].map({"<=50K": 0, ">50K": 1})
X = df.drop("income", axis=1)

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate_and_save(model, model_name):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    # ===== metrics =====
    scores = [
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]

    # ===== confusion matrix =====
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")

    # Create the directory if it doesn't exist
    os.makedirs("confusion_matrices", exist_ok=True)
    plt.savefig(f"confusion_matrices/{model_name}_cm.png")
    plt.close()

    return scores

# =================================================
# DEFINE 6 MODELS
# =================================================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# =================================================
# TRAIN + EVALUATE ALL
# =================================================
import joblib
import os

results = []

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

for name, clf in models.items():

    print(f"Training {name}...")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", clf)
    ])

    pipe.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipe, f"models/{name}.pkl")

    scores = evaluate_and_save(pipe, name)

    results.append([name] + scores)

# =================================================
# SAVE RESULTS TABLE
# =================================================
columns = [
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

results_df = pd.DataFrame(results, columns=columns)

print("\nMODEL COMPARISON TABLE")
print(results_df)

results_df.to_csv("model_results.csv", index=False)

from IPython.display import Image
Image('confusion_matrices/Logistic_Regression_cm.png')

from IPython.display import Image
Image('confusion_matrices/Decision_Tree_cm.png')

from IPython.display import Image
Image('confusion_matrices/KNN_cm.png')

from IPython.display import Image
Image('confusion_matrices/Naive_Bayes_cm.png')

from IPython.display import Image
Image('confusion_matrices/Random_Forest_cm.png')

from IPython.display import Image
Image('confusion_matrices/XGBoost_cm.png')