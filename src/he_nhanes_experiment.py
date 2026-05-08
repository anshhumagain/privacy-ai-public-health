import time
import numpy as np
import pandas as pd
import tenseal as ts

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "datasets" / "nhanes_merged.csv", low_memory=False)

features = [
    "RIDAGEYR",
    "RIAGENDR",
    "INDFMPIR",
    "LBXGH",
    "LBXSGL",
    "LBXTC",
    "LBXTR",
    "LBDLDL",
    "LBDHDD",
]

features = [f for f in features if f in df.columns]
target = "DIQ010"

df = df[features + [target]].copy()
df = df[df[target].isin([1, 2])]
df[target] = df[target].map({1: 1, 2: 0})
df = df.dropna()

print("Cleaned dataset shape:", df.shape)
print("Target distribution:")
print(df[target].value_counts())

X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

plain_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

start = time.time()
plain_model.fit(X_train, y_train)
plain_pred = plain_model.predict(X_test)
plain_runtime = time.time() - start

plain_metrics = {
    "Method": "Normal Logistic Regression",
    "Accuracy": accuracy_score(y_test, plain_pred),
    "F1 Score": f1_score(y_test, plain_pred, zero_division=0),
    "Precision": precision_score(y_test, plain_pred, zero_division=0),
    "Recall": recall_score(y_test, plain_pred, zero_division=0),
    "Runtime Seconds": plain_runtime
}

print("\nNormal Logistic Regression Results:")
print(plain_metrics)
print("Confusion Matrix:")
print(confusion_matrix(y_test, plain_pred))

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.global_scale = 2 ** 40
context.generate_galois_keys()

weights = plain_model.coef_[0]
bias = plain_model.intercept_[0]

def sigmoid_approx(x):
    """
    Approximation used after decrypting the encrypted linear score.
    Logistic sigmoid itself is not directly evaluated under this simple HE setup.
    """
    return 1 / (1 + np.exp(-x))

he_preds = []
start = time.time()

for row in X_test.to_numpy():
    encrypted_row = ts.ckks_vector(context, row.tolist())

    encrypted_score = encrypted_row.dot(weights) + bias
    decrypted_score = encrypted_score.decrypt()[0]

    prob = sigmoid_approx(decrypted_score)
    pred = 1 if prob >= 0.5 else 0
    he_preds.append(pred)

he_runtime = time.time() - start

he_metrics = {
    "Method": "HE Logistic Regression Inference",
    "Accuracy": accuracy_score(y_test, he_preds),
    "F1 Score": f1_score(y_test, he_preds, zero_division=0),
    "Precision": precision_score(y_test, he_preds, zero_division=0),
    "Recall": recall_score(y_test, he_preds, zero_division=0),
    "Runtime Seconds": he_runtime
}

print("\nHE Logistic Regression Inference Results:")
print(he_metrics)
print("Confusion Matrix:")
print(confusion_matrix(y_test, he_preds))

plain_cm = confusion_matrix(y_test, plain_pred)
he_cm = confusion_matrix(y_test, he_preds)

cm_results = pd.DataFrame({
    "Model": [
        "Normal Logistic Regression",
        "Normal Logistic Regression",
        "HE Logistic Regression Inference",
        "HE Logistic Regression Inference"
    ],
    "Actual Class": [0, 1, 0, 1],
    "Predicted 0": [
        plain_cm[0][0],
        plain_cm[1][0],
        he_cm[0][0],
        he_cm[1][0]
    ],
    "Predicted 1": [
        plain_cm[0][1],
        plain_cm[1][1],
        he_cm[0][1],
        he_cm[1][1]
    ]
})

cm_results.to_csv(BASE_DIR / "results" / "he_nhanes_confusion_matrices.csv", index=False)

print("\nSaved confusion matrices to results/he_nhanes_confusion_matrices.csv")
print(cm_results)

results = pd.DataFrame([plain_metrics, he_metrics])
results.to_csv(BASE_DIR / "results" / "he_nhanes_results.csv", index=False)

print("\nSaved results to he_nhanes_results.csv")
print(results)