from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import tenseal as ts

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SIZE = 50000

DATASET_PATH = DATASETS_DIR / "covid.csv"

RAW_RESULTS_PATH = RESULTS_DIR / "he_covid_results_raw.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "he_covid_results_summary.csv"
MAIN_RESULTS_PATH = RESULTS_DIR / "he_covid_results.csv"
CONFUSION_RESULTS_PATH = RESULTS_DIR / "he_covid_confusion_matrices.csv"


def sigmoid_approx(x):
    return 1 / (1 + np.exp(-x))


def calculate_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
    }


df = pd.read_csv(DATASET_PATH, low_memory=False)

if "death_yn" in df.columns:
    target = "death_yn"
elif "hosp_yn" in df.columns:
    target = "hosp_yn"
else:
    raise ValueError("Could not find COVID target column. Expected death_yn or hosp_yn.")

df = df[df[target].isin(["Yes", "No"])].copy()
df[target] = df[target].map({"Yes": 1, "No": 0})

if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)

drop_columns = [
    target,
    "cdc_case_earliest_dt",
    "cdc_report_dt",
    "pos_spec_dt",
    "onset_dt",
]

drop_columns = [column for column in drop_columns if column in df.columns]

X = df.drop(columns=drop_columns)
y = df[target]

missing_ratio = X.isna().mean()
X = X.loc[:, missing_ratio < 0.5]

for column in X.columns:
    if pd.api.types.is_numeric_dtype(X[column]):
        X[column] = X[column].fillna(X[column].median())
    else:
        X[column] = X[column].fillna("Unknown")

for column in X.columns:
    if not pd.api.types.is_numeric_dtype(X[column]):
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("COVID HE cleaned dataset shape:", X_scaled.shape)
print("COVID HE target used:", target)
print("COVID HE target distribution:")
print(y.value_counts())

all_results = []
all_confusions = []

for seed in RANDOM_SEEDS:
    print(f"\n{'=' * 60}")
    print(f"Running COVID HE experiment with seed {seed}")
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    plain_model = LogisticRegression(max_iter=1000, random_state=seed)

    start = time.time()
    plain_model.fit(X_train, y_train)
    plain_pred = plain_model.predict(X_test)
    plain_runtime = time.time() - start

    plain_metrics = calculate_metrics(y_test, plain_pred)

    all_results.append({
        "Dataset": "COVID-19",
        "Seed": seed,
        "Method": "Normal Logistic Regression",
        **plain_metrics,
        "Runtime Seconds": plain_runtime,
    })

    plain_cm = confusion_matrix(y_test, plain_pred)

    all_confusions.append({
        "Dataset": "COVID-19",
        "Seed": seed,
        "Model": "Normal Logistic Regression",
        "Actual Class": 0,
        "Predicted 0": plain_cm[0][0],
        "Predicted 1": plain_cm[0][1],
    })

    all_confusions.append({
        "Dataset": "COVID-19",
        "Seed": seed,
        "Model": "Normal Logistic Regression",
        "Actual Class": 1,
        "Predicted 0": plain_cm[1][0],
        "Predicted 1": plain_cm[1][1],
    })

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )

    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    weights = plain_model.coef_[0]
    bias = plain_model.intercept_[0]

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

    he_metrics = calculate_metrics(y_test, he_preds)

    all_results.append({
        "Dataset": "COVID-19",
        "Seed": seed,
        "Method": "HE Logistic Regression Inference",
        **he_metrics,
        "Runtime Seconds": he_runtime,
    })

    he_cm = confusion_matrix(y_test, he_preds)

    all_confusions.append({
        "Dataset": "COVID-19",
        "Seed": seed,
        "Model": "HE Logistic Regression Inference",
        "Actual Class": 0,
        "Predicted 0": he_cm[0][0],
        "Predicted 1": he_cm[0][1],
    })

    all_confusions.append({
        "Dataset": "COVID-19",
        "Seed": seed,
        "Model": "HE Logistic Regression Inference",
        "Actual Class": 1,
        "Predicted 0": he_cm[1][0],
        "Predicted 1": he_cm[1][1],
    })

raw_results = pd.DataFrame(all_results)
raw_results.to_csv(RAW_RESULTS_PATH, index=False)

summary_results = (
    raw_results
    .groupby(["Dataset", "Method"])
    .agg({
        "Accuracy": ["mean", "std"],
        "F1 Score": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "Runtime Seconds": ["mean", "std"],
    })
    .reset_index()
)

summary_results.columns = [
    "Dataset",
    "Method",
    "Accuracy mean",
    "Accuracy std",
    "F1 Score mean",
    "F1 Score std",
    "Precision mean",
    "Precision std",
    "Recall mean",
    "Recall std",
    "Runtime Seconds mean",
    "Runtime Seconds std",
]

summary_results.to_csv(SUMMARY_RESULTS_PATH, index=False)

main_results = summary_results.copy()

for metric in ["Accuracy", "F1 Score", "Precision", "Recall", "Runtime Seconds"]:
    main_results[metric] = main_results.apply(
        lambda row: f"{row[f'{metric} mean']:.4f} ± {row[f'{metric} std']:.4f}",
        axis=1,
    )

main_results = main_results[
    [
        "Dataset",
        "Method",
        "Accuracy",
        "F1 Score",
        "Precision",
        "Recall",
        "Runtime Seconds",
    ]
]

main_results.to_csv(MAIN_RESULTS_PATH, index=False)

cm_results = pd.DataFrame(all_confusions)
cm_results.to_csv(CONFUSION_RESULTS_PATH, index=False)

print("\nSaved:")
print(f"- {RAW_RESULTS_PATH}")
print(f"- {SUMMARY_RESULTS_PATH}")
print(f"- {MAIN_RESULTS_PATH}")
print(f"- {CONFUSION_RESULTS_PATH}")

print("\nCOVID HE summary:")
print(main_results)