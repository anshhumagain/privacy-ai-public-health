from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEEDS = [42, 43, 44, 45, 46]
N_CLIENTS = 5
N_ROUNDS = 20
LOCAL_EPOCHS = 1
SAMPLE_SIZE = 50000

DATASET_PATH = DATASETS_DIR / "covid.csv"

RAW_RESULTS_PATH = RESULTS_DIR / "fl_covid_results_raw.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "fl_covid_results_summary.csv"
MAIN_RESULTS_PATH = RESULTS_DIR / "fl_covid_results.csv"
CONFUSION_RESULTS_PATH = RESULTS_DIR / "fl_covid_confusion_matrices.csv"

df = pd.read_csv(DATASET_PATH, low_memory=False)

if "death_yn" in df.columns:
    target = "death_yn"
elif "hosp_yn" in df.columns:
    target = "hosp_yn"
else:
    raise ValueError("Could not find a valid COVID target column. Expected 'death_yn' or 'hosp_yn'.")

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

print("COVID cleaned dataset shape:", X_scaled.shape)
print("COVID target used:", target)
print("COVID target distribution:")
print(y.value_counts())


def calculate_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
    }


def split_clients(X_train, y_train, n_clients):
    indices = np.arange(len(X_train))
    client_indices = np.array_split(indices, n_clients)

    X_array = np.asarray(X_train)
    y_array = np.asarray(y_train)

    clients = []

    for idx in client_indices:
        clients.append((X_array[idx], y_array[idx]))

    return clients


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def federated_predict(X_test, global_weights, global_bias):
    scores = np.asarray(X_test).dot(global_weights.reshape(-1)) + global_bias
    probabilities = sigmoid(scores)
    return (probabilities >= 0.5).astype(int)


def run_baseline(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    train_time = time.time() - start_time

    metrics = calculate_metrics(y_test, predictions)

    return {
        "Dataset": "COVID-19",
        "Seed": seed,
        "Model": "Logistic Regression",
        "Privacy Setting": "Baseline (No Privacy)",
        **metrics,
        "Train Time (s)": train_time,
    }, predictions


def run_federated_learning(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()

    clients = split_clients(X_train, y_train, N_CLIENTS)
    n_features = X_train.shape[1]

    global_weights = np.zeros((1, n_features))
    global_bias = np.zeros(1)
    classes = np.array([0, 1])

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(y_train)
    )

    class_weights = {
        0: class_weights_array[0],
        1: class_weights_array[1],
    }

    for _ in range(N_ROUNDS):
        local_weights = []
        local_biases = []
        sample_counts = []

        for X_client, y_client in clients:
            local_model = SGDClassifier(
                loss="log_loss",
                max_iter=LOCAL_EPOCHS,
                tol=None,
                learning_rate="constant",
                eta0=0.01,
                random_state=seed,
                warm_start=True,
                class_weight=class_weights,
            )

            local_model.partial_fit(X_client, y_client, classes=classes)
            local_model.coef_ = global_weights.copy()
            local_model.intercept_ = global_bias.copy()
            local_model.partial_fit(X_client, y_client, classes=classes)

            local_weights.append(local_model.coef_.copy())
            local_biases.append(local_model.intercept_.copy())
            sample_counts.append(len(y_client))

        total_samples = sum(sample_counts)

        global_weights = sum(
            weight * (count / total_samples)
            for weight, count in zip(local_weights, sample_counts)
        )

        global_bias = sum(
            bias * (count / total_samples)
            for bias, count in zip(local_biases, sample_counts)
        )

    predictions = federated_predict(X_test, global_weights, global_bias)
    train_time = time.time() - start_time

    metrics = calculate_metrics(y_test, predictions)

    return {
        "Dataset": "COVID-19",
        "Seed": seed,
        "Model": "Federated Logistic Regression",
        "Privacy Setting": f"FL ({N_CLIENTS} clients, {N_ROUNDS} rounds)",
        **metrics,
        "Train Time (s)": train_time,
    }, predictions


all_results = []
all_confusion_matrices = []

for seed in RANDOM_SEEDS:
    print(f"\nRunning COVID-19 FL experiments with random seed {seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    baseline_result, baseline_predictions = run_baseline(
        X_train, X_test, y_train, y_test, seed
    )

    fl_result, fl_predictions = run_federated_learning(
        X_train, X_test, y_train, y_test, seed
    )

    all_results.append(baseline_result)
    all_results.append(fl_result)

    for model_name, privacy_setting, predictions in [
        ("Logistic Regression", "Baseline (No Privacy)", baseline_predictions),
        ("Federated Logistic Regression", f"FL ({N_CLIENTS} clients, {N_ROUNDS} rounds)", fl_predictions),
    ]:
        cm = confusion_matrix(y_test, predictions)

        all_confusion_matrices.append({
            "Dataset": "COVID-19",
            "Seed": seed,
            "Model": model_name,
            "Privacy Setting": privacy_setting,
            "Actual Class": 0,
            "Predicted 0": cm[0][0],
            "Predicted 1": cm[0][1],
        })

        all_confusion_matrices.append({
            "Dataset": "COVID-19",
            "Seed": seed,
            "Model": model_name,
            "Privacy Setting": privacy_setting,
            "Actual Class": 1,
            "Predicted 0": cm[1][0],
            "Predicted 1": cm[1][1],
        })


raw_results = pd.DataFrame(all_results)
raw_results.to_csv(RAW_RESULTS_PATH, index=False)

confusion_results = pd.DataFrame(all_confusion_matrices)
confusion_results.to_csv(CONFUSION_RESULTS_PATH, index=False)

summary_results = (
    raw_results
    .groupby(["Dataset", "Model", "Privacy Setting"])
    .agg({
        "Accuracy": ["mean", "std"],
        "F1 Score": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "Train Time (s)": ["mean", "std"],
    })
    .reset_index()
)

summary_results.columns = [
    "Dataset",
    "Model",
    "Privacy Setting",
    "Accuracy mean",
    "Accuracy std",
    "F1 Score mean",
    "F1 Score std",
    "Precision mean",
    "Precision std",
    "Recall mean",
    "Recall std",
    "Train Time (s) mean",
    "Train Time (s) std",
]

summary_results.to_csv(SUMMARY_RESULTS_PATH, index=False)

main_results = summary_results.copy()
main_results["Accuracy"] = main_results["Accuracy mean"]
main_results["F1 Score"] = main_results["F1 Score mean"]
main_results["Precision"] = main_results["Precision mean"]
main_results["Recall"] = main_results["Recall mean"]
main_results["Train Time (s)"] = main_results["Train Time (s) mean"]

main_results = main_results[
    [
        "Dataset",
        "Model",
        "Privacy Setting",
        "Accuracy",
        "F1 Score",
        "Precision",
        "Recall",
        "Train Time (s)",
    ]
]

main_results.to_csv(MAIN_RESULTS_PATH, index=False)

print("\nSaved COVID-19 FL files:")
print(f"- {RAW_RESULTS_PATH}")
print(f"- {SUMMARY_RESULTS_PATH}")
print(f"- {MAIN_RESULTS_PATH}")
print(f"- {CONFUSION_RESULTS_PATH}")

print("\nCOVID-19 FL summary:")
print(main_results)