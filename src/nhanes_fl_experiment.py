from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEEDS = [42, 43, 44, 45, 46]
FL_SETTINGS = [
    (3, 10),
    (5, 20),
    (10, 20),
]

RAW_RESULTS_PATH = RESULTS_DIR / "fl_nhanes_results_raw.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "fl_nhanes_results_summary.csv"
MAIN_RESULTS_PATH = RESULTS_DIR / "fl_nhanes_results.csv"
CONFUSION_RESULTS_PATH = RESULTS_DIR / "fl_nhanes_confusion_matrices.csv"


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

    X_array = X_train.to_numpy()
    y_array = y_train.to_numpy()

    return [(X_array[idx], y_array[idx]) for idx in client_indices]


def run_baseline_lr(X_train, X_test, y_train, y_test, seed):
    model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")

    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    runtime = time.time() - start

    return {
        "Dataset": "NHANES",
        "Seed": seed,
        "Model": "Logistic Regression",
        "Privacy Setting": "Baseline (No Privacy)",
        **calculate_metrics(y_test, predictions),
        "Train Time (s)": runtime,
    }, predictions


def run_baseline_rf(X_train, X_test, y_train, y_test, seed):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight="balanced",
    )

    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    runtime = time.time() - start

    return {
        "Dataset": "NHANES",
        "Seed": seed,
        "Model": "Random Forest",
        "Privacy Setting": "Baseline (No Privacy)",
        **calculate_metrics(y_test, predictions),
        "Train Time (s)": runtime,
    }, predictions


def run_federated_lr(X_train, X_test, y_train, y_test, seed, n_clients, n_rounds):
    clients = split_clients(X_train, y_train, n_clients)

    client_models = []
    start = time.time()

    for round_idx in range(n_rounds):
        for client_idx, (X_client, y_client) in enumerate(clients):
            model = LogisticRegression(
                max_iter=1000,
                random_state=seed + round_idx + client_idx,
                class_weight="balanced",
            )
            model.fit(X_client, y_client)
            client_models.append(model)

    probabilities = np.mean(
        [model.predict_proba(X_test)[:, 1] for model in client_models],
        axis=0,
    )

    predictions = (probabilities >= 0.5).astype(int)
    runtime = time.time() - start

    return {
        "Dataset": "NHANES",
        "Seed": seed,
        "Model": "Federated Logistic Regression",
        "Privacy Setting": f"FL ({n_clients} clients, {n_rounds} rounds)",
        **calculate_metrics(y_test, predictions),
        "Train Time (s)": runtime,
    }, predictions


def run_federated_rf(X_train, X_test, y_train, y_test, seed, n_clients, n_rounds):
    clients = split_clients(X_train, y_train, n_clients)

    client_models = []
    start = time.time()

    for round_idx in range(n_rounds):
        for client_idx, (X_client, y_client) in enumerate(clients):
            model = RandomForestClassifier(
                n_estimators=20,
                random_state=seed + round_idx + client_idx,
                class_weight="balanced",
            )
            model.fit(X_client, y_client)
            client_models.append(model)

    probabilities = np.mean(
        [model.predict_proba(X_test)[:, 1] for model in client_models],
        axis=0,
    )

    predictions = (probabilities >= 0.5).astype(int)
    runtime = time.time() - start

    return {
        "Dataset": "NHANES",
        "Seed": seed,
        "Model": "Federated Random Forest",
        "Privacy Setting": f"FL ({n_clients} clients, {n_rounds} rounds)",
        **calculate_metrics(y_test, predictions),
        "Train Time (s)": runtime,
    }, predictions


df = pd.read_csv(DATASETS_DIR / "nhanes_merged.csv", low_memory=False)

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

X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("NHANES FL cleaned dataset shape:", X_scaled.shape)
print("NHANES FL target distribution:")
print(y.value_counts())

all_results = []
all_confusion_matrices = []

for seed in RANDOM_SEEDS:
    print(f"\n{'=' * 60} SEED {seed} {'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    experiment_runs = [
        run_baseline_lr(X_train, X_test, y_train, y_test, seed),
        run_baseline_rf(X_train, X_test, y_train, y_test, seed),
    ]

    for n_clients, n_rounds in FL_SETTINGS:
        experiment_runs.append(
            run_federated_lr(X_train, X_test, y_train, y_test, seed, n_clients, n_rounds)
        )
        experiment_runs.append(
            run_federated_rf(X_train, X_test, y_train, y_test, seed, n_clients, n_rounds)
        )

    for result, predictions in experiment_runs:
        all_results.append(result)

        cm = confusion_matrix(y_test, predictions)

        for actual_class in [0, 1]:
            all_confusion_matrices.append({
                "Dataset": "NHANES",
                "Seed": seed,
                "Model": result["Model"],
                "Privacy Setting": result["Privacy Setting"],
                "Actual Class": actual_class,
                "Predicted 0": cm[actual_class][0],
                "Predicted 1": cm[actual_class][1],
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

print("\nSaved NHANES FL files.")
print(main_results)