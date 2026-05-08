from pathlib import Path
import time
import warnings

import diffprivlib.models as dp_models
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEEDS = [42, 43, 44, 45, 46]
EPSILONS = [10.0, 5.0, 1.0, 0.5, 0.1]

DATASET_PATH = DATASETS_DIR / "nhanes_merged.csv"

RAW_RESULTS_PATH = RESULTS_DIR / "dp_nhanes_results_raw.csv"
SUMMARY_RESULTS_PATH = RESULTS_DIR / "dp_nhanes_results_summary.csv"
MAIN_RESULTS_PATH = RESULTS_DIR / "dp_nhanes_results.csv"

df = pd.read_csv(DATASET_PATH)

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

features = [feature for feature in features if feature in df.columns]
target = "DIQ010"

df = df[features + [target]].copy()
df = df[df[target].isin([1, 2])]
df[target] = df[target].map({1: 1, 2: 0})
df = df.dropna()

print("NHANES cleaned dataset shape:", df.shape)
print("NHANES target distribution:")
print(df[target].value_counts())

X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, privacy_setting, seed):
    start_time = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    train_time = time.time() - start_time

    return {
        "Dataset": "NHANES",
        "Seed": seed,
        "Model": model_name,
        "Privacy Setting": privacy_setting,
        "Accuracy": accuracy_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "Train Time (s)": train_time,
    }

all_results = []

for seed in RANDOM_SEEDS:
    print(f"\nRunning NHANES experiments with random seed {seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    baseline_lr = LogisticRegression(max_iter=1000, random_state=seed)
    all_results.append(
        evaluate_model(
            baseline_lr,
            X_train,
            X_test,
            y_train,
            y_test,
            "Logistic Regression",
            "Baseline (No Privacy)",
            seed,
        )
    )

    baseline_rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    all_results.append(
        evaluate_model(
            baseline_rf,
            X_train,
            X_test,
            y_train,
            y_test,
            "Random Forest",
            "Baseline (No Privacy)",
            seed,
        )
    )

    for epsilon in EPSILONS:
        dp_lr = dp_models.LogisticRegression(
            epsilon=epsilon,
            data_norm=1.0,
            max_iter=1000,
            random_state=seed,
        )

        all_results.append(
            evaluate_model(
                dp_lr,
                X_train,
                X_test,
                y_train,
                y_test,
                "DP Logistic Regression",
                f"DP e={epsilon}",
                seed,
            )
        )

    for epsilon in EPSILONS:
        dp_rf = dp_models.RandomForestClassifier(
            epsilon=epsilon,
            n_estimators=100,
            random_state=seed,
        )

        all_results.append(
            evaluate_model(
                dp_rf,
                X_train,
                X_test,
                y_train,
                y_test,
                "DP Random Forest",
                f"DP e={epsilon}",
                seed,
            )
        )

raw_results = pd.DataFrame(all_results)
raw_results.to_csv(RAW_RESULTS_PATH, index=False)

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

print("\nSaved NHANES DP files:")
print(f"- {RAW_RESULTS_PATH}")
print(f"- {SUMMARY_RESULTS_PATH}")
print(f"- {MAIN_RESULTS_PATH}")
print("\nNHANES DP summary:")
print(main_results)