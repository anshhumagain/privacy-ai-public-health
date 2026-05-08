import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import diffprivlib as dp
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [1, 21, 42, 100, 123]
EPSILONS = [10.0, 5.0, 1.0, 0.5, 0.1]


df = pd.read_csv(DATA_DIR / 'nhanes_merged.csv')

features = [
    'RIDAGEYR', 'RIAGENDR', 'INDFMPIR',
    'LBXTC', 'LBXTR', 'LBDLDL', 'LBDHDD',
]

features = [f for f in features if f in df.columns]
target = 'DIQ010'

df = df[features + [target]].copy()
df = df[df[target].isin([1, 2])]
df[target] = df[target].map({1: 1, 2: 0})
df = df.dropna()

print("=" * 60)
print("NHANES — CLASS DISTRIBUTION")
print("=" * 60)
class_counts = df[target].value_counts()
class_pct = df[target].value_counts(normalize=True) * 100
class_table = pd.DataFrame({'Count': class_counts, 'Percentage': class_pct.round(2)})
class_table.index = ['No Diabetes (0)', 'Has Diabetes (1)']
print(class_table.to_string())
print(f"\nTotal records: {len(df)}")
print(f"Features: {features}\n")

X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


def evaluate(model, X_train, X_test, y_train, y_test, name, epsilon_label=None):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_train, 4)

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = round(time.time() - start_pred, 4)

    return {
        'Model': name,
        'Privacy Setting': epsilon_label if epsilon_label else 'Baseline (No Privacy)',
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'F1 Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'Train Time (s)': train_time,
        'Predict Time (s)': pred_time,
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
    }


all_results = []

for seed in SEEDS:
    print(f"\n{'=' * 60}  SEED {seed}  {'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed, stratify=y
    )

    all_results.append({'Seed': seed, **evaluate(
        LogisticRegression(max_iter=1000),
        X_train, X_test, y_train, y_test,
        name='Logistic Regression'
    )})

    all_results.append({'Seed': seed, **evaluate(
        RandomForestClassifier(n_estimators=100, random_state=seed),
        X_train, X_test, y_train, y_test,
        name='Random Forest'
    )})

    all_results.append({'Seed': seed, **evaluate(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_train, X_test, y_train, y_test,
        name='Logistic Regression Balanced'
    )})

    all_results.append({'Seed': seed, **evaluate(
        RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced'),
        X_train, X_test, y_train, y_test,
        name='Random Forest Balanced'
    )})

    for eps in EPSILONS:
        label = f'DP e={eps}'

        all_results.append({'Seed': seed, **evaluate(
            dp.models.LogisticRegression(
                epsilon=eps,
                max_iter=1000
            ),
            X_train, X_test, y_train, y_test,
            name='DP Logistic Regression',
            epsilon_label=label
        )})

        all_results.append({'Seed': seed, **evaluate(
            dp.models.RandomForestClassifier(
                epsilon=eps,
                n_estimators=10
            ),
            X_train, X_test, y_train, y_test,
            name='DP Random Forest',
            epsilon_label=label
        )})

results_df = pd.DataFrame(all_results)
agg_df = results_df.drop(columns=['Confusion Matrix', 'Seed'])

metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Train Time (s)', 'Predict Time (s)']

summary = agg_df.groupby(['Model', 'Privacy Setting'])[metrics].agg(['mean', 'std']).round(4)
summary.columns = [f'{m} {s}' for m, s in summary.columns]
summary = summary.reset_index()

print("\n\n" + "=" * 80)
print("NHANES — RESULTS (mean ± std across 5 seeds)")
print("=" * 80)

display_rows = []
for _, row in summary.iterrows():
    display_rows.append({
        'Model': row['Model'],
        'Privacy Setting': row['Privacy Setting'],
        'Accuracy': f"{row['Accuracy mean']:.4f} ± {row['Accuracy std']:.4f}",
        'F1 Score': f"{row['F1 Score mean']:.4f} ± {row['F1 Score std']:.4f}",
        'Precision': f"{row['Precision mean']:.4f} ± {row['Precision std']:.4f}",
        'Recall': f"{row['Recall mean']:.4f} ± {row['Recall std']:.4f}",
        'Train Time (s)': f"{row['Train Time (s) mean']:.4f} ± {row['Train Time (s) std']:.4f}",
    })

display_df = pd.DataFrame(display_rows)
print(display_df.to_string(index=False))

print("\n\n" + "=" * 80)
print("CONFUSION MATRICES (seed=42)")
print("=" * 80)

for _, row in results_df[results_df['Seed'] == 42].iterrows():
    cm = np.array(row['Confusion Matrix'])
    print(f"\n{row['Model']} | {row['Privacy Setting']}")
    print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")

results_df.drop(columns=['Confusion Matrix']).to_csv(RESULTS_DIR / 'nhanes_results_raw.csv', index=False)
summary.to_csv(RESULTS_DIR / 'nhanes_results_summary.csv', index=False)
display_df.to_csv(RESULTS_DIR / 'nhanes_results.csv', index=False)

print("\nSaved results/nhanes_results.csv, results/nhanes_results_summary.csv, results/nhanes_results_raw.csv")
