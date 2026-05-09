import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

SEEDS = [1, 21, 42, 100, 123]
K_VALUES = [2, 5, 10, 20, 50, 100]

print("Loading NHANES dataset...")

DATA_PATH = Path(__file__).resolve().parent / "nhanes_merged.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Full dataset shape: {df.shape}")

features = [
    'RIDAGEYR', 'RIAGENDR', 'INDFMPIR',
    'LBXGH', 'LBXSGL', 'LBXTC', 'LBXTR', 'LBDLDL', 'LBDHDD',
]

features = [f for f in features if f in df.columns]
target = 'DIQ010'

if target not in df.columns:
    raise ValueError(f"Target column {target} not found in dataset.")

df = df[features + [target]].copy()

# Convert target to numeric
df[target] = pd.to_numeric(df[target], errors='coerce')

# NHANES DIQ010:
# 1 = Doctor told participant they have diabetes
# 2 = No diabetes
df = df[df[target].isin([1, 2])]
df[target] = df[target].map({1: 1, 2: 0})

# Convert feature columns to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# ------------------------------------------------------------
# K-ANONYMITY GENERALISATION
# ------------------------------------------------------------
# For NHANES, raw continuous values create too many unique records.
# So we generalise age and income into broader bands.

if 'RIDAGEYR' in df.columns:
    df['AGE_BAND'] = pd.cut(
        df['RIDAGEYR'],
        bins=[0, 17, 29, 39, 49, 59, 69, 79, 120],
        labels=False,
        include_lowest=True
    )

if 'INDFMPIR' in df.columns:
    df['INCOME_BAND'] = pd.cut(
        df['INDFMPIR'],
        bins=[-0.01, 1, 2, 3, 4, 5.1],
        labels=False,
        include_lowest=True
    )

# Quasi-identifiers are the variables that may identify someone when combined
quasi_identifiers = []

if 'AGE_BAND' in df.columns:
    quasi_identifiers.append('AGE_BAND')

if 'RIAGENDR' in df.columns:
    quasi_identifiers.append('RIAGENDR')

if 'INCOME_BAND' in df.columns:
    quasi_identifiers.append('INCOME_BAND')

# Replace raw age and income with generalised versions for the model
model_features = []

for f in features:
    if f == 'RIDAGEYR' and 'AGE_BAND' in df.columns:
        model_features.append('AGE_BAND')
    elif f == 'INDFMPIR' and 'INCOME_BAND' in df.columns:
        model_features.append('INCOME_BAND')
    else:
        model_features.append(f)

# Remove duplicate feature names
model_features = list(dict.fromkeys(model_features))

df = df.dropna(subset=model_features + quasi_identifiers + [target])

print("\n" + "=" * 60)
print("NHANES — CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df[target].value_counts()
class_pct = df[target].value_counts(normalize=True) * 100

class_table = pd.DataFrame({
    'Count': class_counts,
    'Percentage': class_pct.round(2)
})

class_table.index = class_table.index.map({
    0: 'No Diabetes (0)',
    1: 'Has Diabetes (1)'
})

print(class_table.to_string())
print(f"\nTotal records: {len(df)}")
print(f"Model features: {model_features}")
print(f"Quasi-identifiers for K-anonymity: {quasi_identifiers}\n")


def apply_k_anonymity(train_df, quasi_identifiers, k):
    """
    Applies K-anonymity by suppression.

    Any equivalence class with fewer than k records is removed.
    An equivalence class is a group of records with the same quasi-identifier values.
    """

    df_k = train_df.copy()

    if len(quasi_identifiers) == 0:
        raise ValueError("No quasi-identifiers found for K-anonymity.")

    equivalence_class_sizes = df_k.groupby(quasi_identifiers)[target].transform('size')

    before_rows = len(df_k)
    before_classes = df_k.groupby(quasi_identifiers).ngroups

    df_k = df_k[equivalence_class_sizes >= k].copy()

    after_rows = len(df_k)
    after_classes = df_k.groupby(quasi_identifiers).ngroups if after_rows > 0 else 0

    rows_suppressed = before_rows - after_rows
    suppression_rate = (rows_suppressed / before_rows) * 100 if before_rows > 0 else 0

    if after_rows > 0:
        min_equivalence_class = df_k.groupby(quasi_identifiers).size().min()
    else:
        min_equivalence_class = 0

    stats = {
        'Train Rows Used': after_rows,
        'Rows Suppressed': rows_suppressed,
        'Suppression Rate (%)': round(suppression_rate, 2),
        'Equivalence Classes Before': before_classes,
        'Equivalence Classes After': after_classes,
        'Min Equivalence Class Size': int(min_equivalence_class)
    }

    return df_k, stats


def scale_train_test(train_df, test_df):
    """
    Scales features using only the training set.
    """

    scaler = MinMaxScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[model_features]),
        columns=model_features
    )

    X_test = pd.DataFrame(
        scaler.transform(test_df[model_features]),
        columns=model_features
    )

    y_train = train_df[target].astype(int)
    y_test = test_df[target].astype(int)

    return X_train, X_test, y_train, y_test


def evaluate(model, X_train, X_test, y_train, y_test, name, privacy_label, privacy_stats=None):
    """
    Trains and evaluates the model.
    """

    if privacy_stats is None:
        privacy_stats = {
            'Train Rows Used': len(X_train),
            'Rows Suppressed': 0,
            'Suppression Rate (%)': 0,
            'Equivalence Classes Before': np.nan,
            'Equivalence Classes After': np.nan,
            'Min Equivalence Class Size': np.nan
        }

    result = {
        'Model': name,
        'Privacy Setting': privacy_label,
        **privacy_stats
    }

    if len(y_train) == 0:
        result.update({
            'Accuracy': np.nan,
            'F1 Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'Train Time (s)': np.nan,
            'Predict Time (s)': np.nan,
            'Confusion Matrix': None,
            'Skipped Reason': 'No training rows left after K-anonymity'
        })
        return result

    if len(np.unique(y_train)) < 2:
        result.update({
            'Accuracy': np.nan,
            'F1 Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'Train Time (s)': np.nan,
            'Predict Time (s)': np.nan,
            'Confusion Matrix': None,
            'Skipped Reason': 'Only one class left after K-anonymity'
        })
        return result

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_train, 4)

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = round(time.time() - start_pred, 4)

    result.update({
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'F1 Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'Train Time (s)': train_time,
        'Predict Time (s)': pred_time,
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
        'Skipped Reason': ''
    })

    return result


all_results = []

for seed in SEEDS:
    print(f"\n{'=' * 60}  SEED {seed}  {'=' * 60}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df[target]
    )

    X_train, X_test, y_train, y_test = scale_train_test(train_df, test_df)

    baseline_stats = {
        'Train Rows Used': len(train_df),
        'Rows Suppressed': 0,
        'Suppression Rate (%)': 0,
        'Equivalence Classes Before': train_df.groupby(quasi_identifiers).ngroups,
        'Equivalence Classes After': train_df.groupby(quasi_identifiers).ngroups,
        'Min Equivalence Class Size': int(train_df.groupby(quasi_identifiers).size().min())
    }

    all_results.append({'Seed': seed, **evaluate(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_train, X_test, y_train, y_test,
        name='Logistic Regression',
        privacy_label='Baseline (No K-anonymity)',
        privacy_stats=baseline_stats
    )})

    all_results.append({'Seed': seed, **evaluate(
        RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight='balanced'
        ),
        X_train, X_test, y_train, y_test,
        name='Random Forest',
        privacy_label='Baseline (No K-anonymity)',
        privacy_stats=baseline_stats
    )})

    for k in K_VALUES:
        label = f'K-anonymity k={k}'

        train_k_df, k_stats = apply_k_anonymity(
            train_df=train_df,
            quasi_identifiers=quasi_identifiers,
            k=k
        )

        print(
            f"{label}: retained {k_stats['Train Rows Used']} rows, "
            f"suppressed {k_stats['Rows Suppressed']} rows "
            f"({k_stats['Suppression Rate (%)']}%)"
        )

        if len(train_k_df) > 0:
            X_train_k, X_test_k, y_train_k, y_test_k = scale_train_test(train_k_df, test_df)
        else:
            X_train_k = pd.DataFrame(columns=model_features)
            X_test_k = test_df[model_features]
            y_train_k = pd.Series(dtype=int)
            y_test_k = test_df[target].astype(int)

        all_results.append({'Seed': seed, **evaluate(
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            X_train_k, X_test_k, y_train_k, y_test_k,
            name='K-anonymised Logistic Regression',
            privacy_label=label,
            privacy_stats=k_stats
        )})

        all_results.append({'Seed': seed, **evaluate(
            RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                class_weight='balanced'
            ),
            X_train_k, X_test_k, y_train_k, y_test_k,
            name='K-anonymised Random Forest',
            privacy_label=label,
            privacy_stats=k_stats
        )})


results_df = pd.DataFrame(all_results)

summary_metrics = [
    'Accuracy',
    'F1 Score',
    'Precision',
    'Recall',
    'Train Time (s)',
    'Predict Time (s)',
    'Train Rows Used',
    'Rows Suppressed',
    'Suppression Rate (%)',
    'Equivalence Classes After',
    'Min Equivalence Class Size'
]

agg_df = results_df.drop(columns=['Confusion Matrix', 'Seed'])

summary = agg_df.groupby(['Model', 'Privacy Setting'])[summary_metrics].agg(['mean', 'std']).round(4)
summary.columns = [f'{metric} {stat}' for metric, stat in summary.columns]
summary = summary.reset_index()


def mean_std(row, metric):
    mean = row[f'{metric} mean']
    std = row[f'{metric} std']

    if pd.isna(mean):
        return 'NA'

    if pd.isna(std):
        std = 0

    return f"{mean:.4f} ± {std:.4f}"


print("\n\n" + "=" * 80)
print("NHANES — K-ANONYMITY RESULTS (mean ± std across 5 seeds)")
print("=" * 80)

display_rows = []

for _, row in summary.iterrows():
    display_rows.append({
        'Model': row['Model'],
        'Privacy Setting': row['Privacy Setting'],
        'Accuracy': mean_std(row, 'Accuracy'),
        'F1 Score': mean_std(row, 'F1 Score'),
        'Precision': mean_std(row, 'Precision'),
        'Recall': mean_std(row, 'Recall'),
        'Train Rows Used': mean_std(row, 'Train Rows Used'),
        'Rows Suppressed': mean_std(row, 'Rows Suppressed'),
        'Suppression Rate (%)': mean_std(row, 'Suppression Rate (%)'),
        'Train Time (s)': mean_std(row, 'Train Time (s)')
    })

display_df = pd.DataFrame(display_rows)
print(display_df.to_string(index=False))


print("\n\n" + "=" * 80)
print("CONFUSION MATRICES (seed=42)")
print("=" * 80)

seed_42_results = results_df[results_df['Seed'] == 42]

for _, row in seed_42_results.iterrows():
    print(f"\n{row['Model']} | {row['Privacy Setting']}")

    if row['Confusion Matrix'] is None:
        print(f"  Skipped: {row['Skipped Reason']}")
        continue

    cm = np.array(row['Confusion Matrix'])

    print(f"  TN: {cm[0, 0]}  FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}  TP: {cm[1, 1]}")


results_df.drop(columns=['Confusion Matrix']).to_csv('nhanes_k_anonymity_results_raw.csv', index=False)
summary.to_csv('nhanes_k_anonymity_results_summary.csv', index=False)
display_df.to_csv('nhanes_k_anonymity_results.csv', index=False)

print("\nSaved nhanes_k_anonymity_results.csv, nhanes_k_anonymity_results_summary.csv, nhanes_k_anonymity_results_raw.csv")