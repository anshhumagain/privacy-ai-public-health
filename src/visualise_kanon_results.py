import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"

GRAPHS_DIR.mkdir(exist_ok=True)


def load_summary(filename):
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filename} inside the results folder."
        )

    return pd.read_csv(filepath)


def clean_dataset_name(dataset_name):
    return dataset_name.lower().replace("-", "_").replace(" ", "_")

def prepare_k_order(summary):
    summary = summary.copy()

    summary["K"] = summary["Privacy Setting"].str.extract(r"k=(\d+)")
    summary["K"] = pd.to_numeric(summary["K"], errors="coerce")

    k_rows = summary[summary["K"].notna()].copy()

    k_logistic = k_rows[
        k_rows["Model"].str.contains("Logistic Regression", case=False, na=False)
    ].sort_values("K")

    k_random_forest = k_rows[
        k_rows["Model"].str.contains("Random Forest", case=False, na=False)
    ].sort_values("K")

    baseline_logistic = summary[
        summary["Model"].eq("Logistic Regression")
        & summary["Privacy Setting"].str.contains("Baseline", case=False, na=False)
    ]

    baseline_random_forest = summary[
        summary["Model"].eq("Random Forest")
        & summary["Privacy Setting"].str.contains("Baseline", case=False, na=False)
    ]

    ordered = pd.concat(
        [k_logistic, k_random_forest, baseline_logistic, baseline_random_forest],
        ignore_index=True
    )

    return ordered


def make_label(row):
    model = row["Model"]

    if pd.notna(row["K"]):
        k_value = int(row["K"])

        if "Logistic Regression" in model:
            return f"K Logistic Regression\n(k={k_value})"

        if "Random Forest" in model:
            return f"K Random Forest\n(k={k_value})"

    if model == "Logistic Regression":
        return "Logistic Regression\n(Baseline No Privacy)"

    if model == "Random Forest":
        return "Random Forest\n(Baseline No Privacy)"

    return str(model)

def plot_k_anonymity_performance(summary, dataset_name):
    summary = prepare_k_order(summary)

    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]

    colors = {
        "Accuracy": "#2196F3",
        "F1 Score": "#4CAF50",
        "Precision": "#FF9800",
        "Recall": "#F44336"
    }

    labels = [make_label(row) for _, row in summary.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    fig.suptitle(
        f"{dataset_name} — K-Anonymity Performance (mean ± std, 5 seeds)",
        fontsize=15,
        fontweight="bold"
    )

    for metric, ax in zip(metrics, axes.flatten()):
        mean_col = f"{metric} mean"
        std_col = f"{metric} std"

        if mean_col not in summary.columns or std_col not in summary.columns:
            raise ValueError(
                f"Missing columns: {mean_col} or {std_col}. Check your summary CSV."
            )

        means = summary[mean_col].values
        stds = summary[std_col].values

        bars = ax.bar(
            x,
            means,
            color=colors[metric],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1.2}
        )

        for bar, mean, std in zip(bars, means, stds):
            if pd.isna(mean):
                continue

            if pd.isna(std):
                std = 0

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold"
            )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")

        ax.axhline(
            y=0.9,
            color="red",
            linestyle="--",
            alpha=0.3,
            label="0.9 threshold"
        )

        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()

    filename = f"{clean_dataset_name(dataset_name)}_k_anonymity_performance.png"
    output_path = GRAPHS_DIR / filename

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved: {filename}")

nhanes = load_summary("kanon_nhanes_results_summary.csv")
covid = load_summary("kanon_covid_results_summary.csv")

plot_k_anonymity_performance(nhanes, "NHANES")
plot_k_anonymity_performance(covid, "COVID-19")

print("\nK-anonymity performance graphs saved!")