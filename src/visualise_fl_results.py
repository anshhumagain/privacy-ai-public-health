from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"

GRAPHS_DIR.mkdir(exist_ok=True)


def load_summary(filepath):
    return pd.read_csv(filepath)


def plot_fl_metrics(summary, dataset_name, output_prefix):
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]

    labels = [
        f"{row['Model']}\n({row['Privacy Setting']})"
        for _, row in summary.iterrows()
    ]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, metric in enumerate(metrics):
        values = summary[f"{metric} mean"].tolist()
        stds = summary[f"{metric} std"].fillna(0).tolist()

        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            yerr=stds,
            capsize=3,
            label=metric,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val, std in zip(bars, values, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title(
        f"{dataset_name} — Federated Learning vs Baseline",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"{output_prefix}_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved graph: graphs/{output_prefix}_metrics.png")


def plot_fl_runtime(summary, dataset_name, output_prefix):
    labels = [
        f"{row['Model']}\n({row['Privacy Setting']})"
        for _, row in summary.iterrows()
    ]

    x = np.arange(len(labels))

    values = summary["Train Time (s) mean"].tolist()
    stds = summary["Train Time (s) std"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        x,
        values,
        yerr=stds,
        capsize=3,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    offset = max(values) * 0.02 if max(values) > 0 else 0.01

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        f"{dataset_name} — Federated Learning Runtime Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Training Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"{output_prefix}_runtime.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved graph: graphs/{output_prefix}_runtime.png")


def plot_fl_privacy_utility(summary, dataset_name, output_prefix):
    labels = [
        f"{row['Model']}\n({row['Privacy Setting']})"
        for _, row in summary.iterrows()
    ]

    x = np.arange(len(labels))

    accuracy = summary["Accuracy mean"]
    runtime = summary["Train Time (s) mean"]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.bar(
        x - 0.15,
        accuracy,
        width=0.3,
        label="Accuracy",
        alpha=0.85,
        edgecolor="white",
    )

    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()

    ax2.bar(
        x + 0.15,
        runtime,
        width=0.3,
        label="Train Time (s)",
        alpha=0.55,
        edgecolor="white",
    )

    ax2.set_ylabel("Training Time (seconds)")

    ax1.set_title(
        f"{dataset_name} — FL Privacy-Utility Tradeoff",
        fontsize=14,
        fontweight="bold",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"{output_prefix}_privacy_utility.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved graph: graphs/{output_prefix}_privacy_utility.png")


nhanes = load_summary(RESULTS_DIR / "fl_nhanes_results_summary.csv")
covid = load_summary(RESULTS_DIR / "fl_covid_results_summary.csv")

plot_fl_metrics(nhanes, "NHANES", "fl_nhanes")
plot_fl_runtime(nhanes, "NHANES", "fl_nhanes")
plot_fl_privacy_utility(nhanes, "NHANES", "fl_nhanes")

plot_fl_metrics(covid, "COVID-19", "fl_covid",)
plot_fl_runtime(covid, "COVID-19", "fl_covid")
plot_fl_privacy_utility(covid, "COVID-19", "fl_covid")

print("\nAll FL visualisations saved.")