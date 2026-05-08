import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "results" / "he_nhanes_results.csv"
GRAPHS_DIR = BASE_DIR / "graphs"

GRAPHS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

def plot_he_metrics(df):
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    models = df["Method"].tolist()

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=metric,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold"
            )

    ax.set_title(
        "NHANES — Homomorphic Encryption vs Normal Logistic Regression",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "he_nhanes_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Saved graph: graphs/he_nhanes_metrics.png")

def plot_he_runtime(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        df["Method"],
        df["Runtime Seconds"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5
    )

    for bar, val in zip(bars, df["Runtime Seconds"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(df["Runtime Seconds"]) * 0.02),
            f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_title(
        "NHANES — Runtime Cost of Homomorphic Encryption",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("Runtime Seconds")
    ax.set_xticks(range(len(df["Method"])))
    ax.set_xticklabels(df["Method"], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "he_nhanes_runtime.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Saved graph: graphs/he_nhanes_runtime.png")

def plot_he_privacy_utility(df):
    fig, ax1 = plt.subplots(figsize=(11, 6))

    methods = df["Method"]
    accuracy = df["Accuracy"]
    runtime = df["Runtime Seconds"]

    x = np.arange(len(methods))

    ax1.bar(
        x - 0.15,
        accuracy,
        width=0.3,
        label="Accuracy",
        alpha=0.85,
        edgecolor="white"
    )

    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.15,
        runtime,
        width=0.3,
        label="Runtime Seconds",
        alpha=0.55,
        edgecolor="white"
    )

    ax2.set_ylabel("Runtime Seconds")

    ax1.set_title(
        "NHANES — HE Privacy-Utility Tradeoff",
        fontsize=14,
        fontweight="bold"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")

    ax1.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "he_nhanes_privacy_utility.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Saved graph: graphs/he_nhanes_privacy_utility.png")

plot_he_metrics(df)
plot_he_runtime(df)
plot_he_privacy_utility(df)

print("\nAll HE visualisations saved.")