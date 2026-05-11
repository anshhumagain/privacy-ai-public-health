from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    METRICS,
    METRIC_COLORS,
    RUNTIME_COLOR,
    LR_COLOR,
    BASELINE_LR_COLOR,
    DPI,
    style_axis,
    annotate_bars,
    parse_numeric,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)


def parse_mean_std(value):
    if isinstance(value, str) and "±" in value:
        mean, std = value.split("±")
        return float(mean.strip()), float(std.strip())
    return float(value), 0.0


def load_he_results(dataset_prefix):
    summary_path = RESULTS_DIR / f"he_{dataset_prefix}_results_summary.csv"
    normal_path = RESULTS_DIR / f"he_{dataset_prefix}_results.csv"

    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        df = pd.read_csv(normal_path)

    df = df.copy()

    if "Model" not in df.columns and "Method" in df.columns:
        df["Model"] = df["Method"]

    if "Privacy Setting" not in df.columns:
        df["Privacy Setting"] = df["Model"].apply(
            lambda x: "Encrypted Inference" if "HE" in str(x) else "Baseline"
        )

    for metric in METRICS:
        if f"{metric} mean" not in df.columns:
            means = []
            stds = []
            for value in df[metric]:
                mean, std = parse_mean_std(value)
                means.append(mean)
                stds.append(std)
            df[f"{metric} mean"] = means
            df[f"{metric} std"] = stds

    if "Runtime Seconds mean" not in df.columns:
        runtime_col = "Runtime Seconds" if "Runtime Seconds" in df.columns else "Train Time (s)"
        means = []
        stds = []
        for value in df[runtime_col]:
            mean, std = parse_mean_std(value)
            means.append(mean)
            stds.append(std)
        df["Runtime Seconds mean"] = means
        df["Runtime Seconds std"] = stds

    return sort_he_rows(df)


def sort_he_rows(df):
    df = df.copy()

    def rank(row):
        model = str(row["Model"])
        if "Normal" in model or model == "Logistic Regression" or "Baseline" in str(row["Privacy Setting"]):
            return 0
        if "HE" in model:
            return 1
        return 2

    df["_rank"] = df.apply(rank, axis=1)
    return df.sort_values("_rank").drop(columns="_rank")


def label_for_row(row):
    model = str(row["Model"])
    if "HE" in model:
        return "HE LR\n(Encrypted)"
    return "LR\n(Baseline)"


def plot_metrics(df, dataset, output_prefix):
    labels = [label_for_row(r) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"{dataset} — Homomorphic Encryption Performance",
        fontsize=18,
        fontweight="bold",
    )

    for metric, ax in zip(METRICS, axes.flatten()):
        means = df[f"{metric} mean"].values
        stds = df[f"{metric} std"].fillna(0).values

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=METRIC_COLORS[metric],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        annotate_bars(ax, bars, means, offset=0.01)

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        style_axis(ax, ylabel="Score")

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / f"{output_prefix}_metrics.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.show()
    print(f"Saved: graphs/{output_prefix}_metrics.png")


def plot_tradeoff(df, dataset, output_prefix):
    baseline = df[df["Model"].astype(str).str.contains("Normal|Logistic Regression", case=False, na=False)]
    he = df[df["Model"].astype(str).str.contains("HE", case=False, na=False)]

    if baseline.empty:
        baseline = df[df["Privacy Setting"].astype(str).str.contains("Baseline", case=False, na=False)]

    if he.empty:
        he = df[df["Privacy Setting"].astype(str).str.contains("Encrypted", case=False, na=False)]

    baseline = baseline.iloc[0]
    he = he.iloc[0]

    x = np.array([0, 1])
    x_labels = ["Baseline", "Encrypted\nInference"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{dataset} — Homomorphic Encryption Privacy-Utility Tradeoff",
        fontsize=18,
        fontweight="bold",
    )

    for metric, ax in zip(["Accuracy", "F1 Score"], axes):
        values = [baseline[f"{metric} mean"], he[f"{metric} mean"]]
        errors = [
            baseline[f"{metric} std"] if not pd.isna(baseline[f"{metric} std"]) else 0,
            he[f"{metric} std"] if not pd.isna(he[f"{metric} std"]) else 0,
        ]

        ax.errorbar(
            x,
            values,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=4,
            color=LR_COLOR,
            label="HE LR",
        )

        ax.axhline(
            baseline[f"{metric} mean"],
            linestyle="--",
            color=BASELINE_LR_COLOR,
            alpha=0.7,
            label="LR baseline",
        )

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xlabel("Inference setting", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1.05)
        style_axis(ax, ylabel="Score")
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / f"{output_prefix}_privacy_utility.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.show()
    print(f"Saved: graphs/{output_prefix}_privacy_utility.png")


def plot_runtime(df, dataset, output_prefix):
    labels = [label_for_row(r) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    means = df["Runtime Seconds mean"].values
    stds = df["Runtime Seconds std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(
        f"{dataset} — Homomorphic Encryption Runtime Comparison",
        fontsize=18,
        fontweight="bold",
    )

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=RUNTIME_COLOR,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    offset = max(means) * 0.02 if max(means) > 0 else 0.001
    annotate_bars(ax, bars, means, offset=offset, suffix="s")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    style_axis(ax, ylabel="Runtime (seconds)")

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / f"{output_prefix}_runtime.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.show()
    print(f"Saved: graphs/{output_prefix}_runtime.png")


for dataset, prefix, dataset_prefix in [
    ("NHANES", "he_nhanes", "nhanes"),
    ("COVID-19", "he_covid", "covid"),
]:
    data = load_he_results(dataset_prefix)

    plot_metrics(data, dataset, prefix)
    plot_tradeoff(data, dataset, prefix)
    plot_runtime(data, dataset, prefix)

print("\nAll HE visualisations saved.")
