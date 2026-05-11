from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    METRICS,
    METRIC_COLORS,
    RUNTIME_COLOR,
    LR_COLOR,
    RF_COLOR,
    BASELINE_LR_COLOR,
    BASELINE_RF_COLOR,
    DPI,
    short_label,
    style_axis,
    annotate_bars,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)


def load_summary(path):
    return pd.read_csv(path)


def setting_rank(setting):
    if setting == "Baseline (No Privacy)":
        return 99

    if "3 clients, 10 rounds" in setting:
        return 1
    if "5 clients, 20 rounds" in setting:
        return 2
    if "10 clients, 20 rounds" in setting:
        return 3

    return 98


def model_rank(model):
    if model in ["Federated Logistic Regression", "Logistic Regression"]:
        return 0
    if model in ["Federated Random Forest", "Random Forest"]:
        return 1
    return 2


def sort_fl_rows(df):
    df = df.copy()
    df["_setting_rank"] = df["Privacy Setting"].apply(setting_rank)
    df["_model_rank"] = df["Model"].apply(model_rank)

    # FL settings first, baselines last.
    df = df.sort_values(["_setting_rank", "_model_rank"])

    return df.drop(columns=["_setting_rank", "_model_rank"])


def plot_metrics(df, dataset, output_prefix):
    df = sort_fl_rows(df)

    labels = [short_label(r["Model"], r["Privacy Setting"]) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"{dataset} — Federated Learning Performance",
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
        ax.set_xticklabels(labels, rotation=25, ha="right")
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
    df = sort_fl_rows(df)

    baseline = df[df["Privacy Setting"] == "Baseline (No Privacy)"]
    lr_base = baseline[baseline["Model"] == "Logistic Regression"].iloc[0]
    rf_base = baseline[baseline["Model"] == "Random Forest"].iloc[0]

    fl_df = df[df["Privacy Setting"] != "Baseline (No Privacy)"].copy()
    fl_df["_setting_rank"] = fl_df["Privacy Setting"].apply(setting_rank)
    fl_df = fl_df.sort_values(["_setting_rank", "_model_rank"] if "_model_rank" in fl_df.columns else ["_setting_rank"])

    setting_order = [
        "FL (3 clients, 10 rounds)",
        "FL (5 clients, 20 rounds)",
        "FL (10 clients, 20 rounds)",
    ]

    x_labels = [
        "3 clients\n10 rounds",
        "5 clients\n20 rounds",
        "10 clients\n20 rounds",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{dataset} — Federated Learning Privacy-Utility Tradeoff",
        fontsize=18,
        fontweight="bold",
    )

    for metric, ax in zip(["Accuracy", "F1 Score"], axes):
        lr_values = []
        lr_errors = []
        rf_values = []
        rf_errors = []

        for setting in setting_order:
            lr_row = fl_df[
                (fl_df["Model"] == "Federated Logistic Regression")
                & (fl_df["Privacy Setting"] == setting)
            ].iloc[0]

            rf_row = fl_df[
                (fl_df["Model"] == "Federated Random Forest")
                & (fl_df["Privacy Setting"] == setting)
            ].iloc[0]

            lr_values.append(lr_row[f"{metric} mean"])
            lr_errors.append(lr_row[f"{metric} std"] if not pd.isna(lr_row[f"{metric} std"]) else 0)

            rf_values.append(rf_row[f"{metric} mean"])
            rf_errors.append(rf_row[f"{metric} std"] if not pd.isna(rf_row[f"{metric} std"]) else 0)

        x = np.arange(len(x_labels))

        ax.errorbar(
            x,
            lr_values,
            yerr=lr_errors,
            marker="o",
            linewidth=2,
            capsize=4,
            color=LR_COLOR,
            label="FL LR",
        )

        ax.errorbar(
            x,
            rf_values,
            yerr=rf_errors,
            marker="o",
            linewidth=2,
            capsize=4,
            color=RF_COLOR,
            label="FL RF",
        )

        ax.axhline(
            lr_base[f"{metric} mean"],
            linestyle="--",
            color=BASELINE_LR_COLOR,
            alpha=0.7,
            label="LR baseline",
        )

        ax.axhline(
            rf_base[f"{metric} mean"],
            linestyle="--",
            color=BASELINE_RF_COLOR,
            alpha=0.7,
            label="RF baseline",
        )

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xlabel("Federated learning setting", fontsize=11)
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
    df = sort_fl_rows(df)

    labels = [short_label(r["Model"], r["Privacy Setting"]) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    means = df["Train Time (s) mean"].values
    stds = df["Train Time (s) std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(
        f"{dataset} — Federated Learning Runtime Comparison",
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
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axis(ax, ylabel="Runtime (seconds)")

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / f"{output_prefix}_runtime.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.show()
    print(f"Saved: graphs/{output_prefix}_runtime.png")


for dataset, prefix, file in [
    ("NHANES", "fl_nhanes", "fl_nhanes_results_summary.csv"),
    ("COVID-19", "fl_covid", "fl_covid_results_summary.csv"),
]:
    data = load_summary(RESULTS_DIR / file)

    plot_metrics(data, dataset, prefix)
    plot_tradeoff(data, dataset, prefix)
    plot_runtime(data, dataset, prefix)

print("\nAll FL visualisations saved.")
