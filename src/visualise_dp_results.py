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


def sort_dp_rows(df):
    df = df.copy()

    def key(row):
        setting = row["Privacy Setting"]
        model = row["Model"]

        if "Baseline" in setting:
            return (99, 0 if model == "Logistic Regression" else 1)

        eps = float(setting.split("e=")[1])
        model_rank = 0 if "Logistic" in model else 1

        # Sort from less private to more private:
        # ε=10 → ε=5 → ε=1 → ε=0.5 → ε=0.1 → baseline
        return (-eps, model_rank)

    df["_sort"] = df.apply(key, axis=1)
    return df.sort_values("_sort").drop(columns="_sort")


def plot_metrics(df, dataset, output_prefix):
    df = sort_dp_rows(df)

    labels = [short_label(r["Model"], r["Privacy Setting"]) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"{dataset} — Differential Privacy Performance",
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
    df = sort_dp_rows(df)

    dp = df[df["Privacy Setting"].str.startswith("DP")].copy()
    dp["Epsilon"] = dp["Privacy Setting"].str.extract(r"e=(\d+\.?\d*)").astype(float)

    baseline = df[df["Privacy Setting"].str.contains("Baseline", case=False, na=False)]

    lr_base = baseline[baseline["Model"] == "Logistic Regression"].iloc[0]
    rf_base = baseline[baseline["Model"] == "Random Forest"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{dataset} — Differential Privacy Privacy-Utility Tradeoff",
        fontsize=18,
        fontweight="bold",
    )

    for metric, ax in zip(["Accuracy", "F1 Score"], axes):
        for model, color, label in [
            ("DP Logistic Regression", LR_COLOR, "DP LR"),
            ("DP Random Forest", RF_COLOR, "DP RF"),
        ]:
            model_df = dp[dp["Model"] == model].sort_values(
                "Epsilon",
                ascending=False,
            )

            if model_df.empty:
                continue

            ax.errorbar(
                model_df["Epsilon"],
                model_df[f"{metric} mean"],
                yerr=model_df[f"{metric} std"].fillna(0),
                marker="o",
                linewidth=2,
                capsize=4,
                color=color,
                label=label,
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
        ax.set_xlabel("Epsilon (less private → more private)", fontsize=11)
        ax.set_ylim(0, 1.05)

        # Places ε=10 on the left and ε=0.1 on the right.
        ax.invert_xaxis()

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
    df = sort_dp_rows(df)

    labels = [short_label(r["Model"], r["Privacy Setting"]) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    means = df["Train Time (s) mean"].values
    stds = df["Train Time (s) std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(
        f"{dataset} — Differential Privacy Runtime Comparison",
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
    ("NHANES", "dp_nhanes", "dp_nhanes_results_summary.csv"),
    ("COVID-19", "dp_covid", "dp_covid_results_summary.csv"),
]:
    data = load_summary(RESULTS_DIR / file)

    plot_metrics(data, dataset, prefix)
    plot_tradeoff(data, dataset, prefix)
    plot_runtime(data, dataset, prefix)

print("\nAll DP visualisations saved.")
