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


K_ORDER = [2, 5, 10, 20, 50, 100]


def load_summary(path):
    return pd.read_csv(path)


def extract_k(setting):
    if "k=" not in str(setting):
        return None
    return int(str(setting).split("k=")[1])


def model_rank(model):
    if "Logistic" in str(model):
        return 0
    if "Random Forest" in str(model):
        return 1
    return 2


def setting_rank(setting):
    if "Baseline" in str(setting):
        return 99

    k = extract_k(setting)
    if k in K_ORDER:
        return K_ORDER.index(k)

    return 98


def sort_kanon_rows(df):
    df = df.copy()
    df["_setting_rank"] = df["Privacy Setting"].apply(setting_rank)
    df["_model_rank"] = df["Model"].apply(model_rank)
    df = df.sort_values(["_setting_rank", "_model_rank"])
    return df.drop(columns=["_setting_rank", "_model_rank"])


def plot_metrics(df, dataset, output_prefix):
    df = sort_kanon_rows(df)

    labels = [short_label(r["Model"], r["Privacy Setting"]) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"{dataset} — K-Anonymity Performance",
        fontsize=18,
        fontweight="bold",
    )

    for metric, ax in zip(METRICS, axes.flatten()):
        means = df[f"{metric} mean"].values
        stds = df[f"{metric} std"].fillna(0).values if f"{metric} std" in df.columns else np.zeros(len(df))

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
    df = sort_kanon_rows(df)

    baseline = df[df["Privacy Setting"].str.contains("Baseline", case=False, na=False)]
    lr_base = baseline[baseline["Model"].str.contains("Logistic", case=False, na=False)].iloc[0]
    rf_base = baseline[baseline["Model"].str.contains("Random Forest", case=False, na=False)].iloc[0]

    kanon_df = df[~df["Privacy Setting"].str.contains("Baseline", case=False, na=False)].copy()
    kanon_df["K"] = kanon_df["Privacy Setting"].apply(extract_k)
    kanon_df = kanon_df[kanon_df["K"].notna()]
    kanon_df["K"] = kanon_df["K"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{dataset} — K-Anonymity Privacy-Utility Tradeoff",
        fontsize=18,
        fontweight="bold",
    )

    for metric, ax in zip(["Accuracy", "F1 Score"], axes):
        for model_contains, color, label in [
            ("Logistic", LR_COLOR, "K-Anon LR"),
            ("Random Forest", RF_COLOR, "K-Anon RF"),
        ]:
            model_df = kanon_df[
                kanon_df["Model"].str.contains(model_contains, case=False, na=False)
            ].copy()

            model_df = model_df.sort_values("K")

            if model_df.empty:
                continue

            yerr = (
                model_df[f"{metric} std"].fillna(0).values
                if f"{metric} std" in model_df.columns
                else np.zeros(len(model_df))
            )

            ax.errorbar(
                model_df["K"],
                model_df[f"{metric} mean"],
                yerr=yerr,
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
        ax.set_xlabel("K value (lower k → higher privacy risk | higher k → stronger anonymity)", fontsize=11)
        ax.set_xticks(K_ORDER)
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
    df = sort_kanon_rows(df)

    labels = [short_label(r["Model"], r["Privacy Setting"]) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    if "Train Time (s) mean" in df.columns:
        means = df["Train Time (s) mean"].values
        stds = df["Train Time (s) std"].fillna(0).values if "Train Time (s) std" in df.columns else np.zeros(len(df))
    elif "Runtime Seconds mean" in df.columns:
        means = df["Runtime Seconds mean"].values
        stds = df["Runtime Seconds std"].fillna(0).values if "Runtime Seconds std" in df.columns else np.zeros(len(df))
    else:
        raise ValueError("No runtime column found. Expected Train Time (s) mean or Runtime Seconds mean.")

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(
        f"{dataset} — K-Anonymity Runtime Comparison",
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
    ("NHANES", "kanon_nhanes", "kanon_nhanes_results_summary.csv"),
    ("COVID-19", "kanon_covid", "kanon_covid_results_summary.csv"),
]:
    data = load_summary(RESULTS_DIR / file)

    plot_metrics(data, dataset, prefix)
    plot_tradeoff(data, dataset, prefix)
    plot_runtime(data, dataset, prefix)

print("\nAll K-Anonymity visualisations saved.")
