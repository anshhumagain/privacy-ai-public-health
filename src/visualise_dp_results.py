from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"

GRAPHS_DIR.mkdir(exist_ok=True)


def load_summary(filepath):
    return pd.read_csv(filepath)


def safe_dataset_key(dataset_name):
    if dataset_name == "NHANES":
        return "dp_nhanes"
    if dataset_name == "COVID-19":
        return "dp_covid_19"
    return "dp_" + dataset_name.lower().replace("-", "_").replace(" ", "_")


def plot_results_with_errorbars(summary, dataset_name):
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    labels = [
        f"{row['Model']}\n({row['Privacy Setting']})"
        for _, row in summary.iterrows()
    ]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    fig.suptitle(
        f"{dataset_name} — Differential Privacy Performance (mean ± std, 5 seeds)",
        fontsize=15,
        fontweight="bold",
    )

    for metric, ax, color in zip(metrics, axes.flatten(), colors):
        means = summary[f"{metric} mean"].values
        stds = summary[f"{metric} std"].values

        bars = ax.bar(
            x,
            means,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
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
            label="0.9 threshold",
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()

    fname = f"{safe_dataset_key(dataset_name)}_results.png"
    plt.savefig(GRAPHS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: graphs/{fname}")


def plot_dp_tradeoff(summary, dataset_name):
    dp_df = summary[summary["Privacy Setting"].str.startswith("DP")].copy()
    dp_df["Epsilon"] = (
        dp_df["Privacy Setting"]
        .str.extract(r"e=(\d+\.?\d*)")
        .astype(float)
    )

    baseline = summary[summary["Privacy Setting"] == "Baseline (No Privacy)"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"{dataset_name} — Differential Privacy Utility Tradeoff (mean ± std)",
        fontsize=14,
        fontweight="bold",
    )

    model_colors = {
        "DP Logistic Regression": "#E91E63",
        "DP Random Forest": "#9C27B0",
    }

    for ax, metric in zip(axes, ["Accuracy", "F1 Score"]):
        for model_name, color in model_colors.items():
            model_data = dp_df[dp_df["Model"] == model_name].sort_values(
                "Epsilon",
                ascending=False,
            )

            if model_data.empty:
                continue

            ax.errorbar(
                model_data["Epsilon"],
                model_data[f"{metric} mean"],
                yerr=model_data[f"{metric} std"],
                marker="o",
                linewidth=2,
                capsize=4,
                label=model_name,
                color=color,
            )

        baseline_colors = {
            "Logistic Regression": "#2196F3",
            "Random Forest": "#4CAF50",
        }

        for _, row in baseline.iterrows():
            ax.axhline(
                y=row[f"{metric} mean"],
                linestyle="--",
                alpha=0.6,
                color=baseline_colors.get(row["Model"], "gray"),
                label=f"{row['Model']} Baseline",
            )

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epsilon (← less private | more private →)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.invert_xaxis()
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    fname = f"{safe_dataset_key(dataset_name)}_tradeoff.png"
    plt.savefig(GRAPHS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: graphs/{fname}")


def plot_runtime(summary, dataset_name):
    labels = [
        f"{row['Model']}\n({row['Privacy Setting']})"
        for _, row in summary.iterrows()
    ]
    x = np.arange(len(labels))

    means = summary["Train Time (s) mean"].values
    stds = summary["Train Time (s) std"].values

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(
        f"{dataset_name} — Differential Privacy Runtime Comparison (mean ± std)",
        fontsize=14,
        fontweight="bold",
    )

    bars = ax.bar(
        x,
        means,
        color="#607D8B",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        yerr=stds,
        capsize=3,
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{mean:.3f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Training Time (seconds)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fname = f"{safe_dataset_key(dataset_name)}_runtime.png"
    plt.savefig(GRAPHS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: graphs/{fname}")


nhanes = load_summary(RESULTS_DIR / "dp_nhanes_results_summary.csv")
covid = load_summary(RESULTS_DIR / "dp_covid_results_summary.csv")

plot_results_with_errorbars(nhanes, "NHANES")
plot_dp_tradeoff(nhanes, "NHANES")
plot_runtime(nhanes, "NHANES")

plot_results_with_errorbars(covid, "COVID-19")
plot_dp_tradeoff(covid, "COVID-19")
plot_runtime(covid, "COVID-19")

print("\nAll DP graphs saved!")