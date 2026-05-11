import re
import numpy as np

METRICS = ["Accuracy", "F1 Score", "Precision", "Recall"]

METRIC_COLORS = {
    "Accuracy": "#4C9BE8",
    "F1 Score": "#6CC070",
    "Precision": "#F5A623",
    "Recall": "#F25F5C",
}

RUNTIME_COLOR = "#7A92A3"
LR_COLOR = "#E91E63"
RF_COLOR = "#7B1FA2"
BASELINE_LR_COLOR = "#4C9BE8"
BASELINE_RF_COLOR = "#6CC070"

DPI = 300


def parse_numeric(value):
    if isinstance(value, str) and "±" in value:
        value = value.split("±")[0].strip()
    return float(value)


def clean_model_name(model):
    model = str(model)
    replacements = {
        "DP Logistic Regression": "DP LR",
        "DP Random Forest": "DP RF",
        "Federated Logistic Regression": "FL LR",
        "Federated Random Forest": "FL RF",
        "K-anonymised Logistic Regression": "K-Anon LR",
        "K-anonymised Random Forest": "K-Anon RF",
        "HE Logistic Regression Inference": "HE LR",
        "Normal Logistic Regression": "LR",
        "Logistic Regression": "LR",
        "Random Forest": "RF",
    }
    return replacements.get(model, model)


def clean_setting(setting):
    setting = str(setting)

    if "Baseline" in setting:
        return "Baseline"

    k_match = re.search(r"k=(\d+)", setting, re.IGNORECASE)
    if k_match:
        return f"k={k_match.group(1)}"

    eps_match = re.search(r"e=(\d+\.?\d*)", setting)
    if eps_match:
        return f"ε={eps_match.group(1)}"

    fl_match = re.search(r"FL \((\d+) clients, (\d+) rounds\)", setting)
    if fl_match:
        return f"{fl_match.group(1)} clients\n{fl_match.group(2)} rounds"

    if "Encrypted Inference" in setting:
        return "Encrypted"

    return setting


def short_label(model, setting):
    return f"{clean_model_name(model)}\n({clean_setting(setting)})"


def style_axis(ax, ylabel="Score"):
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=9)


def annotate_bars(ax, bars, values, offset=0.01, suffix=""):
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.3f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
