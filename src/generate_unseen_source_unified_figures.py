from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path.cwd()

INPUT_FILE = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_unified_analysis" /
    "unified_unseen_source_model_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "figures" /
    "unseen_source_unified"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# LOAD DATA
# =============================================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Unified comparison file not found:\n"
        f"{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


# =============================================================================
# MODEL DISPLAY ORDER
# =============================================================================

model_order = [

    "ConvNeXtV2-Tiny",

    "DenseNet121",

    "EfficientNet-B0",

    "Proposed Attention EfficientNet-B0"
]


df["model_display"] = pd.Categorical(
    df["model_display"],
    categories=model_order,
    ordered=True
)


df = df.sort_values(
    "model_display"
)


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def save_bar_chart(
    data,
    value_column,
    ylabel,
    title,
    filename,
    lower_is_better=False
):

    plot_df = data.copy()


    if lower_is_better:

        plot_df = plot_df.sort_values(
            value_column,
            ascending=True
        )

    else:

        plot_df = plot_df.sort_values(
            value_column,
            ascending=False
        )


    plt.figure(
        figsize=(11, 7)
    )


    bars = plt.bar(
        plot_df["model_display"],
        plot_df[value_column]
    )


    plt.xlabel(
        "Model",
        fontsize=12
    )


    plt.ylabel(
        ylabel,
        fontsize=12
    )


    plt.title(
        title,
        fontsize=14,
        fontweight="bold"
    )


    plt.xticks(
        rotation=15,
        ha="right"
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    for bar, value in zip(
        bars,
        plot_df[value_column]
    ):

        plt.text(

            bar.get_x() +
            bar.get_width() / 2,

            bar.get_height(),

            f"{value:.2f}",

            ha="center",

            va="bottom",

            fontsize=10
        )


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR /
        filename
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved: {output_path}"
    )


# =============================================================================
# FIGURE 1
# ACCURACY COMPARISON
# =============================================================================

save_bar_chart(

    data=df,

    value_column="accuracy_percent",

    ylabel="Accuracy (%)",

    title=(
        "Unseen-Source Classification Accuracy Comparison"
    ),

    filename=(
        "unseen_source_accuracy_comparison.png"
    )
)


# =============================================================================
# FIGURE 2
# MACRO F1 COMPARISON
# =============================================================================

save_bar_chart(

    data=df,

    value_column="macro_f1_percent",

    ylabel="Macro F1-score (%)",

    title=(
        "Unseen-Source Macro F1-score Comparison"
    ),

    filename=(
        "unseen_source_macro_f1_comparison.png"
    )
)


# =============================================================================
# FIGURE 3
# ECE COMPARISON
# =============================================================================

save_bar_chart(

    data=df,

    value_column="ece_percent",

    ylabel="Expected Calibration Error (%)",

    title=(
        "Unseen-Source Expected Calibration Error Comparison"
    ),

    filename=(
        "unseen_source_ece_comparison.png"
    ),

    lower_is_better=True
)


# =============================================================================
# FIGURE 4
# BRIER SCORE COMPARISON
# =============================================================================

save_bar_chart(

    data=df,

    value_column="brier_score",

    ylabel="Brier Score",

    title=(
        "Unseen-Source Brier Score Comparison"
    ),

    filename=(
        "unseen_source_brier_score_comparison.png"
    ),

    lower_is_better=True
)


# =============================================================================
# FIGURE 5
# CONFIDENCE VS ACCURACY
# =============================================================================

plt.figure(
    figsize=(11, 7)
)


x = range(
    len(df)
)


plt.plot(

    x,

    df["accuracy_percent"],

    marker="o",

    linewidth=2,

    label="Accuracy"
)


plt.plot(

    x,

    df[
        "average_confidence_percent"
    ],

    marker="s",

    linewidth=2,

    label="Average Confidence"
)


plt.xticks(

    x,

    df["model_display"],

    rotation=15,

    ha="right"
)


plt.xlabel(
    "Model",
    fontsize=12
)


plt.ylabel(
    "Percentage (%)",
    fontsize=12
)


plt.title(
    "Accuracy versus Average Prediction Confidence",
    fontsize=14,
    fontweight="bold"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


output_path = (
    OUTPUT_DIR /
    "unseen_source_accuracy_confidence_comparison.png"
)


plt.savefig(

    output_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {output_path}"
)


# =============================================================================
# FIGURE 6
# COMBINED PERFORMANCE COMPARISON
# =============================================================================

comparison_columns = [

    "accuracy_percent",

    "macro_precision_percent",

    "macro_recall_percent",

    "macro_f1_percent",

    "weighted_f1_percent"
]


plot_df = df[
    [
        "model_display"
    ] +
    comparison_columns
].copy()


plot_df = plot_df.set_index(
    "model_display"
)


plot_df.T.plot(

    kind="bar",

    figsize=(13, 8)
)


plt.xlabel(
    "Performance Metric",
    fontsize=12
)


plt.ylabel(
    "Score (%)",
    fontsize=12
)


plt.title(
    "Comprehensive Unseen-Source Classification Performance",
    fontsize=14,
    fontweight="bold"
)


plt.xticks(
    rotation=0
)


plt.legend(
    title="Model"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


output_path = (
    OUTPUT_DIR /
    "unseen_source_comprehensive_performance_comparison.png"
)


plt.savefig(

    output_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    f"Saved: {output_path}"
)


# =============================================================================
# COMPLETION MESSAGE
# =============================================================================

print()

print("=" * 100)

print(
    "UNIFIED UNSEEN-SOURCE FIGURE GENERATION COMPLETED"
)

print("=" * 100)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)