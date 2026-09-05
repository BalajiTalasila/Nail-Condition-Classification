import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


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
    "unseen_source_unified_analysis"
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
# MODEL NAMES
# =============================================================================

model_names = df[
    "model_display"
].tolist()


# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("=" * 100)
print("GENERATING UNIFIED UNSEEN-SOURCE VISUALIZATIONS")
print("=" * 100)

print()
print("Input file:")
print(INPUT_FILE)

print()
print("Models:")
for model in model_names:
    print(" -", model)


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def add_value_labels(
    bars,
    decimals=2
):

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() +
            bar.get_width() / 2,

            height,

            f"{height:.{decimals}f}",

            ha="center",

            va="bottom",

            fontsize=10
        )


# =============================================================================
# FIGURE 1:
# ACCURACY COMPARISON
# =============================================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    model_names,
    df["accuracy_percent"]
)

add_value_labels(
    bars
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Unseen-Source Classification Accuracy Comparison"
)

plt.ylim(
    0,
    105
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

accuracy_file = (
    OUTPUT_DIR /
    "unified_unseen_source_accuracy_comparison.png"
)

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(accuracy_file)


# =============================================================================
# FIGURE 2:
# MACRO F1 COMPARISON
# =============================================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    model_names,
    df["macro_f1_percent"]
)

add_value_labels(
    bars
)

plt.ylabel(
    "Macro F1-score (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Unseen-Source Macro F1-score Comparison"
)

plt.ylim(
    0,
    105
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

macro_f1_file = (
    OUTPUT_DIR /
    "unified_unseen_source_macro_f1_comparison.png"
)

plt.savefig(
    macro_f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(macro_f1_file)


# =============================================================================
# FIGURE 3:
# ACCURACY VS CONFIDENCE
# =============================================================================

x = np.arange(
    len(model_names)
)

width = 0.35


plt.figure(
    figsize=(13, 7)
)

bars1 = plt.bar(
    x - width / 2,
    df["accuracy_percent"],
    width,
    label="Accuracy"
)

bars2 = plt.bar(
    x + width / 2,
    df["average_confidence_percent"],
    width,
    label="Average Confidence"
)

for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() +
            bar.get_width() / 2,

            height,

            f"{height:.2f}",

            ha="center",

            va="bottom",

            fontsize=9
        )


plt.ylabel(
    "Percentage (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Accuracy vs Average Confidence on Unseen Source"
)

plt.xticks(
    x,
    model_names,
    rotation=15,
    ha="right"
)

plt.ylim(
    0,
    110
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

accuracy_confidence_file = (
    OUTPUT_DIR /
    "unified_unseen_source_accuracy_vs_confidence.png"
)

plt.savefig(
    accuracy_confidence_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(accuracy_confidence_file)


# =============================================================================
# FIGURE 4:
# ECE COMPARISON
# =============================================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    model_names,
    df["ece_percent"]
)

add_value_labels(
    bars
)

plt.ylabel(
    "Expected Calibration Error (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Unseen-Source Expected Calibration Error Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

ece_file = (
    OUTPUT_DIR /
    "unified_unseen_source_ece_comparison.png"
)

plt.savefig(
    ece_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(ece_file)


# =============================================================================
# FIGURE 5:
# BRIER SCORE COMPARISON
# =============================================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    model_names,
    df["brier_score"]
)

add_value_labels(
    bars,
    decimals=4
)

plt.ylabel(
    "Brier Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Unseen-Source Brier Score Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

brier_file = (
    OUTPUT_DIR /
    "unified_unseen_source_brier_score_comparison.png"
)

plt.savefig(
    brier_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(brier_file)


# =============================================================================
# FIGURE 6:
# HIGH-CONFIDENCE ERRORS
# =============================================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    model_names,
    df["high_confidence_errors"]
)

add_value_labels(
    bars,
    decimals=0
)

plt.ylabel(
    "Number of Errors"
)

plt.xlabel(
    "Model"
)

plt.title(
    "High-Confidence Errors (Confidence = 90%)"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

high_confidence_file = (
    OUTPUT_DIR /
    "unified_unseen_source_high_confidence_errors.png"
)

plt.savefig(
    high_confidence_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(high_confidence_file)


# =============================================================================
# FIGURE 7:
# CONFIDENCE-ACCURACY GAP
# =============================================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    model_names,
    df["confidence_accuracy_gap_percent"]
)

for bar in bars:

    height = bar.get_height()

    if height >= 0:

        vertical_alignment = "bottom"

    else:

        vertical_alignment = "top"


    plt.text(
        bar.get_x() +
        bar.get_width() / 2,

        height,

        f"{height:.2f}",

        ha="center",

        va=vertical_alignment,

        fontsize=10
    )


plt.axhline(
    y=0,
    linewidth=1
)

plt.ylabel(
    "Confidence - Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Confidence-Accuracy Gap on Unseen Source"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

gap_file = (
    OUTPUT_DIR /
    "unified_unseen_source_confidence_accuracy_gap.png"
)

plt.savefig(
    gap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(gap_file)


# =============================================================================
# FIGURE 8:
# OVERALL MODEL COMPARISON
# =============================================================================

comparison_columns = [
    "accuracy_percent",
    "macro_f1_percent",
    "average_confidence_percent"
]

comparison_labels = [
    "Accuracy",
    "Macro F1",
    "Average Confidence"
]


x = np.arange(
    len(model_names)
)

width = 0.25


plt.figure(
    figsize=(14, 8)
)


for index, (
    column,
    label
) in enumerate(
    zip(
        comparison_columns,
        comparison_labels
    )
):

    bars = plt.bar(
        x +
        (
            index - 1
        ) *
        width,

        df[column],

        width,

        label=label
    )


plt.xticks(
    x,
    model_names,
    rotation=15,
    ha="right"
)

plt.ylabel(
    "Percentage (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Overall Performance Comparison on Unseen Source"
)

plt.ylim(
    0,
    110
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

overall_file = (
    OUTPUT_DIR /
    "unified_unseen_source_overall_performance.png"
)

plt.savefig(
    overall_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Generated:")
print(overall_file)


# =============================================================================
# CREATE A CLEAN SUMMARY TABLE
# =============================================================================

summary_columns = [

    "model_display",

    "accuracy_percent",

    "macro_precision_percent",

    "macro_recall_percent",

    "macro_f1_percent",

    "weighted_f1_percent",

    "average_confidence_percent",

    "confidence_accuracy_gap_percent",

    "ece_percent",

    "brier_score",

    "high_confidence_errors",

    "calibration_behavior"

]


summary_df = df[
    summary_columns
].copy()


summary_file = (
    OUTPUT_DIR /
    "unified_unseen_source_final_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


print()
print("=" * 100)
print("VISUALIZATION GENERATION COMPLETED SUCCESSFULLY")
print("=" * 100)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Final summary:")
print(summary_file)

print()
print("Total figures generated: 8")

