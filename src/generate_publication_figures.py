from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


DETAILED_RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_detailed_results"
)


CALIBRATION_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "confidence_calibration_analysis"
)


PUBLICATION_STATS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "publication_statistical_analysis"
)


FIGURES_DIR = (
    PROJECT_ROOT /
    "results" /
    "figures"
)


FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_NAMES = [
    "convnextv2_tiny",
    "densenet121",
    "efficientnet_b0",
    "proposed_attention_efficientnet_b0"
]


MODEL_DISPLAY_NAMES = {
    "convnextv2_tiny":
    "ConvNeXtV2-Tiny",

    "densenet121":
    "DenseNet121",

    "efficientnet_b0":
    "EfficientNet-B0",

    "proposed_attention_efficientnet_b0":
    "Proposed Model"
}


# ============================================================
# GLOBAL FIGURE SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})


print("=" * 80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("=" * 80)


# ============================================================
# LOAD CALIBRATION SUMMARY
# ============================================================

calibration_summary_file = (
    CALIBRATION_DIR /
    "model_calibration_summary.csv"
)


calibration_df = pd.read_csv(
    calibration_summary_file
)


print("\nLoaded calibration summary:")
print(calibration_summary_file)


# ============================================================
# FIGURE 1
# MODEL ACCURACY COMPARISON
# ============================================================

print("\nGenerating Figure 1: Model Accuracy Comparison")


plot_df = calibration_df.copy()


plot_df[
    "display_name"
] = plot_df[
    "model"
].map(
    MODEL_DISPLAY_NAMES
)


plot_df = plot_df.sort_values(
    "accuracy_percent",
    ascending=False
)


plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    plot_df["display_name"],
    plot_df["accuracy_percent"]
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(
    "Model Accuracy on Unseen-Source Test Set (n = 71)"
)


plt.ylim(
    0,
    100
)


for bar, value in zip(
    bars,
    plot_df["accuracy_percent"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()


output_file = (
    FIGURES_DIR /
    "model_accuracy_comparison.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(output_file)


# ============================================================
# FIGURE 2
# ACCURACY VS CONFIDENCE
# ============================================================

print("\nGenerating Figure 2: Accuracy vs Confidence")


plot_df = calibration_df.copy()


plot_df[
    "display_name"
] = plot_df[
    "model"
].map(
    MODEL_DISPLAY_NAMES
)


plot_df = plot_df.sort_values(
    "accuracy_percent",
    ascending=False
)


x = np.arange(
    len(plot_df)
)


width = 0.35


plt.figure(
    figsize=(11, 6)
)


bars_accuracy = plt.bar(
    x - width / 2,
    plot_df["accuracy_percent"],
    width,
    label="Accuracy"
)


bars_confidence = plt.bar(
    x + width / 2,
    plot_df["average_confidence_percent"],
    width,
    label="Average Confidence"
)


plt.xticks(
    x,
    plot_df["display_name"],
    rotation=15
)


plt.ylabel(
    "Percentage (%)"
)


plt.ylim(
    0,
    105
)


plt.title(
    "Accuracy and Average Prediction Confidence (n = 71)"
)


plt.legend()


for bars in [
    bars_accuracy,
    bars_confidence
]:
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}",
            ha="center",
            fontsize=9
        )


plt.tight_layout()


output_file = (
    FIGURES_DIR /
    "accuracy_vs_confidence.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(output_file)


# ============================================================
# FIGURE 3
# CALIBRATION METRICS
# ============================================================

print("\nGenerating Figure 3: Calibration Metrics")


plot_df = calibration_df.copy()


plot_df[
    "display_name"
] = plot_df[
    "model"
].map(
    MODEL_DISPLAY_NAMES
)


plot_df = plot_df.sort_values(
    "ece_percent",
    ascending=True
)


x = np.arange(
    len(plot_df)
)


width = 0.35


plt.figure(
    figsize=(11, 6)
)


bars_ece = plt.bar(
    x - width / 2,
    plot_df["ece_percent"],
    width,
    label="ECE"
)


bars_mce = plt.bar(
    x + width / 2,
    plot_df["mce_percent"],
    width,
    label="MCE"
)


plt.xticks(
    x,
    plot_df["display_name"],
    rotation=15
)


plt.ylabel(
    "Calibration Error (%)"
)


plt.title(
    "Confidence Calibration Error Comparison (n = 71)"
)


plt.legend()


for bars in [
    bars_ece,
    bars_mce
]:
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}",
            ha="center",
            fontsize=9
        )


plt.tight_layout()


output_file = (
    FIGURES_DIR /
    "model_calibration_metrics.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(output_file)


# ============================================================
# FIGURE 4
# BRIER SCORE COMPARISON
# ============================================================

print("\nGenerating Figure 4: Brier Score Comparison")


plot_df = calibration_df.copy()


plot_df[
    "display_name"
] = plot_df[
    "model"
].map(
    MODEL_DISPLAY_NAMES
)


plot_df = plot_df.sort_values(
    "brier_score",
    ascending=True
)


plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    plot_df["display_name"],
    plot_df["brier_score"]
)


plt.ylabel(
    "Brier Score"
)


plt.title(
    "Multiclass Brier Score Comparison (Lower is Better)"
)


for bar, value in zip(
    bars,
    plot_df["brier_score"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.002,
        f"{value:.4f}",
        ha="center"
    )


plt.tight_layout()


output_file = (
    FIGURES_DIR /
    "brier_score_comparison.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(output_file)


# ============================================================
# FIGURE 5
# PAIRED CONFIDENCE DIFFERENCES
# ============================================================

print("\nGenerating Figure 5: Paired Confidence Comparison")


comparison_files = [
    (
        "ConvNeXtV2-Tiny",
        PUBLICATION_STATS_DIR /
        "proposed_vs_convnextv2_tiny_paired_analysis.csv"
    ),

    (
        "DenseNet121",
        PUBLICATION_STATS_DIR /
        "proposed_vs_densenet121_paired_analysis.csv"
    ),

    (
        "EfficientNet-B0",
        PUBLICATION_STATS_DIR /
        "proposed_vs_efficientnet_b0_paired_analysis.csv"
    )
]


confidence_difference_data = []

comparison_labels = []


for model_label, file_path in comparison_files:

    if not file_path.exists():

        print(
            f"WARNING: File not found: "
            f"{file_path}"
        )

        continue


    dataframe = pd.read_csv(
        file_path
    )


    confidence_columns = [
        column
        for column in dataframe.columns
        if "confidence" in column.lower()
    ]


    difference_column = None


    for column in dataframe.columns:

        if (
            "difference"
            in column.lower()
            and
            "confidence"
            in column.lower()
        ):

            difference_column = column

            break


    if difference_column is None:

        print(
            f"WARNING: Confidence difference column "
            f"not found in {file_path}"
        )

        continue


    confidence_difference_data.append(
        dataframe[
            difference_column
        ] * 100
    )


    comparison_labels.append(
        f"Proposed -\n{model_label}"
    )


if confidence_difference_data:

    plt.figure(
        figsize=(11, 6)
    )


    plt.boxplot(
        confidence_difference_data,
        labels=comparison_labels
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.ylabel(
        "Confidence Difference (Percentage Points)"
    )


    plt.title(
        "Paired Prediction Confidence Differences"
    )


    plt.tight_layout()


    output_file = (
        FIGURES_DIR /
        "paired_confidence_comparison.png"
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(output_file)


# ============================================================
# FIGURE 6-9
# CONFUSION MATRICES
# ============================================================

print("\nGenerating Confusion Matrices")


for model_name in MODEL_NAMES:

    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        print(
            f"WARNING: Missing prediction file: "
            f"{prediction_file}"
        )

        continue


    dataframe = pd.read_csv(
        prediction_file
    )


    class_names = sorted(
        dataframe[
            "true_class"
        ].unique()
    )


    matrix = confusion_matrix(
        dataframe["true_class"],
        dataframe["predicted_class"],
        labels=class_names
    )


    figure, axis = plt.subplots(
        figsize=(10, 8)
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )


    display.plot(
        ax=axis,
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )


    axis.set_title(
        f"{MODEL_DISPLAY_NAMES[model_name]} "
        f"Confusion Matrix (n = 71)"
    )


    plt.tight_layout()


    output_file = (
        FIGURES_DIR /
        f"{model_name}_confusion_matrix.png"
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(output_file)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("ALL PUBLICATION FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)

print("\nFigures directory:")
print(FIGURES_DIR)
