import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

VISUALIZATIONS_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
)

VISUALIZATIONS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENTNET-B0 RESULTS VISUALIZATION")
print("=" * 70)


training_history_path = (
    METRICS_DIR
    / "efficientnet_b0_training_history.csv"
)

confusion_matrix_path = (
    METRICS_DIR
    / "efficientnet_b0_confusion_matrix.csv"
)

classwise_metrics_path = (
    METRICS_DIR
    / "efficientnet_b0_classwise_metrics.csv"
)


print("\nLoading training history...")

history_df = pd.read_csv(
    training_history_path
)


print("Loading confusion matrix...")

confusion_matrix_df = pd.read_csv(
    confusion_matrix_path,
    index_col=0
)


print("Loading class-wise metrics...")

classwise_metrics_df = pd.read_csv(
    classwise_metrics_path
)


print(
    f"\nTraining epochs available: "
    f"{len(history_df)}"
)


# ============================================================
# TRAINING AND VALIDATION LOSS
# ============================================================

print(
    "\nGenerating loss graph..."
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    label="Training Loss"
)


plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    marker="o",
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "EfficientNet-B0 Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


loss_plot_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_loss_curve.png"
)


plt.savefig(
    loss_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {loss_plot_path.name}"
)


# ============================================================
# TRAINING AND VALIDATION ACCURACY
# ============================================================

print(
    "\nGenerating accuracy graph..."
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)


plt.plot(
    history_df["epoch"],
    history_df["validation_accuracy"],
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "EfficientNet-B0 Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


accuracy_plot_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_accuracy_curve.png"
)


plt.savefig(
    accuracy_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {accuracy_plot_path.name}"
)


# ============================================================
# CONFUSION MATRIX HEATMAP
# ============================================================

print(
    "\nGenerating confusion matrix visualization..."
)


cm = confusion_matrix_df.values

class_names = (
    confusion_matrix_df.index.tolist()
)


plt.figure(
    figsize=(12, 10)
)


image = plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)


plt.colorbar(
    image
)


tick_marks = np.arange(
    len(class_names)
)


plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)


plt.yticks(
    tick_marks,
    class_names
)


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "EfficientNet-B0 Confusion Matrix"
)


threshold = (
    cm.max()
    / 2
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            format(
                cm[i, j],
                "d"
            ),

            horizontalalignment="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.tight_layout()


confusion_plot_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_confusion_matrix.png"
)


plt.savefig(
    confusion_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {confusion_plot_path.name}"
)


# ============================================================
# CLASS-WISE PRECISION
# ============================================================

print(
    "\nGenerating class-wise precision graph..."
)


plt.figure(
    figsize=(12, 6)
)


plt.bar(

    classwise_metrics_df["class"],

    classwise_metrics_df["precision"]
)


plt.xlabel(
    "Class"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "EfficientNet-B0 Class-wise Precision"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(
    0,
    1.1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


precision_plot_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_classwise_precision.png"
)


plt.savefig(
    precision_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {precision_plot_path.name}"
)


# ============================================================
# CLASS-WISE RECALL
# ============================================================

print(
    "\nGenerating class-wise recall graph..."
)


plt.figure(
    figsize=(12, 6)
)


plt.bar(

    classwise_metrics_df["class"],

    classwise_metrics_df["recall"]
)


plt.xlabel(
    "Class"
)

plt.ylabel(
    "Recall"
)

plt.title(
    "EfficientNet-B0 Class-wise Recall"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(
    0,
    1.1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


recall_plot_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_classwise_recall.png"
)


plt.savefig(
    recall_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {recall_plot_path.name}"
)


# ============================================================
# CLASS-WISE F1 SCORE
# ============================================================

print(
    "\nGenerating class-wise F1-score graph..."
)


plt.figure(
    figsize=(12, 6)
)


plt.bar(

    classwise_metrics_df["class"],

    classwise_metrics_df["f1_score"]
)


plt.xlabel(
    "Class"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "EfficientNet-B0 Class-wise F1 Score"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(
    0,
    1.1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


f1_plot_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_classwise_f1_score.png"
)


plt.savefig(
    f1_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {f1_plot_path.name}"
)


# ============================================================
# COMBINED CLASS-WISE METRICS
# ============================================================

print(
    "\nGenerating combined class-wise metrics graph..."
)


classes = (
    classwise_metrics_df["class"]
)

x = np.arange(
    len(classes)
)

width = 0.25


plt.figure(
    figsize=(14, 7)
)


plt.bar(

    x - width,

    classwise_metrics_df["precision"],

    width,

    label="Precision"
)


plt.bar(

    x,

    classwise_metrics_df["recall"],

    width,

    label="Recall"
)


plt.bar(

    x + width,

    classwise_metrics_df["f1_score"],

    width,

    label="F1 Score"
)


plt.xlabel(
    "Class"
)

plt.ylabel(
    "Score"
)

plt.title(
    "EfficientNet-B0 Class-wise Performance Metrics"
)

plt.xticks(

    x,

    classes,

    rotation=45,

    ha="right"
)

plt.ylim(
    0,
    1.1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


combined_metrics_path = (
    VISUALIZATIONS_DIR
    / "efficientnet_b0_combined_classwise_metrics.png"
)


plt.savefig(
    combined_metrics_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {combined_metrics_path.name}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "VISUALIZATION COMPLETED"
)

print("=" * 70)


print(
    "\nAll visualizations saved in:"
)

print(
    VISUALIZATIONS_DIR
)


print(
    "\nGenerated files:"
)


for file_path in sorted(
    VISUALIZATIONS_DIR.glob(
        "efficientnet_b0_*.png"
    )
):

    print(
        f" - {file_path.name}"
    )
