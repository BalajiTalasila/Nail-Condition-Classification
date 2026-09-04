import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix
)

from sklearn.preprocessing import label_binarize


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

TRAINING_HISTORY_PATH = os.path.join(
    METRICS_DIR,
    "convnextv2_tiny_training_history.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    METRICS_DIR,
    "convnextv2_tiny_confusion_matrix.csv"
)

PREDICTIONS_PATH = os.path.join(
    METRICS_DIR,
    "convnextv2_tiny_test_predictions.csv"
)


# ============================================================
# TRAINING CURVES
# ============================================================

def generate_training_curves():
    print("\nGenerating training curves...")

    history_path = "results/metrics/convnextv2_tiny_training_history.csv"

    history = pd.read_csv(history_path)

    print("\nTraining history columns:")
    print(list(history.columns))

    # ==============================================================
    # FIGURE 1: TRAINING AND VALIDATION ACCURACY
    # ==============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        history["epoch"],
        history["validation_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ConvNeXtV2-Tiny Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    accuracy_path = "results/figures/convnextv2_tiny_accuracy_curve.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Accuracy curve saved: {accuracy_path}")

    # ==============================================================
    # FIGURE 2: TRAINING AND VALIDATION LOSS
    # ==============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXtV2-Tiny Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    loss_path = "results/figures/convnextv2_tiny_loss_curve.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Loss curve saved: {loss_path}")

    # ==============================================================
    # FIGURE 3: LEARNING RATE
    # ==============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["learning_rate"],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("ConvNeXtV2-Tiny Learning Rate Schedule")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    lr_path = "results/figures/convnextv2_tiny_learning_rate.png"

    plt.savefig(
        lr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Learning rate curve saved: {lr_path}")

    # ==============================================================
    # FIGURE 4: EPOCH TRAINING TIME
    # ==============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["epoch"],
        history["epoch_time_seconds"],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Time (seconds)")
    plt.title("ConvNeXtV2-Tiny Training Time per Epoch")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    time_path = "results/figures/convnextv2_tiny_epoch_time.png"

    plt.savefig(
        time_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Epoch time figure saved: {time_path}")


# ============================================================
# CONFUSION MATRIX HEATMAP
# ============================================================

def generate_confusion_matrix():

    print(
        "\nGenerating confusion matrix heatmap..."
    )

    cm_df = pd.read_csv(
        CONFUSION_MATRIX_PATH,
        index_col=0
    )

    class_names = cm_df.columns.tolist()

    cm = cm_df.values

    plt.figure(
        figsize=(11, 9)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        "Confusion Matrix: ConvNeXtV2-Tiny"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "confusion_matrix.png"
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


# ============================================================
# ROC CURVES
# ============================================================

def generate_roc_curves():

    print(
        "\nGenerating ROC curves..."
    )

    predictions_df = pd.read_csv(
        PREDICTIONS_PATH
    )

    # Get class names from probability columns
    probability_columns = [
        column
        for column in predictions_df.columns
        if column.startswith(
            "Probability_"
        )
    ]

    class_names = [
        column.replace(
            "Probability_",
            ""
        )
        for column in probability_columns
    ]

    # Map true labels to numerical labels
    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(class_names)
    }

    true_labels = predictions_df[
        "True_Label"
    ].map(
        class_to_index
    ).values

    probabilities = predictions_df[
        probability_columns
    ].values

    # Convert labels to one-vs-rest format
    true_labels_binarized = label_binarize(
        true_labels,
        classes=list(
            range(
                len(class_names)
            )
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    auc_results = []

    for class_index, class_name in enumerate(
        class_names
    ):

        fpr, tpr, _ = roc_curve(
            true_labels_binarized[
                :,
                class_index
            ],
            probabilities[
                :,
                class_index
            ]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        auc_results.append({
            "Class": class_name,
            "AUC": roc_auc
        })

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC = {roc_auc:.4f})"
            )
        )

    # Random classifier reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves: ConvNeXtV2-Tiny"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "roc_curves.png"
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

    # --------------------------------------------------------
    # SAVE AUC RESULTS
    # --------------------------------------------------------

    auc_df = pd.DataFrame(
        auc_results
    )

    auc_output_path = os.path.join(
        METRICS_DIR,
        "convnextv2_tiny_auc_scores.csv"
    )

    auc_df.to_csv(
        auc_output_path,
        index=False
    )

    print(
        "\nAUC Scores:"
    )

    print(
        auc_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: "
        f"{auc_output_path}"
    )

os.makedirs("results/figures", exist_ok=True)
# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING CONVNEXTV2-TINY FIGURES"
    )

    print(
        "=" * 70
    )

    generate_training_curves()

    generate_confusion_matrix()

    generate_roc_curves()

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL FIGURES GENERATED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFigures location:\n"
        f"{FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()