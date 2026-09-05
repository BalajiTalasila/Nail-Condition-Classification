import os
import time
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    matthews_corrcoef
)

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# ============================================================
# IMPORT MODEL
# ============================================================

from proposed_attention_efficientnet import AttentionEfficientNetB0

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "proposed_attention_efficientnet_b0_best.pth"
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_SIZE = 224


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("EVALUATING PROPOSED ATTENTION-EFFICIENTNET-B0 MODEL")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    # ========================================================
    # TEST TRANSFORMS
    # ========================================================

    test_transform = transforms.Compose([

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    # ========================================================
    # LOAD TEST DATASET
    # ========================================================

    print("\nLoading test dataset...")

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    class_names = test_dataset.classes

    print(f"Test Images: {len(test_dataset)}")
    print(f"Classes: {class_names}")

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\nCreating model...")

    model = AttentionEfficientNetB0(
        num_classes=len(class_names)
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\nLoading trained model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)

    model.eval()

    print("âœ“ Best model loaded successfully")

    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # EVALUATION VARIABLES
    # ========================================================

    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_paths = []

    total_loss = 0.0
    total_inference_time = 0.0
    total_inference_images = 0

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    print("\nEvaluating model...")

    with torch.no_grad():

        for images, labels in tqdm(
            test_loader,
            desc="Testing"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            total_inference_time += (
                end_time - start_time
            )

            total_inference_images += (
                images.size(0)
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    average_loss = (
        total_loss / len(test_dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    mcc = matthews_corrcoef(
        all_labels,
        all_predictions
    )

    average_inference_time_seconds = (
        total_inference_time / total_inference_images
    )

    average_inference_time_ms = (
        average_inference_time_seconds * 1000
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"\nTest Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"MCC: {mcc:.4f}")
    print(
        f"Average Inference Time: "
        f"{average_inference_time_ms:.4f} ms/image"
    )

    print("\nMACRO AVERAGE METRICS")

    print(
        f"Precision: "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"F1-Score: "
        f"{f1_macro * 100:.2f}%"
    )

    print("\nWEIGHTED AVERAGE METRICS")

    print(
        f"Precision: "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"F1-Score: "
        f"{f1_weighted * 100:.2f}%"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    # Save classification report

    report_path = os.path.join(
        METRICS_DIR,
        "proposed_model_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # Save raw confusion matrix

    cm_dataframe = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_csv_path = os.path.join(
        METRICS_DIR,
        "proposed_model_confusion_matrix.csv"
    )

    cm_dataframe.to_csv(
        cm_csv_path
    )

    # ========================================================
    # CREATE CONFUSION MATRIX FIGURE
    # ========================================================

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Confusion Matrix - Proposed Attention-EfficientNet-B0"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        FIGURES_DIR,
        "proposed_model_confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # NORMALIZED CONFUSION MATRIX
    # ========================================================

    cm_normalized = cm.astype(
        "float"
    ) / cm.sum(
        axis=1
    )[:, np.newaxis]

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Normalized Confusion Matrix - Proposed Model"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    normalized_cm_path = os.path.join(
        FIGURES_DIR,
        "proposed_model_normalized_confusion_matrix.png"
    )

    plt.savefig(
        normalized_cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    final_metrics = {

        "Accuracy": [
            accuracy
        ],

        "Precision_Macro": [
            precision_macro
        ],

        "Recall_Macro": [
            recall_macro
        ],

        "F1_Macro": [
            f1_macro
        ],

        "Precision_Weighted": [
            precision_weighted
        ],

        "Recall_Weighted": [
            recall_weighted
        ],

        "F1_Weighted": [
            f1_weighted
        ],

        "MCC": [
            mcc
        ],

        "Average_Inference_Time_Seconds": [
            average_inference_time_seconds
        ],

        "Average_Inference_Time_ms": [
            average_inference_time_ms
        ]

    }
    metrics_dataframe = pd.DataFrame(
        final_metrics
    )

    metrics_path = os.path.join(
        METRICS_DIR,
        "proposed_model_final_test_metrics.csv"
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    # Convert probabilities to NumPy array
    all_probabilities_array = np.array(
        all_probabilities
    )

    predictions_dataframe = pd.DataFrame({

        "True_Label": [
            class_names[label]
            for label in all_labels
        ],

        "Predicted_Label": [
            class_names[prediction]
            for prediction in all_predictions
        ],

        "Correct": (
            np.array(all_labels)
            == np.array(all_predictions)
        )

    })

    # Add probability column for each class
    for class_index, class_name in enumerate(class_names):

        predictions_dataframe[
            f"Probability_{class_name}"
        ] = all_probabilities_array[
            :,
            class_index
        ]

    predictions_path = os.path.join(
        METRICS_DIR,
        "proposed_model_test_predictions.csv"
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # COMPLETION MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nResults saved in:"
    )

    print(
        f"\nMetrics: "
        f"{METRICS_DIR}"
    )

    print(
        f"\nFigures: "
        f"{FIGURES_DIR}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()




