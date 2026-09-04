import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "test"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "models",
    "densenet121_best.pth"
)

METRICS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "metrics"
)

os.makedirs(METRICS_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 6
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# TEST TRANSFORMATIONS
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("DENSENET121 TEST EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # LOAD TEST DATASET
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    class_names = test_dataset.classes

    print(f"\nTest images: {len(test_dataset)}")
    print(f"Classes: {len(class_names)}")

    print("\nClass mapping:")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading DENSENET121 model...")

    model = timm.create_model(
        "densenet121",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    print("Loading trained weights...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handles either a direct state_dict or checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    all_predictions = []
    all_targets = []
    all_probabilities = []

    total_inference_time = 0.0
    total_images = 0

    print("\nRunning test evaluation...\n")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            # Synchronize before timing GPU inference
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            inference_time = end_time - start_time

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            total_inference_time += inference_time
            total_images += images.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    mcc = matthews_corrcoef(
        all_targets,
        all_predictions
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_targets,
        all_predictions
    )

    # --------------------------------------------------------
    # SPECIFICITY PER CLASS
    # --------------------------------------------------------

    specificity_scores = []

    for class_index in range(NUM_CLASSES):

        tp = cm[class_index, class_index]

        fn = np.sum(cm[class_index, :]) - tp

        fp = np.sum(cm[:, class_index]) - tp

        tn = (
            np.sum(cm)
            - tp
            - fn
            - fp
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        specificity_scores.append(
            specificity
        )

    # --------------------------------------------------------
    # CLASS-WISE METRICS
    # --------------------------------------------------------

    precision_class, recall_class, f1_class, support_class = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0
        )
    )

    class_metrics_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision_class,
        "Recall": recall_class,
        "F1_Score": f1_class,
        "Specificity": specificity_scores,
        "Support": support_class
    })

    # --------------------------------------------------------
    # OVERALL METRICS DATAFRAME
    # --------------------------------------------------------

    average_inference_time = (
        total_inference_time
        / total_images
    )

    metrics_df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro,
        "F1_Macro": f1_macro,
        "Precision_Weighted": precision_weighted,
        "Recall_Weighted": recall_weighted,
        "F1_Weighted": f1_weighted,
        "MCC": mcc,
        "Average_Inference_Time_Seconds": (
            average_inference_time
        ),
        "Average_Inference_Time_ms": (
            average_inference_time * 1000
        )
    }])

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    metrics_path = os.path.join(
        METRICS_DIR,
        "densenet121_test_metrics.csv"
    )

    class_metrics_path = os.path.join(
        METRICS_DIR,
        "densenet121_class_metrics.csv"
    )

    confusion_matrix_path = os.path.join(
        METRICS_DIR,
        "densenet121_confusion_matrix.csv"
    )

    predictions_path = os.path.join(
        METRICS_DIR,
        "densenet121_test_predictions.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    class_metrics_df.to_csv(
        class_metrics_path,
        index=False
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(
        confusion_matrix_path
    )

    predictions_df = pd.DataFrame({
        "True_Label": [
            class_names[index]
            for index in all_targets
        ],

        "Predicted_Label": [
            class_names[index]
            for index in all_predictions
        ],

        "Correct": (
            all_targets
            == all_predictions
        )
    })

    # Add probability columns
    for class_index, class_name in enumerate(class_names):

        predictions_df[
            f"Probability_{class_name}"
        ] = all_probabilities[:, class_index]

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Macro Recall: "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"Macro F1-Score: "
        f"{f1_macro * 100:.2f}%"
    )

    print(
        f"Weighted F1-Score: "
        f"{f1_weighted * 100:.2f}%"
    )

    print(
        f"MCC: "
        f"{mcc:.4f}"
    )

    print(
        f"Average Inference Time: "
        f"{average_inference_time * 1000:.2f} ms/image"
    )

    print("\nClass-wise Metrics:")

    print(
        class_metrics_df.to_string(
            index=False
        )
    )

    print("\nConfusion Matrix:")

    print(cm_df)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print("\nResults saved to:")

    print(metrics_path)
    print(class_metrics_path)
    print(confusion_matrix_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
