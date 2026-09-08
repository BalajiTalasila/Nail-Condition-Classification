# ============================================================
# EVALUATION SCRIPT
# EFFICIENTNET-B0
# SIX-CLASS NAIL CONDITION CLASSIFICATION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnet_b0"

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent.parent


DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)


MODELS_DIR = (
    RESULTS_DIR
    / "models"
)


METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)


MODEL_PATH = (
    MODELS_DIR
    / "efficientnet_b0_best.pth"
)


METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

validation_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],

        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),

])


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

def load_validation_dataset():

    validation_dataset = datasets.ImageFolder(

        DATA_DIR / "val",

        transform=validation_transform,
    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )


    return (

        validation_dataset,

        validation_loader,

    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading model checkpoint..."
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE,
    )


    class_names = checkpoint[
        "class_names"
    ]


    model_name = checkpoint[
        "model_name"
    ]


    model = timm.create_model(

        model_name,

        pretrained=False,

        num_classes=len(
            class_names
        ),
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    return (

        model,

        checkpoint,

    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(

    model,

    data_loader,

):

    predicted_labels = []

    true_labels = []


    with torch.no_grad():

        progress_bar = tqdm(

            data_loader,

            desc="Evaluating",
        )


        for images, labels in progress_bar:


            images = images.to(

                DEVICE,

                non_blocking=True,
            )


            labels = labels.to(

                DEVICE,

                non_blocking=True,
            )


            with torch.amp.autocast(

                device_type=DEVICE.type,

                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):


                outputs = model(
                    images
                )


            predictions = torch.argmax(

                outputs,

                dim=1,
            )


            predicted_labels.extend(

                predictions.cpu().numpy()
            )


            true_labels.extend(

                labels.cpu().numpy()
            )


    true_labels = np.array(
        true_labels
    )


    predicted_labels = np.array(
        predicted_labels
    )


    accuracy = accuracy_score(

        true_labels,

        predicted_labels,

    ) * 100


    return (

        true_labels,

        predicted_labels,

        accuracy,

    )


# ============================================================
# SAVE CONFUSION MATRIX VISUALIZATION
# ============================================================

def save_confusion_matrix_plot(

    cm,

    class_names,

    output_path,

    normalized=False,

):

    plt.figure(

        figsize=(
            10,
            8,
        )
    )


    if normalized:

        cm_display = (

            cm.astype(
                float
            )

            /

            cm.sum(
                axis=1,
                keepdims=True,
            )
        )


        cm_display = np.nan_to_num(
            cm_display
        )


        annotation_format = ".2f"

        title = (
            "Normalized Confusion Matrix"
        )


    else:

        cm_display = cm

        annotation_format = "d"

        title = (
            "Confusion Matrix"
        )


    sns.heatmap(

        cm_display,

        annot=True,

        fmt=annotation_format,

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names,

        cbar=True,

    )


    plt.title(
        title
    )


    plt.xlabel(
        "Predicted Label"
    )


    plt.ylabel(
        "True Label"
    )


    plt.xticks(
        rotation=45,
        ha="right",
    )


    plt.yticks(
        rotation=0,
    )


    plt.tight_layout()


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight",
    )


    plt.close()


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )


    print(
        "EFFICIENTNET-B0 MODEL EVALUATION"
    )


    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # DEVICE INFORMATION
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )


    if DEVICE.type == "cuda":

        print(

            "GPU: "

            + torch.cuda.get_device_name(
                0
            )
        )


    # --------------------------------------------------------
    # CHECK MODEL FILE
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "\nERROR: Model file not found!"
        )


        print(
            f"Expected location:\n{MODEL_PATH}"
        )


        return


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print(
        "\nLoading validation dataset..."
    )


    (

        validation_dataset,

        validation_loader,

    ) = load_validation_dataset()


    print(

        f"Validation images: "
        f"{len(validation_dataset)}"
    )


    print(
        "\nDataset class mapping:"
    )


    for index, class_name in enumerate(

        validation_dataset.classes

    ):

        print(

            f"{index}: "
            f"{class_name}"
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    (

        model,

        checkpoint,

    ) = load_model()


    checkpoint_class_names = checkpoint[
        "class_names"
    ]


    print(

        f"\nModel: "
        f"{checkpoint['model_name']}"
    )


    print(
        "\nCheckpoint class mapping:"
    )


    for index, class_name in enumerate(

        checkpoint_class_names

    ):

        print(

            f"{index}: "
            f"{class_name}"
        )


    # --------------------------------------------------------
    # VERIFY CLASS ORDER
    # --------------------------------------------------------

    if (

        validation_dataset.classes
        !=
        checkpoint_class_names

    ):

        print(
            "\nWARNING: Class order mismatch detected!"
        )


        print(
            "Dataset class order:"
        )


        print(
            validation_dataset.classes
        )


        print(
            "\nCheckpoint class order:"
        )


        print(
            checkpoint_class_names
        )


        print(
            "\nEvaluation stopped because "
            "class mappings do not match."
        )


        return


    print(
        "\nClass mapping verification: PASSED"
    )


    # --------------------------------------------------------
    # CHECKPOINT INFORMATION
    # --------------------------------------------------------

    print(

        f"\nCheckpoint epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )


    if (
        "validation_loss"
        in checkpoint
    ):

        print(

            f"Best validation loss: "
            f"{checkpoint['validation_loss']:.4f}"
        )


    if (
        "validation_accuracy"
        in checkpoint
    ):

        print(

            f"Validation accuracy "
            f"at checkpoint: "
            f"{checkpoint['validation_accuracy']:.2f}%"
        )


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print(
        "\nStarting evaluation...\n"
    )


    (

        true_labels,

        predicted_labels,

        accuracy,

    ) = evaluate_model(

        model,

        validation_loader,

    )


    # --------------------------------------------------------
    # RESULTS HEADER
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )


    print(
        "EVALUATION RESULTS"
    )


    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # OVERALL ACCURACY
    # --------------------------------------------------------

    print(

        f"\nOverall Accuracy: "
        f"{accuracy:.2f}%"
    )


    class_names = (
        validation_dataset.classes
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        true_labels,

        predicted_labels,

        labels=range(
            len(class_names)
        ),

        target_names=class_names,

        digits=4,

        zero_division=0,
    )


    print(
        "\nClassification Report:\n"
    )


    print(
        report
    )


    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_dict = classification_report(

        true_labels,

        predicted_labels,

        labels=range(
            len(class_names)
        ),

        target_names=class_names,

        output_dict=True,

        zero_division=0,
    )


    report_df = pd.DataFrame(
        report_dict
    ).transpose()


    report_path = (

        METRICS_DIR

        / "efficientnet_b0_classification_report.csv"
    )


    report_df.to_csv(
        report_path
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        true_labels,

        predicted_labels,

        labels=range(
            len(class_names)
        ),
    )


    print(
        "\nConfusion Matrix:\n"
    )


    print(
        cm
    )


    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX CSV
    # --------------------------------------------------------

    confusion_matrix_df = pd.DataFrame(

        cm,

        index=class_names,

        columns=class_names,
    )


    confusion_matrix_path = (

        METRICS_DIR

        / "efficientnet_b0_confusion_matrix.csv"
    )


    confusion_matrix_df.to_csv(
        confusion_matrix_path
    )


    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX IMAGE
    # --------------------------------------------------------

    confusion_matrix_image_path = (

        METRICS_DIR

        / "efficientnet_b0_confusion_matrix.png"
    )


    save_confusion_matrix_plot(

        cm,

        class_names,

        confusion_matrix_image_path,

        normalized=False,

    )


    # --------------------------------------------------------
    # SAVE NORMALIZED CONFUSION MATRIX
    # --------------------------------------------------------

    normalized_confusion_matrix_path = (

        METRICS_DIR

        / "efficientnet_b0_normalized_confusion_matrix.png"
    )


    save_confusion_matrix_plot(

        cm,

        class_names,

        normalized_confusion_matrix_path,

        normalized=True,

    )


    # --------------------------------------------------------
    # CLASS-WISE METRICS
    # --------------------------------------------------------

    (

        precision,

        recall,

        f1,

        support,

    ) = precision_recall_fscore_support(

        true_labels,

        predicted_labels,

        labels=range(
            len(class_names)
        ),

        zero_division=0,

    )


    classwise_metrics_df = pd.DataFrame({

        "class": class_names,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "support": support,

    })


    classwise_metrics_path = (

        METRICS_DIR

        / "efficientnet_b0_classwise_metrics.csv"
    )


    classwise_metrics_df.to_csv(

        classwise_metrics_path,

        index=False,

    )


    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    weighted_precision = precision_score(

        true_labels,

        predicted_labels,

        average="weighted",

        zero_division=0,

    )


    weighted_recall = recall_score(

        true_labels,

        predicted_labels,

        average="weighted",

        zero_division=0,

    )


    weighted_f1 = f1_score(

        true_labels,

        predicted_labels,

        average="weighted",

        zero_division=0,

    )


    macro_precision = precision_score(

        true_labels,

        predicted_labels,

        average="macro",

        zero_division=0,

    )


    macro_recall = recall_score(

        true_labels,

        predicted_labels,

        average="macro",

        zero_division=0,

    )


    macro_f1 = f1_score(

        true_labels,

        predicted_labels,

        average="macro",

        zero_division=0,

    )


    overall_metrics_df = pd.DataFrame({

        "metric": [

            "accuracy_percent",

            "weighted_precision",

            "weighted_recall",

            "weighted_f1_score",

            "macro_precision",

            "macro_recall",

            "macro_f1_score",

        ],

        "value": [

            accuracy,

            weighted_precision,

            weighted_recall,

            weighted_f1,

            macro_precision,

            macro_recall,

            macro_f1,

        ],

    })


    overall_metrics_path = (

        METRICS_DIR

        / "efficientnet_b0_overall_metrics.csv"
    )


    overall_metrics_df.to_csv(

        overall_metrics_path,

        index=False,

    )


    # --------------------------------------------------------
    # PRINT SAVED FILES
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )


    print(
        "SAVED EVALUATION FILES"
    )


    print(
        "-" * 70
    )


    print(

        f"\n1. Classification Report:\n"
        f"{report_path}"
    )


    print(

        f"\n2. Confusion Matrix CSV:\n"
        f"{confusion_matrix_path}"
    )


    print(

        f"\n3. Confusion Matrix Image:\n"
        f"{confusion_matrix_image_path}"
    )


    print(

        f"\n4. Normalized Confusion Matrix:\n"
        f"{normalized_confusion_matrix_path}"
    )


    print(

        f"\n5. Class-wise Metrics:\n"
        f"{classwise_metrics_path}"
    )


    print(

        f"\n6. Overall Metrics:\n"
        f"{overall_metrics_path}"
    )


    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )


    print(
        "EVALUATION COMPLETED SUCCESSFULLY"
    )


    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()