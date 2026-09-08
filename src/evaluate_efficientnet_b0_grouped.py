# ============================================================
# EVALUATE EFFICIENTNET-B0
# LEAKAGE-SAFE GROUPED DATASET SPLIT
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import timm

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnet_b0"

NUM_CLASSES = 6

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

VALIDATION_DIR = (
    PROJECT_DIR
    / "data"
    / "grouped_split"
    / "val"
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

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHECKPOINT_PATH = (
    MODELS_DIR
    / "efficientnet_b0_grouped_best.pth"
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
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "EFFICIENTNET-B0 LEAKAGE-SAFE MODEL EVALUATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # LOAD VALIDATION DATASET
    # --------------------------------------------------------

    print(
        "\nLoading validation dataset..."
    )

    validation_dataset = datasets.ImageFolder(

        VALIDATION_DIR,

        transform=validation_transform
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True
    )


    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Classes: "
        f"{validation_dataset.classes}"
    )


    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nLoading model checkpoint..."
    )

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE,

        weights_only=False
    )


    checkpoint_classes = (
        checkpoint["class_names"]
    )


    print(
        f"\nModel classes: "
        f"{checkpoint_classes}"
    )


    # --------------------------------------------------------
    # VERIFY CLASS MAPPING
    # --------------------------------------------------------

    if (
        validation_dataset.classes
        != checkpoint_classes
    ):

        raise ValueError(

            "\nERROR: Class mapping mismatch!"

            "\nDataset classes: "
            f"{validation_dataset.classes}"

            "\nModel classes: "
            f"{checkpoint_classes}"
        )


    print(
        "\nClass mapping verification: PASSED"
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = timm.create_model(

        MODEL_NAME,

        pretrained=False,

        num_classes=NUM_CLASSES
    )


    model.load_state_dict(

        checkpoint["model_state_dict"]
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print(
        "\nEvaluating model..."
    )


    all_predictions = []

    all_labels = []

    all_probabilities = []

    image_paths = []


    with torch.no_grad():

        progress_bar = tqdm(

            validation_loader,

            desc="Evaluation"
        )


        image_index = 0


        for images, labels in progress_bar:


            images = images.to(

                DEVICE,

                non_blocking=True
            )


            labels = labels.to(

                DEVICE,

                non_blocking=True
            )


            with torch.amp.autocast(

                device_type=DEVICE.type,

                enabled=(
                    DEVICE.type == "cuda"
                )
            ):

                outputs = model(
                    images
                )


            probabilities = torch.softmax(

                outputs,

                dim=1
            )


            predictions = torch.argmax(

                probabilities,

                dim=1
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


            batch_size = images.size(0)


            for i in range(batch_size):

                image_paths.append(

                    validation_dataset.samples[
                        image_index
                    ][0]
                )

                image_index += 1


    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )


    # --------------------------------------------------------
    # OVERALL ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(

        all_labels,

        all_predictions
    )


    # --------------------------------------------------------
    # CLASS-WISE METRICS
    # --------------------------------------------------------

    precision, recall, f1_score, support = (

        precision_recall_fscore_support(

            all_labels,

            all_predictions,

            labels=list(
                range(NUM_CLASSES)
            ),

            zero_division=0
        )
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        all_labels,

        all_predictions,

        labels=list(
            range(NUM_CLASSES)
        )
    )


    # --------------------------------------------------------
    # CLASS-WISE DATAFRAME
    # --------------------------------------------------------

    class_results = []


    for index, class_name in enumerate(

        checkpoint_classes

    ):

        class_accuracy = (

            cm[index, index]

            / cm[index].sum()

            * 100
        )


        class_results.append({

            "class":

            class_name,


            "precision":

            precision[index]


            * 100,


            "recall":

            recall[index]


            * 100,


            "f1_score":

            f1_score[index]


            * 100,


            "support":

            support[index],


            "class_accuracy":

            class_accuracy
        })


    class_results_df = pd.DataFrame(

        class_results
    )


    # --------------------------------------------------------
    # SAVE CLASS-WISE RESULTS
    # --------------------------------------------------------

    class_results_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_classwise_results.csv"
    )


    class_results_df.to_csv(

        class_results_path,

        index=False
    )


    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_matrix_df = pd.DataFrame(

        cm,

        index=checkpoint_classes,

        columns=checkpoint_classes
    )


    confusion_matrix_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_confusion_matrix.csv"
    )


    confusion_matrix_df.to_csv(

        confusion_matrix_path
    )


    # --------------------------------------------------------
    # DETAILED PREDICTIONS
    # --------------------------------------------------------

    detailed_predictions = []


    for index in range(

        len(all_labels)

    ):


        true_class = (

            checkpoint_classes[
                all_labels[index]
            ]
        )


        predicted_class = (

            checkpoint_classes[
                all_predictions[index]
            ]
        )


        confidence = (

            all_probabilities[
                index,
                all_predictions[index]
            ]
        )


        detailed_predictions.append({

            "image_path":

            image_paths[index],


            "image_name":

            Path(
                image_paths[index]
            ).name,


            "true_class":

            true_class,


            "predicted_class":

            predicted_class,


            "confidence_percent":

            confidence
            * 100,


            "correct_prediction":

            all_labels[index]
            ==
            all_predictions[index]
        })


    predictions_df = pd.DataFrame(

        detailed_predictions
    )


    predictions_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_predictions.csv"
    )


    predictions_df.to_csv(

        predictions_path,

        index=False
    )


    # --------------------------------------------------------
    # SAVE INCORRECT PREDICTIONS
    # --------------------------------------------------------

    incorrect_predictions_df = (

        predictions_df[

            predictions_df[
                "correct_prediction"
            ]
            == False

        ]
    )


    incorrect_predictions_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_incorrect_predictions.csv"
    )


    incorrect_predictions_df.to_csv(

        incorrect_predictions_path,

        index=False
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        all_labels,

        all_predictions,

        target_names=checkpoint_classes,

        digits=4,

        zero_division=0
    )


    report_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_classification_report.txt"
    )


    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            report
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "EVALUATION RESULTS"
    )

    print("=" * 70)


    print(

        f"\nTotal Validation Images: "
        f"{len(all_labels)}"
    )


    print(

        f"Correct Predictions: "
        f"{np.sum(all_labels == all_predictions)}"
    )


    print(

        f"Incorrect Predictions: "
        f"{np.sum(all_labels != all_predictions)}"
    )


    print(

        f"\nOverall Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    print("\n" + "-" * 70)

    print(
        "CLASS-WISE PERFORMANCE"
    )

    print("-" * 70)


    print(

        class_results_df.to_string(

            index=False
        )
    )


    print("\n" + "-" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("-" * 70)


    print(
        confusion_matrix_df
    )


    print("\n" + "=" * 70)

    print(
        "SAVED EVALUATION FILES"
    )

    print("=" * 70)


    print(

        "\n1. Class-wise Results:"
    )

    print(
        class_results_path
    )


    print(

        "\n2. Confusion Matrix:"
    )

    print(
        confusion_matrix_path
    )


    print(

        "\n3. Detailed Predictions:"
    )

    print(
        predictions_path
    )


    print(

        "\n4. Incorrect Predictions:"
    )

    print(
        incorrect_predictions_path
    )


    print(

        "\n5. Classification Report:"
    )

    print(
        report_path
    )


    print("\n" + "=" * 70)

    print(
        "EVALUATION COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()