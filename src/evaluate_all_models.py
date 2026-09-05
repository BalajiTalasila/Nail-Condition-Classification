import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    cohen_kappa_score,
    matthews_corrcoef
)

from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_DIR)
)

from src.proposed_attention_efficientnet import (
    AttentionEfficientNetB0
)


TEST_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "test"
)

MODELS_DIR = (
    PROJECT_DIR
    / "results"
    / "models"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

PREDICTIONS_DIR = (
    RESULTS_DIR
    / "predictions"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


for directory in [
    METRICS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0


MODELS = {

    "EfficientNet-B0": {

        "checkpoint":
        "efficientnet_b0_best.pth",

        "architecture":
        "efficientnet_b0"

    },

    "DenseNet121": {

        "checkpoint":
        "densenet121_best.pth",

        "architecture":
        "densenet121"

    },

    "ConvNeXtV2-Tiny": {

        "checkpoint":
        "convnextv2_tiny_best.pth",

        "architecture":
        "convnextv2_tiny"

    },

    "Proposed Attention-EfficientNet-B0": {

        "checkpoint":
        "proposed_attention_efficientnet_b0_best.pth",

        "architecture":
        "proposed"

    }
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )

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
# CREATE TEST LOADER
# ============================================================

def create_test_loader():

    if not TEST_DIR.exists():

        raise FileNotFoundError(

            f"Test directory not found:\n"
            f"{TEST_DIR}"
        )

    test_dataset = datasets.ImageFolder(

        TEST_DIR,

        transform=test_transform
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )

    return (

        test_dataset,

        test_loader
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(

    model_name,

    model_info,

    class_names

):

    checkpoint_path = (

        MODELS_DIR

        / model_info["checkpoint"]
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print(

        f"\nLoading checkpoint:\n"
        f"{checkpoint_path.name}"
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE
    )

    saved_classes = checkpoint.get(

        "class_names",

        None
    )

    if saved_classes is not None:

        if list(saved_classes) != list(class_names):

            raise ValueError(

                "\nCLASS ORDER MISMATCH!\n"
                f"Dataset classes: {class_names}\n"
                f"Checkpoint classes: {saved_classes}"
            )

    architecture = (

        model_info["architecture"]
    )

    # --------------------------------------------------------
    # STANDARD TIMM MODELS
    # --------------------------------------------------------

    if architecture != "proposed":

        model = timm.create_model(

            architecture,

            pretrained=False,

            num_classes=len(class_names)
        )

    # --------------------------------------------------------
    # PROPOSED MODEL
    # --------------------------------------------------------

    else:

        model = AttentionEfficientNetB0(

            num_classes=len(class_names),

            dropout=0.30,

            pretrained=False
        )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    state_dict = checkpoint.get(

        "model_state_dict",

        checkpoint
    )

    load_result = model.load_state_dict(

        state_dict,

        strict=True
    )

    print(

        "✓ Model weights loaded successfully"
    )

    if (

        len(load_result.missing_keys) > 0

        or

        len(load_result.unexpected_keys) > 0

    ):

        print(

            "WARNING: State dictionary mismatch detected."
        )

        print(

            "Missing keys:",

            load_result.missing_keys
        )

        print(

            "Unexpected keys:",

            load_result.unexpected_keys
        )

    model = model.to(

        DEVICE
    )

    model.eval()

    return (

        model,

        checkpoint
    )


# ============================================================
# RUN PREDICTION
# ============================================================

def predict_model(

    model,

    test_loader,

    test_dataset

):

    all_predictions = []

    all_probabilities = []

    all_labels = []

    all_filepaths = []

    print(

        "\nRunning inference..."
    )

    sample_index = 0

    with torch.no_grad():

        for images, labels in tqdm(

            test_loader,

            desc="Evaluating",

            unit="batch"
        ):

            images = images.to(

                DEVICE,

                non_blocking=True
            )

            labels = labels.to(

                DEVICE,

                non_blocking=True
            )

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

            batch_size = labels.size(

                0
            )

            batch_paths = [

                test_dataset.samples[
                    sample_index + i
                ][0]

                for i in range(batch_size)
            ]

            sample_index += batch_size

            all_predictions.extend(

                predictions.cpu().numpy()
            )

            all_probabilities.extend(

                probabilities.cpu().numpy()
            )

            all_labels.extend(

                labels.cpu().numpy()
            )

            all_filepaths.extend(

                batch_paths
            )

    return (

        np.array(all_labels),

        np.array(all_predictions),

        np.array(all_probabilities),

        all_filepaths
    )


# ============================================================
# CALCULATE SPECIFICITY
# ============================================================

def calculate_specificity(

    confusion_matrix_array

):

    number_of_classes = (

        confusion_matrix_array.shape[0]
    )

    total = (

        confusion_matrix_array.sum()
    )

    specificities = []

    for class_index in range(

        number_of_classes
    ):

        true_positive = (

            confusion_matrix_array[
                class_index,
                class_index
            ]
        )

        false_positive = (

            confusion_matrix_array[
                :,
                class_index
            ].sum()

            - true_positive
        )

        false_negative = (

            confusion_matrix_array[
                class_index,
                :
            ].sum()

            - true_positive
        )

        true_negative = (

            total

            - true_positive

            - false_positive

            - false_negative
        )

        denominator = (

            true_negative

            + false_positive
        )

        if denominator == 0:

            specificity = 0.0

        else:

            specificity = (

                true_negative

                / denominator
            )

        specificities.append(

            specificity
        )

    return np.array(

        specificities
    )


# ============================================================
# CREATE CONFUSION MATRIX FIGURE
# ============================================================

def save_confusion_matrix(

    cm,

    class_names,

    model_name,

    output_path

):

    figure_size = max(

        8,

        len(class_names) * 1.5
    )

    fig, ax = plt.subplots(

        figsize=(
            figure_size,
            figure_size
        )
    )

    image = ax.imshow(

        cm,

        interpolation="nearest",

        cmap="Blues"
    )

    fig.colorbar(

        image,

        ax=ax
    )

    ax.set(

        xticks=np.arange(

            len(class_names)
        ),

        yticks=np.arange(

            len(class_names)
        ),

        xticklabels=class_names,

        yticklabels=class_names,

        ylabel="True Label",

        xlabel="Predicted Label",

        title=f"{model_name} - Confusion Matrix"
    )

    plt.setp(

        ax.get_xticklabels(),

        rotation=45,

        ha="right"
    )

    threshold = (

        cm.max() / 2.0

        if cm.max() > 0

        else 0
    )

    for i in range(

        cm.shape[0]
    ):

        for j in range(

            cm.shape[1]
        ):

            ax.text(

                j,

                i,

                format(
                    cm[i, j],
                    "d"
                ),

                ha="center",

                va="center",

                color=(
                    "white"

                    if cm[i, j] > threshold

                    else "black"
                )
            )

    fig.tight_layout()

    fig.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close(

        fig
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(

    model_name,

    model_info,

    test_loader,

    test_dataset,

    class_names

):

    print(

        "\n" + "=" * 70
    )

    print(

        f"EVALUATING: {model_name}"
    )

    print(

        "=" * 70
    )

    model, checkpoint = load_model(

        model_name,

        model_info,

        class_names
    )

    true_labels, predictions, probabilities, filepaths = (

        predict_model(

            model,

            test_loader,

            test_dataset
        )
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        true_labels,

        predictions
    )

    precision_macro, recall_macro, f1_macro, _ = (

        precision_recall_fscore_support(

            true_labels,

            predictions,

            average="macro",

            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (

        precision_recall_fscore_support(

            true_labels,

            predictions,

            average="weighted",

            zero_division=0
        )
    )

    kappa = cohen_kappa_score(

        true_labels,

        predictions
    )

    mcc = matthews_corrcoef(

        true_labels,

        predictions
    )

    cm = confusion_matrix(

        true_labels,

        predictions,

        labels=np.arange(
            len(class_names)
        )
    )

    # --------------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------------

    precision, recall, f1, support = (

        precision_recall_fscore_support(

            true_labels,

            predictions,

            labels=np.arange(
                len(class_names)
            ),

            average=None,

            zero_division=0
        )
    )

    specificity = calculate_specificity(

        cm
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(

        "\nOVERALL TEST RESULTS"
    )

    print(

        "-" * 70
    )

    print(

        f"Test Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print(

        f"Macro Precision: "
        f"{precision_macro * 100:.4f}%"
    )

    print(

        f"Macro Recall: "
        f"{recall_macro * 100:.4f}%"
    )

    print(

        f"Macro F1-Score: "
        f"{f1_macro * 100:.4f}%"
    )

    print(

        f"Weighted F1-Score: "
        f"{f1_weighted * 100:.4f}%"
    )

    print(

        f"Cohen's Kappa: "
        f"{kappa:.6f}"
    )

    print(

        f"Matthews Correlation Coefficient: "
        f"{mcc:.6f}"
    )

    print(

        "\nPER-CLASS RESULTS"
    )

    print(

        "-" * 70
    )

    per_class_dataframe = pd.DataFrame({

        "class":

        class_names,

        "precision":

        precision,

        "recall_sensitivity":

        recall,

        "specificity":

        specificity,

        "f1_score":

        f1,

        "support":

        support
    })

    print(

        per_class_dataframe.to_string(

            index=False
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        true_labels,

        predictions,

        target_names=class_names,

        zero_division=0,

        output_dict=True
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    safe_name = (

        model_name

        .lower()

        .replace(
            " ",
            "_"
        )

        .replace(
            "-",
            "_"
        )
    )

    predictions_dataframe = pd.DataFrame({

        "filepath":

        filepaths,

        "true_label_index":

        true_labels,

        "true_class":

        [

            class_names[index]

            for index in true_labels
        ],

        "predicted_label_index":

        predictions,

        "predicted_class":

        [

            class_names[index]

            for index in predictions
        ],

        "confidence":

        probabilities.max(
            axis=1
        ),

        "correct":

        true_labels == predictions
    })

    for class_index, class_name in enumerate(

        class_names
    ):

        predictions_dataframe[

            f"probability_{class_name}"

        ] = probabilities[

            :,

            class_index
        ]

    predictions_path = (

        PREDICTIONS_DIR

        / f"{safe_name}_predictions.csv"
    )

    predictions_dataframe.to_csv(

        predictions_path,

        index=False
    )

    # --------------------------------------------------------
    # SAVE PER-CLASS METRICS
    # --------------------------------------------------------

    per_class_path = (

        METRICS_DIR

        / f"{safe_name}_per_class_metrics.csv"
    )

    per_class_dataframe.to_csv(

        per_class_path,

        index=False
    )

    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_path = (

        METRICS_DIR

        / f"{safe_name}_classification_report.json"
    )

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            report,

            file,

            indent=4
        )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_matrix_dataframe = pd.DataFrame(

        cm,

        index=class_names,

        columns=class_names
    )

    confusion_matrix_csv_path = (

        METRICS_DIR

        / f"{safe_name}_confusion_matrix.csv"
    )

    confusion_matrix_dataframe.to_csv(

        confusion_matrix_csv_path
    )

    confusion_matrix_figure_path = (

        FIGURES_DIR

        / f"{safe_name}_confusion_matrix.png"
    )

    save_confusion_matrix(

        cm,

        class_names,

        model_name,

        confusion_matrix_figure_path
    )

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    model_summary = {

        "model":

        model_name,

        "checkpoint":

        model_info["checkpoint"],

        "checkpoint_epoch":

        checkpoint.get(
            "epoch",
            None
        ),

        "validation_accuracy":

        checkpoint.get(
            "validation_accuracy",
            None
        ),

        "validation_loss":

        checkpoint.get(
            "validation_loss",
            None
        ),

        "test_accuracy":

        accuracy,

        "macro_precision":

        precision_macro,

        "macro_recall_sensitivity":

        recall_macro,

        "macro_f1_score":

        f1_macro,

        "weighted_precision":

        precision_weighted,

        "weighted_recall":

        recall_weighted,

        "weighted_f1_score":

        f1_weighted,

        "cohens_kappa":

        kappa,

        "matthews_correlation_coefficient":

        mcc,

        "test_samples":

        len(true_labels)
    }

    # --------------------------------------------------------
    # CLEAN GPU MEMORY
    # --------------------------------------------------------

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return (

        model_summary,

        per_class_dataframe
    )


# ============================================================
# SAVE MODEL COMPARISON FIGURE
# ============================================================

def save_model_comparison_figure(

    comparison_dataframe,

    output_path

):

    metrics = [

        "test_accuracy",

        "macro_precision",

        "macro_recall_sensitivity",

        "macro_f1_score",

        "weighted_f1_score"
    ]

    plot_dataframe = (

        comparison_dataframe[
            [
                "model"
            ]

            + metrics
        ]

        .set_index(
            "model"
        )

        * 100
    )

    ax = plot_dataframe.plot(

        kind="bar",

        figsize=(
            14,
            7
        )
    )

    ax.set_ylabel(

        "Score (%)"
    )

    ax.set_xlabel(

        "Model"
    )

    ax.set_title(

        "Test Performance Comparison"
    )

    ax.set_ylim(

        0,

        105
    )

    plt.xticks(

        rotation=20,

        ha="right"
    )

    plt.tight_layout()

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(

        RANDOM_SEED
    )

    print(

        "\n" + "=" * 70
    )

    print(

        "SIX-CLASS NAIL DISEASE MODEL TEST EVALUATION"
    )

    print(

        "=" * 70
    )

    print(

        f"\nDevice: "
        f"{DEVICE}"
    )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_dataset, test_loader = (

        create_test_loader()
    )

    class_names = (

        test_dataset.classes
    )

    print(

        f"\nTest dataset: "
        f"{TEST_DIR}"
    )

    print(

        f"Test images: "
        f"{len(test_dataset)}"
    )

    print(

        f"Classes: "
        f"{len(class_names)}"
    )

    print(

        "\nClass names:"
    )

    for index, class_name in enumerate(

        class_names
    ):

        print(

            f"{index}: "
            f"{class_name}"
        )

    if len(test_dataset) != 576:

        print(

            "\nWARNING: Expected 576 test images "
            f"but found {len(test_dataset)}."
        )

    # --------------------------------------------------------
    # EVALUATE ALL MODELS
    # --------------------------------------------------------

    model_results = []

    all_per_class_results = []

    for model_name, model_info in MODELS.items():

        model_summary, per_class_dataframe = (

            evaluate_model(

                model_name,

                model_info,

                test_loader,

                test_dataset,

                class_names
            )
        )

        model_results.append(

            model_summary
        )

        per_class_dataframe = (

            per_class_dataframe.copy()
        )

        per_class_dataframe.insert(

            0,

            "model",

            model_name
        )

        all_per_class_results.append(

            per_class_dataframe
        )

    # --------------------------------------------------------
    # SAVE MODEL COMPARISON
    # --------------------------------------------------------

    comparison_dataframe = pd.DataFrame(

        model_results
    )

    comparison_dataframe = (

        comparison_dataframe.sort_values(

            by="test_accuracy",

            ascending=False
        )
    )

    comparison_path = (

        METRICS_DIR

        / "model_test_comparison.csv"
    )

    comparison_dataframe.to_csv(

        comparison_path,

        index=False
    )

    # --------------------------------------------------------
    # SAVE PER-CLASS COMPARISON
    # --------------------------------------------------------

    per_class_comparison = pd.concat(

        all_per_class_results,

        ignore_index=True
    )

    per_class_comparison_path = (

        METRICS_DIR

        / "per_class_model_comparison.csv"
    )

    per_class_comparison.to_csv(

        per_class_comparison_path,

        index=False
    )

    # --------------------------------------------------------
    # SAVE COMPARISON FIGURE
    # --------------------------------------------------------

    comparison_figure_path = (

        FIGURES_DIR

        / "model_test_performance_comparison.png"
    )

    save_model_comparison_figure(

        comparison_dataframe,

        comparison_figure_path
    )

    # --------------------------------------------------------
    # PRINT FINAL RANKING
    # --------------------------------------------------------

    print(

        "\n" + "=" * 70
    )

    print(

        "FINAL MODEL RANKING - TEST DATASET"
    )

    print(

        "=" * 70
    )

    ranking_columns = [

        "model",

        "test_accuracy",

        "macro_precision",

        "macro_recall_sensitivity",

        "macro_f1_score",

        "weighted_f1_score",

        "cohens_kappa",

        "matthews_correlation_coefficient"
    ]

    ranking_dataframe = (

        comparison_dataframe[
            ranking_columns
        ].copy()
    )

    for column in [

        "test_accuracy",

        "macro_precision",

        "macro_recall_sensitivity",

        "macro_f1_score",

        "weighted_f1_score"
    ]:

        ranking_dataframe[column] *= 100

    print(

        ranking_dataframe.to_string(

            index=False
        )
    )

    best_model = (

        comparison_dataframe.iloc[0]
    )

    print(

        "\nBEST MODEL BASED ON TEST ACCURACY:"
    )

    print(

        f"{best_model['model']}"
    )

    print(

        f"Test Accuracy: "
        f"{best_model['test_accuracy'] * 100:.4f}%"
    )

    # --------------------------------------------------------
    # OUTPUT LOCATIONS
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

    print(

        "\nGenerated directories:"
    )

    print(

        METRICS_DIR
    )

    print(

        PREDICTIONS_DIR
    )

    print(

        FIGURES_DIR
    )


if __name__ == "__main__":

    main()
