# ============================================================
# PREDICTION CONFIDENCE ANALYSIS
# EFFICIENTNET-B0
# SIX-CLASS NAIL CONDITION CLASSIFICATION
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np

import torch
import timm

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

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
# LOAD DATASET
# ============================================================

def load_validation_dataset():

    dataset = datasets.ImageFolder(

        DATA_DIR / "val",

        transform=validation_transform,

    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

    )


    return dataset, loader


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


    return model, class_names


# ============================================================
# ANALYZE PREDICTIONS
# ============================================================

def analyze_predictions(
    model,
    dataset,
    loader,
    class_names,
):

    results = []


    sample_index = 0


    with torch.no_grad():

        progress_bar = tqdm(

            loader,

            desc="Analyzing predictions",

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


            probabilities = torch.softmax(

                outputs,

                dim=1,

            )


            top_probabilities, top_indices = torch.topk(

                probabilities,

                k=3,

                dim=1,

            )


            for batch_index in range(

                images.size(0)

            ):


                image_path, true_label = (

                    dataset.samples[
                        sample_index
                    ]

                )


                predicted_index = (

                    top_indices[
                        batch_index,
                        0
                    ]
                    .item()
                )


                confidence = (

                    top_probabilities[
                        batch_index,
                        0
                    ]
                    .item()
                    * 100
                )


                top2_index = (

                    top_indices[
                        batch_index,
                        1
                    ]
                    .item()
                )


                top2_probability = (

                    top_probabilities[
                        batch_index,
                        1
                    ]
                    .item()
                    * 100
                )


                top3_index = (

                    top_indices[
                        batch_index,
                        2
                    ]
                    .item()
                )


                top3_probability = (

                    top_probabilities[
                        batch_index,
                        2
                    ]
                    .item()
                    * 100
                )


                results.append({

                    "image_path":
                        str(image_path),

                    "image_name":
                        Path(
                            image_path
                        ).name,

                    "true_class":
                        class_names[
                            true_label
                        ],

                    "predicted_class":
                        class_names[
                            predicted_index
                        ],

                    "correct_prediction":
                        predicted_index
                        == true_label,

                    "confidence_percent":
                        confidence,

                    "top_2_class":
                        class_names[
                            top2_index
                        ],

                    "top_2_probability":
                        top2_probability,

                    "top_3_class":
                        class_names[
                            top3_index
                        ],

                    "top_3_probability":
                        top3_probability,

                })


                sample_index += 1


    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "EFFICIENTNET-B0 PREDICTION CONFIDENCE ANALYSIS"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print(
            "\nERROR: Model checkpoint not found."
        )

        print(
            MODEL_PATH
        )

        return


    if not (
        DATA_DIR / "val"
    ).exists():

        print(
            "\nERROR: Validation dataset not found."
        )

        return


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print()

    print(
        f"Device: {DEVICE}"
    )


    if DEVICE.type == "cuda":

        print(

            "GPU: "
            + torch.cuda.get_device_name(0)

        )


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print(
        "\nLoading validation dataset..."
    )


    dataset, loader = (

        load_validation_dataset()

    )


    print(

        f"Validation images: "
        f"{len(dataset)}"

    )


    print(
        f"Classes: {dataset.classes}"
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, class_names = (

        load_model()

    )


    print(
        f"\nModel classes: "
        f"{class_names}"
    )


    # --------------------------------------------------------
    # VERIFY CLASS MAPPING
    # --------------------------------------------------------

    if dataset.classes != class_names:

        print()

        print(
            "ERROR: Dataset and model "
            "class mappings do not match!"
        )

        print(
            f"Dataset: {dataset.classes}"
        )

        print(
            f"Model: {class_names}"
        )

        return


    print(
        "\nClass mapping verification: PASSED"
    )


    # --------------------------------------------------------
    # ANALYZE PREDICTIONS
    # --------------------------------------------------------

    print()

    print(
        "Analyzing predictions..."
    )


    results_df = analyze_predictions(

        model,

        dataset,

        loader,

        class_names,

    )


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    METRICS_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    # --------------------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------------------

    detailed_path = (

        METRICS_DIR

        / "efficientnet_b0_prediction_confidence.csv"

    )


    results_df.to_csv(

        detailed_path,

        index=False,

    )


    # --------------------------------------------------------
    # CALCULATE SUMMARY
    # --------------------------------------------------------

    total_images = len(
        results_df
    )


    correct_predictions = int(

        results_df[
            "correct_prediction"
        ].sum()

    )


    incorrect_predictions = (

        total_images
        - correct_predictions

    )


    average_confidence = (

        results_df[
            "confidence_percent"
        ].mean()

    )


    minimum_confidence = (

        results_df[
            "confidence_percent"
        ].min()

    )


    maximum_confidence = (

        results_df[
            "confidence_percent"
        ].max()

    )


    # --------------------------------------------------------
    # CLASS-WISE CONFIDENCE
    # --------------------------------------------------------

    class_confidence_df = (

        results_df

        .groupby(
            "true_class"
        )

        .agg(

            total_images=(
                "image_name",
                "count"
            ),

            average_confidence=(
                "confidence_percent",
                "mean"
            ),

            minimum_confidence=(
                "confidence_percent",
                "min"
            ),

            maximum_confidence=(
                "confidence_percent",
                "max"
            ),

        )

        .reset_index()

    )


    class_confidence_path = (

        METRICS_DIR

        / "efficientnet_b0_class_confidence.csv"

    )


    class_confidence_df.to_csv(

        class_confidence_path,

        index=False,

    )


    # --------------------------------------------------------
    # LOWEST CONFIDENCE PREDICTIONS
    # --------------------------------------------------------

    lowest_confidence_df = (

        results_df

        .sort_values(
            by="confidence_percent",
            ascending=True,
        )

        .head(10)

    )


    lowest_confidence_path = (

        METRICS_DIR

        / "efficientnet_b0_lowest_confidence_predictions.csv"

    )


    lowest_confidence_df.to_csv(

        lowest_confidence_path,

        index=False,

    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "CONFIDENCE ANALYSIS RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal Validation Images: "
        f"{total_images}"
    )


    print(
        f"Correct Predictions: "
        f"{correct_predictions}"
    )


    print(
        f"Incorrect Predictions: "
        f"{incorrect_predictions}"
    )


    print(
        f"\nAverage Confidence: "
        f"{average_confidence:.2f}%"
    )


    print(
        f"Minimum Confidence: "
        f"{minimum_confidence:.2f}%"
    )


    print(
        f"Maximum Confidence: "
        f"{maximum_confidence:.2f}%"
    )


    print()

    print(
        "-" * 70
    )

    print(
        "CLASS-WISE CONFIDENCE"
    )

    print(
        "-" * 70
    )


    print(
        class_confidence_df.to_string(

            index=False

        )

    )


    print()

    print(
        "-" * 70
    )

    print(
        "10 LOWEST CONFIDENCE PREDICTIONS"
    )

    print(
        "-" * 70
    )


    display_columns = [

        "image_name",

        "true_class",

        "predicted_class",

        "confidence_percent",

        "correct_prediction",

        "top_2_class",

        "top_2_probability",

    ]


    print(

        lowest_confidence_df[
            display_columns
        ].to_string(

            index=False

        )

    )


    # --------------------------------------------------------
    # SAVED FILES
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "SAVED CONFIDENCE ANALYSIS FILES"
    )

    print(
        "=" * 70
    )


    print(
        "\n1. Detailed Prediction Results:"
    )

    print(
        detailed_path
    )


    print(
        "\n2. Class-wise Confidence:"
    )

    print(
        class_confidence_path
    )


    print(
        "\n3. Lowest Confidence Predictions:"
    )

    print(
        lowest_confidence_path
    )


    print()

    print(
        "=" * 70
    )

    print(
        "CONFIDENCE ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()