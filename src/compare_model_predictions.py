import pandas as pd

from pathlib import Path

from torchvision import datasets


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
)


METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)


# ============================================================
# INPUT FILES
# ============================================================

EFFICIENTNET_PREDICTIONS_FILE = (
    METRICS_DIR
    / "efficientnet_b0_test_predictions.csv"
)


PROPOSED_PREDICTIONS_FILE = (
    METRICS_DIR
    / "proposed_model_test_predictions.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_FILE = (
    METRICS_DIR
    / "model_prediction_level_comparison.csv"
)


SUMMARY_FILE = (
    METRICS_DIR
    / "model_prediction_comparison_summary.txt"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "PREDICTION-LEVEL MODEL COMPARISON"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    for file_path in [

        EFFICIENTNET_PREDICTIONS_FILE,

        PROPOSED_PREDICTIONS_FILE

    ]:

        if not file_path.exists():

            raise FileNotFoundError(

                f"Required file not found:\n"
                f"{file_path}"

            )


    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nLoading EfficientNet-B0 predictions..."
    )

    efficientnet_dataframe = pd.read_csv(

        EFFICIENTNET_PREDICTIONS_FILE

    )


    print(
        f"✓ Loaded {len(efficientnet_dataframe)} predictions"
    )


    print(
        "\nLoading proposed model predictions..."
    )

    proposed_dataframe = pd.read_csv(

        PROPOSED_PREDICTIONS_FILE

    )


    print(
        f"✓ Loaded {len(proposed_dataframe)} predictions"
    )


    # --------------------------------------------------------
    # LOAD TEST DATASET TO RECONSTRUCT IMAGE PATHS
    # --------------------------------------------------------

    print(
        "\nReconstructing test image paths..."
    )


    test_dataset = datasets.ImageFolder(

        TEST_DIR

    )


    image_paths = [

        Path(
            sample_path
        ).relative_to(
            PROJECT_ROOT
        ).as_posix()

        for sample_path, _ in test_dataset.samples

    ]


    class_names = test_dataset.classes


    print(
        f"✓ Test images found: {len(image_paths)}"
    )


    print(
        f"✓ Classes: {class_names}"
    )


    # --------------------------------------------------------
    # VALIDATE DATA LENGTHS
    # --------------------------------------------------------

    dataset_size = len(
        image_paths
    )


    efficientnet_size = len(
        efficientnet_dataframe
    )


    proposed_size = len(
        proposed_dataframe
    )


    if efficientnet_size != dataset_size:

        raise ValueError(

            "EfficientNet prediction count does not "
            "match test dataset size.\n"

            f"Predictions: {efficientnet_size}\n"

            f"Dataset images: {dataset_size}"

        )


    if proposed_size != dataset_size:

        raise ValueError(

            "Proposed model prediction count does not "
            "match test dataset size.\n"

            f"Predictions: {proposed_size}\n"

            f"Dataset images: {dataset_size}"

        )


    print(
        "\n✓ Prediction counts match test dataset"
    )


    # --------------------------------------------------------
    # VERIFY TRUE LABEL CONSISTENCY
    # --------------------------------------------------------

    if not (

        efficientnet_dataframe[
            "True_Label"
        ].values

        ==

        proposed_dataframe[
            "True_Label"
        ].values

    ).all():

        raise ValueError(

            "True label ordering differs between the "
            "two prediction files."

        )


    print(
        "✓ True labels are aligned"
    )


    # --------------------------------------------------------
    # CREATE COMPARISON DATAFRAME
    # --------------------------------------------------------

    comparison_dataframe = pd.DataFrame({

        "Image_Path":

            image_paths,


        "True_Label":

            efficientnet_dataframe[
                "True_Label"
            ],


        "EfficientNet_B0_Prediction":

            efficientnet_dataframe[
                "Predicted_Label"
            ],


        "Proposed_Model_Prediction":

            proposed_dataframe[
                "Predicted_Label"
            ]

    })


    # --------------------------------------------------------
    # CALCULATE CORRECTNESS
    # --------------------------------------------------------

    comparison_dataframe[
        "EfficientNet_B0_Correct"
    ] = (

        comparison_dataframe[
            "True_Label"
        ]

        ==

        comparison_dataframe[
            "EfficientNet_B0_Prediction"
        ]

    )


    comparison_dataframe[
        "Proposed_Model_Correct"
    ] = (

        comparison_dataframe[
            "True_Label"
        ]

        ==

        comparison_dataframe[
            "Proposed_Model_Prediction"
        ]

    )


    # --------------------------------------------------------
    # CLASSIFY COMPARISON OUTCOMES
    # --------------------------------------------------------

    def determine_outcome(row):

        efficientnet_correct = row[
            "EfficientNet_B0_Correct"
        ]


        proposed_correct = row[
            "Proposed_Model_Correct"
        ]


        if (

            efficientnet_correct

            and

            proposed_correct

        ):

            return (
                "Both Correct"
            )


        elif (

            not efficientnet_correct

            and

            not proposed_correct

        ):

            return (
                "Both Incorrect"
            )


        elif (

            efficientnet_correct

            and

            not proposed_correct

        ):

            return (
                "Only EfficientNet Correct"
            )


        else:

            return (
                "Only Proposed Model Correct"
            )


    comparison_dataframe[
        "Comparison_Outcome"
    ] = comparison_dataframe.apply(

        determine_outcome,

        axis=1

    )


    # --------------------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------------------

    comparison_dataframe.to_csv(

        OUTPUT_FILE,

        index=False

    )


    # --------------------------------------------------------
    # CALCULATE SUMMARY STATISTICS
    # --------------------------------------------------------

    outcome_counts = (

        comparison_dataframe[
            "Comparison_Outcome"
        ]

        .value_counts()

    )


    both_correct = outcome_counts.get(

        "Both Correct",

        0

    )


    both_incorrect = outcome_counts.get(

        "Both Incorrect",

        0

    )


    only_efficientnet_correct = outcome_counts.get(

        "Only EfficientNet Correct",

        0

    )


    only_proposed_correct = outcome_counts.get(

        "Only Proposed Model Correct",

        0

    )


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION-LEVEL COMPARISON RESULTS"
    )

    print(
        "=" * 70
    )


    print(

        f"\nTotal Test Images: "

        f"{len(comparison_dataframe)}"

    )


    print(

        f"\nBoth Models Correct: "

        f"{both_correct}"

    )


    print(

        f"Both Models Incorrect: "

        f"{both_incorrect}"

    )


    print(

        f"Only EfficientNet-B0 Correct: "

        f"{only_efficientnet_correct}"

    )


    print(

        f"Only Proposed Model Correct: "

        f"{only_proposed_correct}"

    )


    # --------------------------------------------------------
    # CALCULATE PERCENTAGES
    # --------------------------------------------------------

    total_images = len(
        comparison_dataframe
    )


    print(

        "\nPercentage Distribution:"

    )


    print(

        f"Both Correct: "

        f"{both_correct / total_images * 100:.2f}%"

    )


    print(

        f"Both Incorrect: "

        f"{both_incorrect / total_images * 100:.2f}%"

    )


    print(

        f"Only EfficientNet Correct: "

        f"{only_efficientnet_correct / total_images * 100:.2f}%"

    )


    print(

        f"Only Proposed Model Correct: "

        f"{only_proposed_correct / total_images * 100:.2f}%"

    )


    # --------------------------------------------------------
    # CREATE TEXT SUMMARY
    # --------------------------------------------------------

    summary_lines = [

        "=" * 70,

        "PREDICTION-LEVEL MODEL COMPARISON",

        "=" * 70,

        "",

        f"Total Test Images: {total_images}",

        "",

        "COMPARISON OUTCOMES",

        "-" * 70,

        f"Both Models Correct: {both_correct}",

        f"Both Models Incorrect: {both_incorrect}",

        f"Only EfficientNet-B0 Correct: {only_efficientnet_correct}",

        f"Only Proposed Model Correct: {only_proposed_correct}",

        "",

        "PERCENTAGE DISTRIBUTION",

        "-" * 70,

        f"Both Correct: {both_correct / total_images * 100:.2f}%",

        f"Both Incorrect: {both_incorrect / total_images * 100:.2f}%",

        f"Only EfficientNet Correct: "
        f"{only_efficientnet_correct / total_images * 100:.2f}%",

        f"Only Proposed Model Correct: "
        f"{only_proposed_correct / total_images * 100:.2f}%",

        "",

        "=" * 70

    ]


    with open(

        SUMMARY_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            "\n".join(
                summary_lines
            )

        )


    print(

        "\n✓ Detailed comparison saved to:"

    )

    print(
        OUTPUT_FILE
    )


    print(

        "\n✓ Summary saved to:"

    )

    print(
        SUMMARY_FILE
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION-LEVEL COMPARISON COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
