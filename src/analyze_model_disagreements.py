import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)


FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = (
    METRICS_DIR
    / "model_prediction_level_comparison.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def create_image_grid(
    dataframe,
    category_name,
    filename
):

    if dataframe.empty:

        print(
            f"\nNo images found for: "
            f"{category_name}"
        )

        return


    total_images = len(
        dataframe
    )


    columns = min(
        3,
        total_images
    )


    rows = (
        total_images
        + columns
        - 1
    ) // columns


    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * 5,
            rows * 5
        )
    )


    if total_images == 1:

        axes = [
            axes
        ]

    else:

        axes = axes.flatten()


    for axis, (
        _,
        row
    ) in zip(
        axes,
        dataframe.iterrows()
    ):

        image_path = Path(
            row["Image_Path"]
        )


        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


            axis.imshow(
                image
            )


            title = (
                f"True: "
                f"{row['True_Label']}\n\n"
                f"EfficientNet: "
                f"{row['EfficientNet_B0_Prediction']}\n\n"
                f"Proposed: "
                f"{row['Proposed_Model_Prediction']}"
            )


            axis.set_title(
                title,
                fontsize=9
            )


        except Exception as error:

            axis.text(
                0.5,
                0.5,
                f"Could not load image\n"
                f"{image_path.name}",
                ha="center",
                va="center"
            )


            print(
                f"\nError loading image:\n"
                f"{image_path}\n"
                f"{error}"
            )


        axis.axis(
            "off"
        )


    for axis in axes[
        total_images:
    ]:

        axis.axis(
            "off"
        )


    figure.suptitle(
        category_name,
        fontsize=16,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94
        ]
    )


    output_path = (
        FIGURES_DIR
        / filename
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"\n✓ Figure saved:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )


    print(
        "QUALITATIVE MODEL ERROR ANALYSIS"
    )


    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nPrediction comparison file "
            f"not found:\n{INPUT_FILE}"
        )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        INPUT_FILE
    )


    print(
        "\n✓ Prediction-level comparison data loaded"
    )


    print(
        f"✓ Total records: "
        f"{len(dataframe)}"
    )


    print(
        "\nAvailable columns:"
    )


    for column in dataframe.columns:

        print(
            f" - {column}"
        )


    # --------------------------------------------------------
    # IDENTIFY COMPARISON COLUMN
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )


    print(
        "AVAILABLE COMPARISON OUTCOMES"
    )


    print(
        "-" * 70
    )


    outcome_counts = (
        dataframe[
            "Comparison_Outcome"
        ]
        .value_counts()
    )


    print(
        outcome_counts
    )


    # --------------------------------------------------------
    # CATEGORY 1
    # --------------------------------------------------------

    efficientnet_only = dataframe[
        dataframe[
            "Comparison_Outcome"
        ]
        == "Only EfficientNet Correct"
    ]


    # --------------------------------------------------------
    # CATEGORY 2
    # --------------------------------------------------------

    proposed_only = dataframe[
        dataframe[
            "Comparison_Outcome"
        ]
        == "Only Proposed Model Correct"
    ]


    # --------------------------------------------------------
    # CATEGORY 3
    # --------------------------------------------------------

    both_incorrect = dataframe[
        dataframe[
            "Comparison_Outcome"
        ]
        == "Both Incorrect"
    ]


    # --------------------------------------------------------
    # PRINT CATEGORY SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )


    print(
        "ERROR ANALYSIS SUMMARY"
    )


    print(
        "=" * 70
    )


    print(
        f"\nOnly EfficientNet-B0 Correct: "
        f"{len(efficientnet_only)}"
    )


    print(
        f"Only Proposed Model Correct: "
        f"{len(proposed_only)}"
    )


    print(
        f"Both Models Incorrect: "
        f"{len(both_incorrect)}"
    )


    # --------------------------------------------------------
    # TRUE CLASS DISTRIBUTION
    # --------------------------------------------------------

    categories = {

        "Only EfficientNet Correct":
        efficientnet_only,

        "Only Proposed Model Correct":
        proposed_only,

        "Both Incorrect":
        both_incorrect

    }


    print(
        "\n"
        + "=" * 70
    )


    print(
        "TRUE CLASS DISTRIBUTION"
    )


    print(
        "=" * 70
    )


    for name, subset in categories.items():

        print(
            f"\n{name}"
        )


        print(
            "-" * len(name)
        )


        if not subset.empty:

            print(
                subset[
                    "True_Label"
                ]
                .value_counts()
            )

        else:

            print(
                "No images"
            )


    # --------------------------------------------------------
    # GENERATE IMAGE FIGURES
    # --------------------------------------------------------

    create_image_grid(

        efficientnet_only,

        "Cases Where Only EfficientNet-B0 "
        "Was Correct",

        "only_efficientnet_correct.png"

    )


    create_image_grid(

        proposed_only,

        "Cases Where Only Proposed Model "
        "Was Correct",

        "only_proposed_model_correct.png"

    )


    create_image_grid(

        both_incorrect,

        "Cases Where Both Models "
        "Were Incorrect",

        "both_models_incorrect.png"

    )


    # --------------------------------------------------------
    # SAVE DETAILED CSV FILES
    # --------------------------------------------------------

    efficientnet_path = (

        METRICS_DIR

        / "only_efficientnet_correct_cases.csv"

    )


    proposed_path = (

        METRICS_DIR

        / "only_proposed_model_correct_cases.csv"

    )


    incorrect_path = (

        METRICS_DIR

        / "both_models_incorrect_cases.csv"

    )


    efficientnet_only.to_csv(

        efficientnet_path,

        index=False

    )


    proposed_only.to_csv(

        proposed_path,

        index=False

    )


    both_incorrect.to_csv(

        incorrect_path,

        index=False

    )


    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )


    print(
        "QUALITATIVE ERROR ANALYSIS COMPLETED"
    )


    print(
        "=" * 70
    )


    print(
        "\nGenerated CSV files:"
    )


    print(
        efficientnet_path
    )


    print(
        proposed_path
    )


    print(
        incorrect_path
    )


    print(
        "\nGenerated figures:"
    )


    print(
        FIGURES_DIR
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()


