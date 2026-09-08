# ============================================================
# VISUALIZE EFFICIENTNET-B0 GROUPED SPLIT ERRORS
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

FIGURES_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "grouped_error_analysis"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

PREDICTIONS_FILE = (
    METRICS_DIR
    / "efficientnet_b0_grouped_predictions.csv"
)

INCORRECT_FILE = (
    METRICS_DIR
    / "efficientnet_b0_grouped_incorrect_predictions.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    return image


# ============================================================
# VISUALIZE PREDICTIONS
# ============================================================

def visualize_predictions(
    dataframe,
    title,
    output_filename,
    images_per_page=20
):

    total_images = len(
        dataframe
    )

    print(
        f"\nCreating visualization: {title}"
    )

    print(
        f"Total images: {total_images}"
    )

    if total_images == 0:

        print(
            "No images found."
        )

        return


    # --------------------------------------------------------
    # SPLIT INTO PAGES
    # --------------------------------------------------------

    for page_start in range(
        0,
        total_images,
        images_per_page
    ):

        page_end = min(

            page_start
            + images_per_page,

            total_images
        )

        page_dataframe = dataframe.iloc[
            page_start:page_end
        ]


        # ----------------------------------------------------
        # GRID CONFIGURATION
        # ----------------------------------------------------

        columns = 4

        rows = (

            len(page_dataframe)
            + columns - 1

        ) // columns


        figure, axes = plt.subplots(

            rows,

            columns,

            figsize=(
                18,
                rows * 5
            )
        )


        # Convert axes into flat list
        if rows == 1:

            axes = axes.flatten()

        else:

            axes = axes.flatten()


        # ----------------------------------------------------
        # DISPLAY IMAGES
        # ----------------------------------------------------

        for index, (

            dataframe_index,
            row

        ) in enumerate(

            page_dataframe.iterrows()
        ):

            axis = axes[index]

            image_path = Path(
                row["image_path"]
            )

            try:

                image = load_image(
                    image_path
                )

                axis.imshow(
                    image
                )

            except Exception as error:

                print(
                    f"Could not load: "
                    f"{image_path}"
                )

                print(
                    f"Error: {error}"
                )

                axis.text(

                    0.5,

                    0.5,

                    "Image\nNot Found",

                    horizontalalignment="center",

                    verticalalignment="center",

                    fontsize=12
                )


            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            true_class = (
                row["true_class"]
            )

            predicted_class = (
                row["predicted_class"]
            )

            confidence = (
                row.get(
                    "confidence_percent",
                    None
                )
            )


            if confidence is not None:

                title_text = (

                    f"True: {true_class}\n"

                    f"Pred: {predicted_class}\n"

                    f"Confidence: "
                    f"{confidence:.2f}%"
                )

            else:

                title_text = (

                    f"True: {true_class}\n"

                    f"Pred: {predicted_class}"
                )


            axis.set_title(

                title_text,

                fontsize=10,

                fontweight="bold"
            )

            axis.axis(
                "off"
            )


        # ----------------------------------------------------
        # HIDE UNUSED AXES
        # ----------------------------------------------------

        for index in range(

            len(page_dataframe),

            len(axes)

        ):

            axes[index].axis(
                "off"
            )


        # ----------------------------------------------------
        # PAGE TITLE
        # ----------------------------------------------------

        figure.suptitle(

            f"{title}\n"

            f"Images "
            f"{page_start + 1}"
            f"-"
            f"{page_end} "
            f"of {total_images}",

            fontsize=18,

            fontweight="bold"
        )


        plt.tight_layout(

            rect=[
                0,
                0,
                1,
                0.95
            ]
        )


        # ----------------------------------------------------
        # SAVE FIGURE
        # ----------------------------------------------------

        page_number = (

            page_start
            // images_per_page

        ) + 1


        output_path = (

            FIGURES_DIR

            / (
                f"{output_filename}"
                f"_page_"
                f"{page_number}.png"
            )
        )


        plt.savefig(

            output_path,

            dpi=200,

            bbox_inches="tight"
        )

        plt.close()


        print(

            f"Saved: "
            f"{output_path}"
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GROUPED SPLIT ERROR VISUALIZATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading prediction files..."
    )


    predictions_df = pd.read_csv(

        PREDICTIONS_FILE
    )


    incorrect_df = pd.read_csv(

        INCORRECT_FILE
    )


    print(

        f"Total predictions: "
        f"{len(predictions_df)}"
    )


    print(

        f"Incorrect predictions: "
        f"{len(incorrect_df)}"
    )


    # ========================================================
    # 1. ALL INCORRECT PREDICTIONS
    # ========================================================

    visualize_predictions(

        dataframe=incorrect_df,

        title="All Incorrect Predictions",

        output_filename="all_incorrect_predictions"
    )


    # ========================================================
    # 2. BLUE_FINGER → CLUBBING
    # ========================================================

    blue_to_clubbing = incorrect_df[

        (
            incorrect_df["true_class"]

            == "blue_finger"
        )

        &

        (
            incorrect_df["predicted_class"]

            == "clubbing"
        )

    ].copy()


    print(

        "\nBlue_finger → clubbing "
        f"errors: "
        f"{len(blue_to_clubbing)}"
    )


    visualize_predictions(

        dataframe=blue_to_clubbing,

        title="Blue Finger Predicted as Clubbing",

        output_filename="blue_finger_to_clubbing"
    )


    # ========================================================
    # 3. HIGH-CONFIDENCE ERRORS
    # ========================================================

    if "confidence_percent" in incorrect_df.columns:

        high_confidence_errors = (

            incorrect_df[

                incorrect_df[
                    "confidence_percent"
                ]

                >= 90

            ].copy()
        )


        print(

            "\nHigh-confidence errors "
            f"(>= 90%): "
            f"{len(high_confidence_errors)}"
        )


        visualize_predictions(

            dataframe=high_confidence_errors,

            title=(
                "High-Confidence "
                "Incorrect Predictions"
            ),

            output_filename=(
                "high_confidence_errors"
            )
        )


    # ========================================================
    # 4. ONYCHOGRYPHOSIS → MELANOMA
    # ========================================================

    onych_to_melanoma = incorrect_df[

        (
            incorrect_df["true_class"]

            == "Onychogryphosis"
        )

        &

        (
            incorrect_df["predicted_class"]

            == (
                "Acral_Lentiginous_Melanoma"
            )
        )

    ].copy()


    print(

        "\nOnychogryphosis → "
        "Acral_Lentiginous_Melanoma "
        f"errors: "
        f"{len(onych_to_melanoma)}"
    )


    visualize_predictions(

        dataframe=onych_to_melanoma,

        title=(
            "Onychogryphosis Predicted "
            "as Acral Lentiginous Melanoma"
        ),

        output_filename=(
            "onychogryphosis_to_melanoma"
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "ERROR VISUALIZATION COMPLETED"
    )

    print("=" * 70)


    print(

        "\nFigures saved in:"
    )

    print(
        FIGURES_DIR
    )


    print(

        "\nGenerated visualizations:"
    )

    print(
        "1. All incorrect predictions"
    )

    print(
        "2. blue_finger → clubbing"
    )

    print(
        "3. High-confidence errors"
    )

    print(
        "4. Onychogryphosis → melanoma"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()