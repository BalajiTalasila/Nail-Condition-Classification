# ============================================================
# VISUALIZE GROUPED DATASET ERROR SOURCES
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

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "grouped_error_sources"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PREDICTIONS_PATH = (
    METRICS_DIR
    / "efficientnet_b0_grouped_predictions.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "VISUAL ERROR INSPECTION"
    )

    print(
        "EFFICIENTNET-B0 GROUPED SPLIT"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading predictions..."
    )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )


    # --------------------------------------------------------
    # BLUE_FINGER → CLUBBING
    # --------------------------------------------------------

    errors = df[

        (

            df["true_class"]

            == "blue_finger"

        )

        &

        (

            df["predicted_class"]

            == "clubbing"

        )

    ].copy()


    print(
        f"\nblue_finger → clubbing errors: "
        f"{len(errors)}"
    )


    # --------------------------------------------------------
    # CREATE FIGURES
    # --------------------------------------------------------

    images_per_page = 12


    for page_start in range(

        0,

        len(errors),

        images_per_page

    ):


        page_errors = errors.iloc[

            page_start:

            page_start + images_per_page

        ]


        rows = 3

        cols = 4


        fig, axes = plt.subplots(

            rows,

            cols,

            figsize=(16, 12)

        )


        axes = axes.flatten()


        for ax in axes:

            ax.axis(
                "off"
            )


        for index, (

            _,

            row

        ) in enumerate(

            page_errors.iterrows()

        ):


            image_path = Path(

                row["image_path"]

            )


            try:

                image = Image.open(

                    image_path

                ).convert(

                    "RGB"

                )


                axes[index].imshow(

                    image

                )


                axes[index].set_title(

                    f"True: {row['true_class']}\n"
                    f"Predicted: {row['predicted_class']}\n"
                    f"Confidence: "
                    f"{row['confidence_percent']:.2f}%",

                    fontsize=10

                )


                axes[index].axis(
                    "off"
                )


            except Exception as error:

                print(

                    f"Could not load: "
                    f"{image_path}"

                )

                print(
                    error
                )


        page_number = (

            page_start

            // images_per_page

            + 1

        )


        output_path = (

            OUTPUT_DIR

            / (

                "blue_finger_to_clubbing_"

                f"page_{page_number}.png"

            )

        )


        plt.tight_layout()


        plt.savefig(

            output_path,

            dpi=200,

            bbox_inches="tight"

        )


        plt.close()


        print(

            f"Saved: {output_path}"

        )


    print(
        "\n" + "=" * 70
    )

    print(
        "VISUAL ERROR ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()