import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


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
    / "final_model_comparison.csv"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPREHENSIVE MODEL COMPARISON FIGURE"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"Input file not found:\n"
            f"{INPUT_FILE}"

        )


    dataframe = pd.read_csv(
        INPUT_FILE
    )


    print(
        "\n✓ Final comparison data loaded"
    )


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    figure, axes = plt.subplots(

        2,

        3,

        figsize=(18, 10)

    )


    figure.suptitle(

        "Comprehensive Model Performance and Complexity Comparison",

        fontsize=18,

        fontweight="bold"

    )


    # --------------------------------------------------------
    # METRIC CONFIGURATION
    # --------------------------------------------------------

    metrics = [

        (

            "Accuracy_Percentage",

            "Test Accuracy (%)",

            "Accuracy Comparison",

            ".2f"

        ),

        (

            "Macro_F1_Percentage",

            "Macro F1-Score (%)",

            "Macro F1 Comparison",

            ".2f"

        ),

        (

            "MCC",

            "MCC",

            "Matthews Correlation Coefficient",

            ".4f"

        ),

        (

            "Average_Inference_Time_ms",

            "Inference Time (ms/image)",

            "Inference Time Comparison",

            ".4f"

        ),

        (

            "Total_Parameters_Millions",

            "Parameters (Millions)",

            "Model Complexity",

            ".2f"

        ),

        (

            "Model_Size_MB",

            "Model Size (MB)",

            "Model Storage Size",

            ".2f"

        )

    ]


    # --------------------------------------------------------
    # GENERATE SUBPLOTS
    # --------------------------------------------------------

    for axis, metric in zip(

        axes.flatten(),

        metrics

    ):


        column = metric[0]

        ylabel = metric[1]

        title = metric[2]

        value_format = metric[3]


        bars = axis.bar(

            dataframe["Model"],

            dataframe[column]

        )


        axis.set_ylabel(

            ylabel

        )


        axis.set_title(

            title,

            fontweight="bold"

        )


        axis.tick_params(

            axis="x",

            rotation=20

        )


        axis.grid(

            axis="y",

            alpha=0.3

        )


        for bar, value in zip(

            bars,

            dataframe[column]

        ):


            axis.text(

                bar.get_x()

                + bar.get_width() / 2,

                bar.get_height(),

                format(

                    value,

                    value_format

                ),

                ha="center",

                va="bottom",

                fontsize=8

            )


    # --------------------------------------------------------
    # ADJUST LAYOUT
    # --------------------------------------------------------

    plt.tight_layout(

        rect=[

            0,

            0,

            1,

            0.95

        ]

    )


    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    output_path = (

        FIGURES_DIR

        / "comprehensive_model_comparison.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(

        f"\n✓ Comprehensive figure saved:\n"
        f"{output_path}"

    )


    print(

        "\n" + "=" * 70

    )

    print(

        "COMPREHENSIVE FIGURE GENERATED SUCCESSFULLY"

    )

    print(

        "=" * 70

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
