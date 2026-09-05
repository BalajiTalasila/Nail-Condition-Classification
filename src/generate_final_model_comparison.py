import os
import pandas as pd

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


# ============================================================
# INPUT FILES
# ============================================================

PERFORMANCE_FILE = (
    METRICS_DIR
    / "model_comparison.csv"
)


COMPLEXITY_FILE = (
    METRICS_DIR
    / "model_complexity_comparison.csv"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = (
    METRICS_DIR
    / "final_model_comparison.csv"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "FINAL MODEL COMPARISON GENERATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    if not PERFORMANCE_FILE.exists():

        raise FileNotFoundError(

            f"Performance file not found:\n"
            f"{PERFORMANCE_FILE}"

        )


    if not COMPLEXITY_FILE.exists():

        raise FileNotFoundError(

            f"Complexity file not found:\n"
            f"{COMPLEXITY_FILE}"

        )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading performance metrics..."
    )


    performance_dataframe = pd.read_csv(

        PERFORMANCE_FILE

    )


    print(
        "✓ Performance metrics loaded"
    )


    print(
        "\nLoading model complexity metrics..."
    )


    complexity_dataframe = pd.read_csv(

        COMPLEXITY_FILE

    )


    print(
        "✓ Complexity metrics loaded"
    )


    # --------------------------------------------------------
    # MERGE DATA
    # --------------------------------------------------------

    print(
        "\nMerging datasets..."
    )


    final_dataframe = pd.merge(

        performance_dataframe,

        complexity_dataframe,

        on="Model",

        how="inner"

    )


    # --------------------------------------------------------
    # CALCULATE ADDITIONAL METRICS
    # --------------------------------------------------------

    print(
        "Calculating additional metrics..."
    )


    final_dataframe[
        "Accuracy_Percentage"
    ] = (

        final_dataframe[
            "Accuracy"
        ]

        * 100

    )


    final_dataframe[
        "Macro_F1_Percentage"
    ] = (

        final_dataframe[
            "F1_Macro"
        ]

        * 100

    )


    final_dataframe[
        "Parameter_Efficiency"
    ] = (

        final_dataframe[
            "Accuracy_Percentage"
        ]

        /

        final_dataframe[
            "Total_Parameters_Millions"
        ]

    )


    # --------------------------------------------------------
    # ROUND DISPLAY VALUES
    # --------------------------------------------------------

    final_dataframe[
        "Accuracy_Percentage"
    ] = final_dataframe[
        "Accuracy_Percentage"
    ].round(4)


    final_dataframe[
        "Macro_F1_Percentage"
    ] = final_dataframe[
        "Macro_F1_Percentage"
    ].round(4)


    final_dataframe[
        "Parameter_Efficiency"
    ] = final_dataframe[
        "Parameter_Efficiency"
    ].round(4)


    final_dataframe[
        "Average_Inference_Time_ms"
    ] = final_dataframe[
        "Average_Inference_Time_ms"
    ].round(4)


    final_dataframe[
        "Total_Parameters_Millions"
    ] = final_dataframe[
        "Total_Parameters_Millions"
    ].round(4)


    final_dataframe[
        "Model_Size_MB"
    ] = final_dataframe[
        "Model_Size_MB"
    ].round(4)


    # --------------------------------------------------------
    # SELECT FINAL COLUMNS
    # --------------------------------------------------------

    final_columns = [

        "Model",

        "Accuracy",

        "Accuracy_Percentage",

        "Precision_Macro",

        "Recall_Macro",

        "F1_Macro",

        "Macro_F1_Percentage",

        "F1_Weighted",

        "MCC",

        "Average_Inference_Time_ms",

        "Total_Parameters",

        "Trainable_Parameters",

        "Total_Parameters_Millions",

        "Model_Size_MB",

        "Parameter_Efficiency"

    ]


    final_dataframe = final_dataframe[
        final_columns
    ]


    # --------------------------------------------------------
    # SORT BY ACCURACY
    # --------------------------------------------------------

    final_dataframe = (

        final_dataframe

        .sort_values(

            by="Accuracy",

            ascending=False

        )

        .reset_index(

            drop=True

        )

    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    final_dataframe.to_csv(

        OUTPUT_FILE,

        index=False

    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 70
    )


    print(

        final_dataframe.to_string(

            index=False

        )

    )


    print(
        "\nFinal comparison saved to:"
    )

    print(
        OUTPUT_FILE
    )


    # --------------------------------------------------------
    # BEST MODELS
    # --------------------------------------------------------

    best_accuracy = (

        final_dataframe.loc[

            final_dataframe[
                "Accuracy"
            ].idxmax()

        ]

    )


    fastest_model = (

        final_dataframe.loc[

            final_dataframe[
                "Average_Inference_Time_ms"
            ].idxmin()

        ]

    )


    smallest_model = (

        final_dataframe.loc[

            final_dataframe[
                "Total_Parameters"
            ].idxmin()

        ]

    )


    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL HIGHLIGHTS"
    )

    print(
        "=" * 70
    )


    print(

        f"\nBest Accuracy:\n"
        f"{best_accuracy['Model']}\n"
        f"{best_accuracy['Accuracy'] * 100:.2f}%"

    )


    print(

        f"\nFastest Model:\n"
        f"{fastest_model['Model']}\n"
        f"{fastest_model['Average_Inference_Time_ms']:.4f} ms/image"

    )


    print(

        f"\nSmallest Model:\n"
        f"{smallest_model['Model']}\n"
        f"{smallest_model['Total_Parameters_Millions']:.4f} million parameters"

    )


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL COMPARISON COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
