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


INPUT_FILE = (
    METRICS_DIR
    / "model_prediction_level_comparison.csv"
)


OUTPUT_FILE = (
    METRICS_DIR
    / "misclassification_patterns_summary.txt"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def analyze_model_errors(
    dataframe,
    prediction_column,
    model_name,
    summary_lines
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{model_name.upper()} MISCLASSIFICATION ANALYSIS"
    )

    print(
        "=" * 70
    )


    summary_lines.append(
        "\n"
        + "=" * 70
    )

    summary_lines.append(
        f"{model_name.upper()} MISCLASSIFICATION ANALYSIS"
    )

    summary_lines.append(
        "=" * 70
    )


    # --------------------------------------------------------
    # IDENTIFY INCORRECT PREDICTIONS
    # --------------------------------------------------------

    errors = dataframe[
        dataframe[
            prediction_column
        ]
        != dataframe[
            "True_Label"
        ]
    ].copy()


    print(
        f"\nTotal Misclassifications: "
        f"{len(errors)}"
    )


    summary_lines.append(
        f"\nTotal Misclassifications: "
        f"{len(errors)}"
    )


    # --------------------------------------------------------
    # CREATE TRUE → PREDICTED TRANSITIONS
    # --------------------------------------------------------

    transitions = (

        errors
        .groupby(
            [
                "True_Label",
                prediction_column
            ]
        )
        .size()
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False
        )

    )


    print(
        "\nMISCLASSIFICATION TRANSITIONS"
    )

    print(
        "-" * 70
    )


    summary_lines.append(
        "\nMISCLASSIFICATION TRANSITIONS"
    )

    summary_lines.append(
        "-" * 70
    )


    for _, row in transitions.iterrows():

        line = (

            f"True: {row['True_Label']} "
            f"→ Predicted: "
            f"{row[prediction_column]} "
            f"| Count: {row['Count']}"

        )


        print(
            line
        )


        summary_lines.append(
            line
        )


    # --------------------------------------------------------
    # ERRORS BY TRUE CLASS
    # --------------------------------------------------------

    true_class_errors = (

        errors[
            "True_Label"
        ]
        .value_counts()

    )


    print(
        "\nERRORS BY TRUE CLASS"
    )

    print(
        "-" * 70
    )


    summary_lines.append(
        "\nERRORS BY TRUE CLASS"
    )

    summary_lines.append(
        "-" * 70
    )


    for class_name, count in true_class_errors.items():

        line = (

            f"{class_name}: "
            f"{count}"

        )


        print(
            line
        )


        summary_lines.append(
            line
        )


    # --------------------------------------------------------
    # MOST COMMON CONFUSION
    # --------------------------------------------------------

    if not transitions.empty:

        most_common = transitions.iloc[
            0
        ]


        line = (

            f"\nMost Common Confusion: "
            f"{most_common['True_Label']} "
            f"→ "
            f"{most_common[prediction_column]} "
            f"({most_common['Count']} image(s))"

        )


        print(
            line
        )


        summary_lines.append(
            line
        )


    return errors


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DETAILED MISCLASSIFICATION PATTERN ANALYSIS"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{INPUT_FILE}"
        )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        INPUT_FILE
    )


    print(
        "\n✓ Prediction comparison data loaded"
    )


    print(
        f"✓ Total images: "
        f"{len(dataframe)}"
    )


    # --------------------------------------------------------
    # SUMMARY STORAGE
    # --------------------------------------------------------

    summary_lines = []


    summary_lines.append(
        "DETAILED MISCLASSIFICATION PATTERN ANALYSIS"
    )


    summary_lines.append(
        "=" * 70
    )


    summary_lines.append(
        f"\nTotal Test Images: "
        f"{len(dataframe)}"
    )


    # --------------------------------------------------------
    # EFFICIENTNET ANALYSIS
    # --------------------------------------------------------

    efficientnet_errors = analyze_model_errors(

        dataframe,

        "EfficientNet_B0_Prediction",

        "EfficientNet-B0",

        summary_lines

    )


    # --------------------------------------------------------
    # PROPOSED MODEL ANALYSIS
    # --------------------------------------------------------

    proposed_errors = analyze_model_errors(

        dataframe,

        "Proposed_Model_Prediction",

        "Proposed Attention-EfficientNet-B0",

        summary_lines

    )


    # --------------------------------------------------------
    # COMMON ERRORS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMMON MISCLASSIFICATIONS"
    )

    print(
        "=" * 70
    )


    common_errors = dataframe[

        (
            dataframe[
                "EfficientNet_B0_Prediction"
            ]
            != dataframe[
                "True_Label"
            ]
        )

        &

        (
            dataframe[
                "Proposed_Model_Prediction"
            ]
            != dataframe[
                "True_Label"
            ]
        )

    ]


    print(
        f"\nImages misclassified by both models: "
        f"{len(common_errors)}"
    )


    summary_lines.append(
        "\n"
        + "=" * 70
    )


    summary_lines.append(
        "COMMON MISCLASSIFICATIONS"
    )


    summary_lines.append(
        "=" * 70
    )


    summary_lines.append(
        f"\nImages misclassified by both models: "
        f"{len(common_errors)}"
    )


    if not common_errors.empty:

        for _, row in common_errors.iterrows():

            line = (

                f"\nTrue: "
                f"{row['True_Label']}"

                f"\nEfficientNet-B0: "
                f"{row['EfficientNet_B0_Prediction']}"

                f"\nProposed Model: "
                f"{row['Proposed_Model_Prediction']}"

            )


            print(
                line
            )


            summary_lines.append(
                line
            )


    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for line in summary_lines:

            file.write(
                line
                + "\n"
            )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "MISCLASSIFICATION ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        f"\n✓ Summary saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()

