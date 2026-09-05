import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

EFFICIENTNET_PATH = os.path.join(
    METRICS_DIR,
    "efficientnet_b0_test_predictions.csv"
)

PROPOSED_PATH = os.path.join(
    METRICS_DIR,
    "proposed_model_test_predictions.csv"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "high_confidence_error_analysis.txt"
)

EFFICIENTNET_ERRORS_PATH = os.path.join(
    METRICS_DIR,
    "efficientnet_high_confidence_errors.csv"
)

PROPOSED_ERRORS_PATH = os.path.join(
    METRICS_DIR,
    "proposed_model_high_confidence_errors.csv"
)


# ============================================================
# FUNCTIONS
# ============================================================

def get_probability_columns(dataframe):

    return [
        column
        for column in dataframe.columns
        if column.startswith("Probability_")
    ]


def add_confidence_column(dataframe):

    probability_columns = get_probability_columns(
        dataframe
    )

    dataframe = dataframe.copy()

    dataframe["Confidence"] = (
        dataframe[
            probability_columns
        ].max(
            axis=1
        )
    )

    dataframe["Confidence_Percentage"] = (
        dataframe["Confidence"] * 100
    )

    return dataframe


def analyze_model_errors(
    dataframe,
    model_name
):

    errors = dataframe[
        dataframe["Correct"] == False
    ].copy()

    errors = errors.sort_values(
        by="Confidence",
        ascending=False
    )

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal Misclassifications: "
        f"{len(errors)}"
    )

    if len(errors) == 0:

        print("No errors found.")

        return errors

    print(
        f"Average Error Confidence: "
        f"{errors['Confidence_Percentage'].mean():.2f}%"
    )

    print(
        f"Highest Error Confidence: "
        f"{errors['Confidence_Percentage'].max():.2f}%"
    )

    print(
        f"Lowest Error Confidence: "
        f"{errors['Confidence_Percentage'].min():.2f}%"
    )

    print("\nHIGH-CONFIDENCE ERRORS")

    high_confidence_errors = errors[
        errors["Confidence_Percentage"] >= 90
    ]

    print(
        f"Errors with confidence >= 90%: "
        f"{len(high_confidence_errors)}"
    )

    if len(high_confidence_errors) > 0:

        for _, row in high_confidence_errors.iterrows():

            print(
                f"\nTrue Label: "
                f"{row['True_Label']}"
            )

            print(
                f"Predicted Label: "
                f"{row['Predicted_Label']}"
            )

            print(
                f"Confidence: "
                f"{row['Confidence_Percentage']:.2f}%"
            )

    else:

        print(
            "No errors with confidence >= 90%."
        )

    print("\nTOP 5 MOST CONFIDENT ERRORS")

    for _, row in errors.head(5).iterrows():

        print(
            f"\nTrue: "
            f"{row['True_Label']}"
        )

        print(
            f"Predicted: "
            f"{row['Predicted_Label']}"
        )

        print(
            f"Confidence: "
            f"{row['Confidence_Percentage']:.2f}%"
        )

    print("\nERROR DISTRIBUTION BY TRUE CLASS")

    print(
        errors[
            "True_Label"
        ].value_counts()
    )

    print("\nERROR DISTRIBUTION BY PREDICTED CLASS")

    print(
        errors[
            "Predicted_Label"
        ].value_counts()
    )

    return errors


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HIGH-CONFIDENCE ERROR ANALYSIS")
    print("=" * 70)

    print("\nLoading prediction data...")

    efficientnet_df = pd.read_csv(
        EFFICIENTNET_PATH
    )

    proposed_df = pd.read_csv(
        PROPOSED_PATH
    )

    # --------------------------------------------------------
    # ADD CONFIDENCE
    # --------------------------------------------------------

    efficientnet_df = add_confidence_column(
        efficientnet_df
    )

    proposed_df = add_confidence_column(
        proposed_df
    )

    # --------------------------------------------------------
    # ANALYZE ERRORS
    # --------------------------------------------------------

    efficientnet_errors = analyze_model_errors(
        efficientnet_df,
        "EfficientNet-B0"
    )

    proposed_errors = analyze_model_errors(
        proposed_df,
        "Proposed Attention-EfficientNet-B0"
    )

    # --------------------------------------------------------
    # SAVE CSV FILES
    # --------------------------------------------------------

    efficientnet_errors.to_csv(
        EFFICIENTNET_ERRORS_PATH,
        index=False
    )

    proposed_errors.to_csv(
        PROPOSED_ERRORS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "HIGH-CONFIDENCE ERROR ANALYSIS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for model_name, errors in [

            (
                "EFFICIENTNET-B0",
                efficientnet_errors
            ),

            (
                "PROPOSED ATTENTION-EFFICIENTNET-B0",
                proposed_errors
            )

        ]:

            file.write(
                model_name + "\n"
            )

            file.write(
                "-" * 70 + "\n"
            )

            file.write(
                f"Total Misclassifications: "
                f"{len(errors)}\n"
            )

            if len(errors) > 0:

                file.write(
                    f"Average Error Confidence: "
                    f"{errors['Confidence_Percentage'].mean():.2f}%\n"
                )

                file.write(
                    f"Highest Error Confidence: "
                    f"{errors['Confidence_Percentage'].max():.2f}%\n"
                )

                file.write(
                    f"Lowest Error Confidence: "
                    f"{errors['Confidence_Percentage'].min():.2f}%\n"
                )

                high_confidence = errors[
                    errors[
                        "Confidence_Percentage"
                    ] >= 90
                ]

                file.write(
                    f"Errors >= 90% confidence: "
                    f"{len(high_confidence)}\n\n"
                )

                file.write(
                    "TOP CONFIDENT ERRORS\n"
                )

                for _, row in errors.head(5).iterrows():

                    file.write(
                        f"True: "
                        f"{row['True_Label']} | "
                        f"Predicted: "
                        f"{row['Predicted_Label']} | "
                        f"Confidence: "
                        f"{row['Confidence_Percentage']:.2f}%\n"
                    )

                file.write("\n")

            file.write("\n")

    print(
        "\n✓ EfficientNet error CSV saved:"
    )

    print(
        EFFICIENTNET_ERRORS_PATH
    )

    print(
        "\n✓ Proposed model error CSV saved:"
    )

    print(
        PROPOSED_ERRORS_PATH
    )

    print(
        "\n✓ Summary saved:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE ERROR ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()
