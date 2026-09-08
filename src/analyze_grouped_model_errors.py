# ============================================================
# DETAILED ERROR ANALYSIS
# EFFICIENTNET-B0 - LEAKAGE-SAFE GROUPED DATASET
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

PREDICTIONS_PATH = (
    METRICS_DIR
    / "efficientnet_b0_grouped_predictions.csv"
)

INCORRECT_PREDICTIONS_PATH = (
    METRICS_DIR
    / "efficientnet_b0_grouped_incorrect_predictions.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "EFFICIENTNET-B0 GROUPED SPLIT ERROR ANALYSIS"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading prediction files...")

    predictions_df = pd.read_csv(
        PREDICTIONS_PATH
    )

    incorrect_df = pd.read_csv(
        INCORRECT_PREDICTIONS_PATH
    )

    print(
        f"Total predictions: "
        f"{len(predictions_df)}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(incorrect_df)}"
    )


    # --------------------------------------------------------
    # DISPLAY AVAILABLE COLUMNS
    # --------------------------------------------------------

    print("\nAvailable columns:")

    for column in predictions_df.columns:

        print(
            f"- {column}"
        )


    # --------------------------------------------------------
    # IDENTIFY COLUMN NAMES
    # --------------------------------------------------------

    possible_true_columns = [

        "true_label",
        "true_class",
        "actual_class",
        "actual_label"
    ]

    possible_predicted_columns = [

        "predicted_label",
        "predicted_class",
        "prediction"
    ]

    possible_confidence_columns = [

        "confidence",
        "prediction_confidence",
        "max_probability",
        "confidence_percent"
    ]


    true_column = None

    predicted_column = None

    confidence_column = None


    for column in possible_true_columns:

        if column in predictions_df.columns:

            true_column = column

            break


    for column in possible_predicted_columns:

        if column in predictions_df.columns:

            predicted_column = column

            break


    for column in possible_confidence_columns:

        if column in predictions_df.columns:

            confidence_column = column

            break


    if true_column is None:

        print(
            "\nERROR: Could not identify "
            "true label column."
        )

        return


    if predicted_column is None:

        print(
            "\nERROR: Could not identify "
            "predicted label column."
        )

        return


    print(
        f"\nTrue label column: "
        f"{true_column}"
    )

    print(
        f"Predicted label column: "
        f"{predicted_column}"
    )

    if confidence_column:

        print(
            f"Confidence column: "
            f"{confidence_column}"
        )

    else:

        print(
            "Confidence column: "
            "Not found"
        )


    # --------------------------------------------------------
    # OVERALL ERROR RATE
    # --------------------------------------------------------

    total_predictions = len(
        predictions_df
    )

    total_errors = len(
        incorrect_df
    )

    total_correct = (

        total_predictions
        - total_errors
    )

    error_rate = (

        total_errors
        / total_predictions
        * 100
    )

    accuracy = (

        total_correct
        / total_predictions
        * 100
    )


    print("\n" + "-" * 70)

    print(
        "OVERALL PERFORMANCE"
    )

    print("-" * 70)

    print(
        f"Total predictions: "
        f"{total_predictions}"
    )

    print(
        f"Correct predictions: "
        f"{total_correct}"
    )

    print(
        f"Incorrect predictions: "
        f"{total_errors}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Error rate: "
        f"{error_rate:.2f}%"
    )


    # --------------------------------------------------------
    # ERROR DISTRIBUTION BY TRUE CLASS
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "ERROR DISTRIBUTION BY TRUE CLASS"
    )

    print("-" * 70)

    total_by_class = (

        predictions_df
        .groupby(true_column)
        .size()
    )

    errors_by_class = (

        incorrect_df
        .groupby(true_column)
        .size()
    )


    class_error_summary = pd.DataFrame({

        "total_images":
        total_by_class,

        "incorrect_predictions":
        errors_by_class

    }).fillna(0)


    class_error_summary[
        "incorrect_predictions"
    ] = class_error_summary[
        "incorrect_predictions"
    ].astype(int)


    class_error_summary[
        "correct_predictions"
    ] = (

        class_error_summary[
            "total_images"
        ]

        -

        class_error_summary[
            "incorrect_predictions"
        ]
    )


    class_error_summary[
        "error_rate_percentage"
    ] = (

        class_error_summary[
            "incorrect_predictions"
        ]

        /

        class_error_summary[
            "total_images"
        ]

        * 100
    )


    class_error_summary[
        "accuracy_percentage"
    ] = (

        class_error_summary[
            "correct_predictions"
        ]

        /

        class_error_summary[
            "total_images"
        ]

        * 100
    )


    class_error_summary = (

        class_error_summary

        .sort_values(

            "error_rate_percentage",

            ascending=False
        )
    )


    print(

        class_error_summary
        .to_string()
    )


    # --------------------------------------------------------
    # MOST COMMON CONFUSIONS
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "MOST COMMON MISCLASSIFICATION PATTERNS"
    )

    print("-" * 70)


    confusion_patterns = (

        incorrect_df

        .groupby([

            true_column,

            predicted_column

        ])

        .size()

        .reset_index(

            name="count"
        )

        .sort_values(

            "count",

            ascending=False
        )
    )


    print(

        confusion_patterns
        .to_string(

            index=False
        )
    )


    # --------------------------------------------------------
    # SAVE CONFUSION PATTERNS
    # --------------------------------------------------------

    confusion_patterns_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_error_patterns.csv"
    )


    confusion_patterns.to_csv(

        confusion_patterns_path,

        index=False
    )


    # --------------------------------------------------------
    # CONFIDENCE ANALYSIS
    # --------------------------------------------------------

    if confidence_column:

        print("\n" + "-" * 70)

        print(
            "ERROR CONFIDENCE ANALYSIS"
        )

        print("-" * 70)


        # Convert confidence values to numeric
        incorrect_df[confidence_column] = pd.to_numeric(

            incorrect_df[confidence_column],

            errors="coerce"
        )


        # Remove missing confidence values
        confidence_values = (

            incorrect_df[
                confidence_column
            ]

            .dropna()
        )


        print(

            f"\nAverage confidence "
            f"for incorrect predictions: "

            f"{confidence_values.mean():.2f}%"
        )


        print(

            f"Median confidence "
            f"for incorrect predictions: "

            f"{confidence_values.median():.2f}%"
        )


        print(

            f"Minimum confidence: "

            f"{confidence_values.min():.2f}%"
        )


        print(

            f"Maximum confidence: "

            f"{confidence_values.max():.2f}%"
        )


        # ----------------------------------------------------
        # CONFIDENCE BY ERROR TYPE
        # ----------------------------------------------------

        print("\nAverage confidence by error type:")


        error_confidence_patterns = (

            incorrect_df

            .groupby([

                true_column,

                predicted_column

            ])

            [confidence_column]

            .agg([

                "count",

                "mean",

                "min",

                "max"

            ])

            .reset_index()

            .sort_values(

                "mean",

                ascending=False
            )
        )


        error_confidence_patterns = (

            error_confidence_patterns

            .rename(

                columns={

                    "count":
                    "error_count",

                    "mean":
                    "average_confidence_percent",

                    "min":
                    "minimum_confidence_percent",

                    "max":
                    "maximum_confidence_percent"

                }
            )
        )


        print(

            error_confidence_patterns

            .to_string(

                index=False
            )
        )


        # ----------------------------------------------------
        # HIGH-CONFIDENCE ERRORS
        # ----------------------------------------------------

        high_confidence_errors = (

            incorrect_df[

                incorrect_df[
                    confidence_column
                ]

                >= 90

            ]

            .sort_values(

                confidence_column,

                ascending=False
            )
        )


        print("\n" + "-" * 70)

        print(
            "HIGH-CONFIDENCE ERRORS"
        )

        print("-" * 70)


        print(

            f"\nIncorrect predictions "
            f"with confidence >= 90%: "

            f"{len(high_confidence_errors)}"
        )


        if len(high_confidence_errors) > 0:

            columns_to_display = [

                column

                for column in [

                    "image_name",

                    "image_path",

                    true_column,

                    predicted_column,

                    confidence_column

                ]

                if column in
                high_confidence_errors.columns

            ]


            print(

                high_confidence_errors[
                    columns_to_display
                ]

                .to_string(

                    index=False
                )
            )


        # ----------------------------------------------------
        # LOW-CONFIDENCE ERRORS
        # ----------------------------------------------------

        low_confidence_errors = (

            incorrect_df

            .sort_values(

                confidence_column,

                ascending=True
            )
        )


        print("\n" + "-" * 70)

        print(
            "LOWEST-CONFIDENCE ERRORS"
        )

        print("-" * 70)


        columns_to_display = [

            column

            for column in [

                "image_name",

                "image_path",

                true_column,

                predicted_column,

                confidence_column

            ]

            if column in
            low_confidence_errors.columns

        ]


        print(

            low_confidence_errors[
                columns_to_display
            ]

            .head(10)

            .to_string(

                index=False
            )
        )


        # ----------------------------------------------------
        # BLUE_FINGER TO CLUBBING CONFIDENCE
        # ----------------------------------------------------

        blue_to_clubbing = (

            incorrect_df[

                (

                    incorrect_df[
                        true_column
                    ]

                    == "blue_finger"

                )

                &

                (

                    incorrect_df[
                        predicted_column
                    ]

                    == "clubbing"

                )

            ]
        )


        print("\n" + "-" * 70)

        print(
            "BLUE_FINGER → CLUBBING CONFIDENCE ANALYSIS"
        )

        print("-" * 70)


        print(

            f"\nNumber of blue_finger → "
            f"clubbing errors: "

            f"{len(blue_to_clubbing)}"
        )


        if len(blue_to_clubbing) > 0:


            print(

                f"Average confidence: "

                f"{blue_to_clubbing[confidence_column].mean():.2f}%"
            )


            print(

                f"Minimum confidence: "

                f"{blue_to_clubbing[confidence_column].min():.2f}%"
            )


            print(

                f"Maximum confidence: "

                f"{blue_to_clubbing[confidence_column].max():.2f}%"
            )


            print("\nIndividual predictions:")


            columns_to_display = [

                column

                for column in [

                    "image_name",

                    true_column,

                    predicted_column,

                    confidence_column

                ]

                if column in
                blue_to_clubbing.columns

            ]


            print(

                blue_to_clubbing[

                    columns_to_display

                ]

                .sort_values(

                    confidence_column,

                    ascending=False
                )

                .to_string(

                    index=False
                )
            )


        # ----------------------------------------------------
        # SAVE CONFIDENCE RESULTS
        # ----------------------------------------------------

        error_confidence_path = (

            METRICS_DIR

            / "efficientnet_b0_grouped_error_confidence_analysis.csv"
        )


        error_confidence_patterns.to_csv(

            error_confidence_path,

            index=False
        )


        high_confidence_path = (

            METRICS_DIR

            / "efficientnet_b0_grouped_high_confidence_errors.csv"
        )


        high_confidence_errors.to_csv(

            high_confidence_path,

            index=False
        )


    # --------------------------------------------------------
    # BLUE_FINGER ANALYSIS
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "BLUE_FINGER MISCLASSIFICATION ANALYSIS"
    )

    print("-" * 70)


    blue_finger_errors = (

        incorrect_df[

            incorrect_df[
                true_column
            ]

            == "blue_finger"

        ]
    )


    print(

        f"\nTotal blue_finger "
        f"misclassifications: "

        f"{len(blue_finger_errors)}"
    )


    if len(blue_finger_errors) > 0:


        blue_finger_predictions = (

            blue_finger_errors

            .groupby(

                predicted_column

            )

            .size()

            .sort_values(

                ascending=False
            )
        )


        print(

            "\nblue_finger was predicted as:"
        )


        print(

            blue_finger_predictions
            .to_string()
        )


    # --------------------------------------------------------
    # CLUBBING FALSE POSITIVES
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "CLUBBING FALSE POSITIVE ANALYSIS"
    )

    print("-" * 70)


    clubbing_false_positives = (

        incorrect_df[

            incorrect_df[
                predicted_column
            ]

            == "clubbing"

        ]
    )


    print(

        f"\nImages incorrectly predicted "
        f"as clubbing: "

        f"{len(clubbing_false_positives)}"
    )


    if len(clubbing_false_positives) > 0:


        clubbing_false_positive_distribution = (

            clubbing_false_positives

            .groupby(

                true_column

            )

            .size()

            .sort_values(

                ascending=False
            )
        )


        print(

            "\nTrue classes of false "
            "clubbing predictions:"
        )


        print(

            clubbing_false_positive_distribution
            .to_string()
        )


    # --------------------------------------------------------
    # SAVE CLASS ERROR SUMMARY
    # --------------------------------------------------------

    class_error_summary_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_class_error_analysis.csv"
    )


    class_error_summary.to_csv(

        class_error_summary_path
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "ERROR ANALYSIS COMPLETED"
    )

    print("=" * 70)


    print(

        "\nSaved files:"
    )


    print(

        f"\n1. Class Error Analysis:\n"
        f"{class_error_summary_path}"
    )


    print(

        f"\n2. Error Patterns:\n"
        f"{confusion_patterns_path}"
    )


    if confidence_column:

        print(

            f"\n3. High-Confidence Errors:\n"
            f"{high_confidence_path}"
        )


if __name__ == "__main__":

    main()