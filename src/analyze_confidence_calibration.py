from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


DETAILED_RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_detailed_results"
)


OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "confidence_calibration_analysis"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_NAMES = [

    "convnextv2_tiny",

    "densenet121",

    "efficientnet_b0",

    "proposed_attention_efficientnet_b0"

]


NUM_BINS = 10


print("=" * 80)
print("CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)


# ============================================================
# CALIBRATION METRIC FUNCTIONS
# ============================================================

def calculate_ece_and_mce(
    confidences,
    correctness,
    num_bins=10
):

    bin_boundaries = np.linspace(
        0,
        1,
        num_bins + 1
    )


    ece = 0.0

    mce = 0.0


    bin_results = []


    for bin_index in range(num_bins):

        lower_bound = bin_boundaries[
            bin_index
        ]


        upper_bound = bin_boundaries[
            bin_index + 1
        ]


        if bin_index == num_bins - 1:

            mask = (

                confidences >= lower_bound

            ) & (

                confidences <= upper_bound

            )

        else:

            mask = (

                confidences >= lower_bound

            ) & (

                confidences < upper_bound

            )


        bin_confidences = confidences[
            mask
        ]


        bin_correctness = correctness[
            mask
        ]


        sample_count = len(
            bin_confidences
        )


        if sample_count == 0:

            average_confidence = np.nan

            accuracy = np.nan

            calibration_gap = np.nan


        else:

            average_confidence = np.mean(
                bin_confidences
            )


            accuracy = np.mean(
                bin_correctness
            )


            calibration_gap = abs(

                accuracy

                -

                average_confidence

            )


            bin_weight = (

                sample_count

                /

                len(confidences)

            )


            ece += (

                bin_weight

                *

                calibration_gap

            )


            mce = max(

                mce,

                calibration_gap

            )


        bin_results.append({

            "bin_index":
            bin_index + 1,

            "lower_bound":
            lower_bound,

            "upper_bound":
            upper_bound,

            "sample_count":
            sample_count,

            "average_confidence":
            average_confidence,

            "accuracy":
            accuracy,

            "calibration_gap":
            calibration_gap

        })


    return (

        ece,

        mce,

        pd.DataFrame(
            bin_results
        )

    )


# ============================================================
# MULTICLASS BRIER SCORE
# ============================================================

def calculate_brier_score(
    probabilities,
    true_class_indices
):

    number_of_classes = probabilities.shape[
        1
    ]


    one_hot_targets = np.eye(
        number_of_classes
    )[
        true_class_indices
    ]


    squared_errors = (

        probabilities

        -

        one_hot_targets

    ) ** 2


    brier_score = np.mean(

        np.sum(

            squared_errors,

            axis=1

        )

    )


    return brier_score


# ============================================================
# LOAD AND ANALYZE EACH MODEL
# ============================================================

calibration_summary = []


all_bin_results = []


for model_name in MODEL_NAMES:

    print("\n" + "=" * 80)
    print(
        f"ANALYZING: {model_name}"
    )
    print("=" * 80)


    prediction_file = (

        DETAILED_RESULTS_DIR /

        f"{model_name}_image_predictions.csv"

    )


    if not prediction_file.exists():

        print(
            f"\nWARNING: File not found:"
        )

        print(
            prediction_file
        )

        continue


    dataframe = pd.read_csv(
        prediction_file
    )


    print(
        f"\nLoaded images: "
        f"{len(dataframe)}"
    )


    probability_columns = [

        column

        for column in dataframe.columns

        if column.startswith(
            "probability_"
        )

    ]


    if not probability_columns:

        raise RuntimeError(

            f"No probability columns found "

            f"for {model_name}"

        )


    print(
        f"Probability columns found: "
        f"{len(probability_columns)}"
    )


    confidences = dataframe[
        "confidence"
    ].to_numpy()


    correctness = (

        dataframe[
            "true_class"
        ]

        ==

        dataframe[
            "predicted_class"
        ]

    ).astype(
        int
    ).to_numpy()


    probabilities = dataframe[
        probability_columns
    ].to_numpy()


    true_class_indices = dataframe[
        "true_class_index"
    ].to_numpy()


    # --------------------------------------------------------
    # ECE AND MCE
    # --------------------------------------------------------

    ece, mce, bin_dataframe = (

        calculate_ece_and_mce(

            confidences,

            correctness,

            NUM_BINS

        )

    )


    bin_dataframe[
        "model"
    ] = model_name


    all_bin_results.append(
        bin_dataframe
    )


    # --------------------------------------------------------
    # BRIER SCORE
    # --------------------------------------------------------

    brier_score = (

        calculate_brier_score(

            probabilities,

            true_class_indices

        )

    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = np.mean(
        correctness
    )


    average_confidence = np.mean(
        confidences
    )


    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Average confidence: "
        f"{average_confidence * 100:.2f}%"
    )


    print(
        f"Expected Calibration Error (ECE): "
        f"{ece * 100:.2f}%"
    )


    print(
        f"Maximum Calibration Error (MCE): "
        f"{mce * 100:.2f}%"
    )


    print(
        f"Brier Score: "
        f"{brier_score:.6f}"
    )


    # --------------------------------------------------------
    # OVERCONFIDENCE / UNDERCONFIDENCE
    # --------------------------------------------------------

    confidence_accuracy_difference = (

        average_confidence

        -

        accuracy

    )


    if confidence_accuracy_difference > 0:

        calibration_direction = (
            "Overconfident"
        )

    elif confidence_accuracy_difference < 0:

        calibration_direction = (
            "Underconfident"
        )

    else:

        calibration_direction = (
            "Perfectly matched overall"
        )


    print(
        f"Calibration tendency: "
        f"{calibration_direction}"
    )


    print(
        f"Confidence - Accuracy difference: "
        f"{confidence_accuracy_difference * 100:.2f}%"
    )


    # --------------------------------------------------------
    # ADD TO SUMMARY
    # --------------------------------------------------------

    calibration_summary.append({

        "model":
        model_name,

        "total_images":
        len(dataframe),

        "accuracy":
        accuracy,

        "average_confidence":
        average_confidence,

        "confidence_accuracy_difference":
        confidence_accuracy_difference,

        "calibration_tendency":
        calibration_direction,

        "ece":
        ece,

        "mce":
        mce,

        "brier_score":
        brier_score

    })


    print("\nCalibration bins:")

    print(

        bin_dataframe.to_string(

            index=False

        )

    )


# ============================================================
# CREATE FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL CALIBRATION COMPARISON")
print("=" * 80)


summary_dataframe = pd.DataFrame(
    calibration_summary
)


summary_dataframe = summary_dataframe.sort_values(

    "ece",

    ascending=True

)


summary_dataframe[
    "accuracy_percent"
] = (

    summary_dataframe[
        "accuracy"
    ]

    *

    100

)


summary_dataframe[
    "average_confidence_percent"
] = (

    summary_dataframe[
        "average_confidence"
    ]

    *

    100

)


summary_dataframe[
    "confidence_accuracy_difference_percent"
] = (

    summary_dataframe[
        "confidence_accuracy_difference"
    ]

    *

    100

)


summary_dataframe[
    "ece_percent"
] = (

    summary_dataframe[
        "ece"
    ]

    *

    100

)


summary_dataframe[
    "mce_percent"
] = (

    summary_dataframe[
        "mce"
    ]

    *

    100

)


print("\n")

print(

    summary_dataframe.to_string(

        index=False

    )

)


# ============================================================
# COMBINE BIN RESULTS
# ============================================================

combined_bin_dataframe = pd.concat(

    all_bin_results,

    ignore_index=True

)


# ============================================================
# SAVE RESULTS
# ============================================================

summary_file = (

    OUTPUT_DIR /

    "model_calibration_summary.csv"

)


summary_dataframe.to_csv(

    summary_file,

    index=False

)


bins_file = (

    OUTPUT_DIR /

    "calibration_bin_results.csv"

)


combined_bin_dataframe.to_csv(

    bins_file,

    index=False

)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    summary_file
)


print(
    bins_file
)


print("\n" + "=" * 80)
print("CONFIDENCE CALIBRATION ANALYSIS COMPLETED")
print("=" * 80)
