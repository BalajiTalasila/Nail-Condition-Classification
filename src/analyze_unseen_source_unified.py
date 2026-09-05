import pandas as pd
import numpy as np

from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path.cwd()

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
    "unseen_source_unified_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODELS = {

    "convnextv2_tiny": {
        "display": "ConvNeXtV2-Tiny"
    },

    "densenet121": {
        "display": "DenseNet121"
    },

    "efficientnet_b0": {
        "display": "EfficientNet-B0"
    },

    "proposed_attention_efficientnet_b0": {
        "display": "Proposed Attention EfficientNet-B0"
    }
}


# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def calculate_calibration_metrics(
    confidence,
    correct,
    num_bins=10
):

    confidence = np.asarray(
        confidence,
        dtype=float
    )

    correct = np.asarray(
        correct,
        dtype=float
    )

    bins = np.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    ece = 0.0
    mce = 0.0

    bin_results = []

    for i in range(num_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == num_bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        sample_count = int(
            mask.sum()
        )

        if sample_count > 0:

            average_confidence = float(
                confidence[mask].mean()
            )

            accuracy = float(
                correct[mask].mean()
            )

            calibration_gap = (
                average_confidence -
                accuracy
            )

            absolute_gap = abs(
                calibration_gap
            )

            ece += (
                sample_count /
                len(confidence)
            ) * absolute_gap

            mce = max(
                mce,
                absolute_gap
            )

        else:

            average_confidence = np.nan
            accuracy = np.nan
            calibration_gap = np.nan

        bin_results.append({

            "bin_lower":
                lower,

            "bin_upper":
                upper,

            "sample_count":
                sample_count,

            "average_confidence":
                average_confidence,

            "accuracy":
                accuracy,

            "calibration_gap":
                calibration_gap
        })

    brier_score = float(
        np.mean(
            (
                confidence -
                correct
            ) ** 2
        )
    )

    return (
        ece,
        mce,
        brier_score,
        pd.DataFrame(
            bin_results
        )
    )


# =============================================================================
# ANALYZE MODELS
# =============================================================================

results = []

print("=" * 100)
print("UNIFIED UNSEEN-SOURCE MODEL ANALYSIS")
print("=" * 100)

print()
print(
    "All metrics are calculated from the same "
    "detailed prediction CSV for each model."
)


for model_name, config in MODELS.items():

    print()
    print("=" * 100)

    print(
        f"ANALYZING: "
        f"{config['display']}"
    )

    print("=" * 100)


    # -------------------------------------------------------------------------
    # FILE PATHS
    # -------------------------------------------------------------------------

    predictions_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )

    report_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_classification_report.csv"
    )


    # -------------------------------------------------------------------------
    # VERIFY FILES
    # -------------------------------------------------------------------------

    if not predictions_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{predictions_file}"
        )

    if not report_file.exists():

        raise FileNotFoundError(
            f"Classification report not found: "
            f"{report_file}"
        )


    # -------------------------------------------------------------------------
    # LOAD PREDICTIONS
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        predictions_file
    )


    print()
    print(
        "Prediction columns:"
    )

    print(
        list(df.columns)
    )


    # -------------------------------------------------------------------------
    # STANDARDIZE CORRECTNESS
    # -------------------------------------------------------------------------

    if "correct" not in df.columns:

        if (
            "true_class" in df.columns
            and
            "predicted_class" in df.columns
        ):

            df["correct"] = (
                df["true_class"]
                ==
                df["predicted_class"]
            )

        else:

            raise ValueError(
                "Could not determine correctness. "
                "true_class and predicted_class "
                "columns are required."
            )

    else:

        if df["correct"].dtype == object:

            df["correct"] = (
                df["correct"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                })
            )

        if df["correct"].isna().any():

            raise ValueError(
                "The correct column contains "
                "unrecognized values."
            )

    df["correct"] = (
        df["correct"]
        .astype(bool)
    )


    # -------------------------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # -------------------------------------------------------------------------

    if "confidence" not in df.columns:

        raise ValueError(
            f"Confidence column missing in "
            f"{predictions_file}"
        )


    # -------------------------------------------------------------------------
    # BASIC METRICS
    # -------------------------------------------------------------------------

    total_images = len(
        df
    )

    correct_predictions = int(
        df["correct"].sum()
    )

    total_errors = (
        total_images -
        correct_predictions
    )

    accuracy = (
        correct_predictions /
        total_images
    )


    # -------------------------------------------------------------------------
    # CONFIDENCE METRICS
    # -------------------------------------------------------------------------

    average_confidence = float(
        df["confidence"].mean()
    )

    confidence_accuracy_gap = (
        average_confidence -
        accuracy
    )


    # -------------------------------------------------------------------------
    # CALIBRATION METRICS
    # -------------------------------------------------------------------------

    (
        ece,
        mce,
        brier_score,
        confidence_bins
    ) = calculate_calibration_metrics(

        confidence=
        df["confidence"].values,

        correct=
        df["correct"]
        .astype(int)
        .values,

        num_bins=10
    )


    # -------------------------------------------------------------------------
    # HIGH-CONFIDENCE ERRORS
    # -------------------------------------------------------------------------

    high_confidence_errors = int(

        (
            (~df["correct"])
            &
            (
                df["confidence"] >=
                0.90
            )
        ).sum()
    )


    # -------------------------------------------------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------------------------------------------------

    report_df = pd.read_csv(
        report_file
    )

    first_column = (
        report_df.columns[0]
    )

    report_df = report_df.rename(
        columns={
            first_column:
            "metric_row"
        }
    )


    def get_report_metric(
        row_name,
        metric_name
    ):

        matching_rows = report_df[

            report_df[
                "metric_row"
            ]
            .astype(str)
            .str.strip()
            .str.lower()

            ==

            row_name.lower()
        ]

        if matching_rows.empty:

            return None

        row = matching_rows.iloc[0]

        for column in report_df.columns:

            if (
                metric_name.lower()
                in
                str(column).lower()
            ):

                try:

                    return float(
                        row[column]
                    )

                except Exception:

                    continue

        return None


    macro_precision = get_report_metric(
        "macro avg",
        "precision"
    )

    macro_recall = get_report_metric(
        "macro avg",
        "recall"
    )

    macro_f1 = get_report_metric(
        "macro avg",
        "f1"
    )

    weighted_f1 = get_report_metric(
        "weighted avg",
        "f1"
    )


    # -------------------------------------------------------------------------
    # CALIBRATION BEHAVIOR
    # -------------------------------------------------------------------------

    gap_percent = (
        confidence_accuracy_gap *
        100
    )

    if abs(gap_percent) <= 2:

        calibration_behavior = (
            "Well calibrated"
        )

    elif gap_percent > 2:

        calibration_behavior = (
            "Overconfident"
        )

    else:

        calibration_behavior = (
            "Underconfident"
        )


    # -------------------------------------------------------------------------
    # SAVE CONFIDENCE BINS
    # -------------------------------------------------------------------------

    bins_file = (
        OUTPUT_DIR /
        f"{model_name}_confidence_bins.csv"
    )

    confidence_bins.to_csv(
        bins_file,
        index=False
    )


    # -------------------------------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------------------------------

    result = {

        "model":
            model_name,

        "model_display":
            config["display"],

        "total_images":
            total_images,

        "correct_predictions":
            correct_predictions,

        "total_errors":
            total_errors,

        "accuracy_percent":
            accuracy * 100,

        "macro_precision_percent":
            (
                macro_precision * 100
                if macro_precision is not None
                else np.nan
            ),

        "macro_recall_percent":
            (
                macro_recall * 100
                if macro_recall is not None
                else np.nan
            ),

        "macro_f1_percent":
            (
                macro_f1 * 100
                if macro_f1 is not None
                else np.nan
            ),

        "weighted_f1_percent":
            (
                weighted_f1 * 100
                if weighted_f1 is not None
                else np.nan
            ),

        "average_confidence_percent":
            average_confidence * 100,

        "confidence_accuracy_gap_percent":
            confidence_accuracy_gap * 100,

        "ece_percent":
            ece * 100,

        "mce_percent":
            mce * 100,

        "brier_score":
            brier_score,

        "high_confidence_errors":
            high_confidence_errors,

        "calibration_behavior":
            calibration_behavior
    }


    results.append(
        result
    )


    # -------------------------------------------------------------------------
    # DISPLAY MODEL RESULTS
    # -------------------------------------------------------------------------

    print()

    print(
        f"Total images: "
        f"{total_images}"
    )

    print(
        f"Correct predictions: "
        f"{correct_predictions}"
    )

    print(
        f"Errors: "
        f"{total_errors}"
    )

    print()

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Confidence gap: "
        f"{confidence_accuracy_gap * 100:.2f}%"
    )

    print()

    print(
        f"Macro Precision: "
        f"{macro_precision * 100:.2f}%"
        if macro_precision is not None
        else
        "Macro Precision: Not available"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall * 100:.2f}%"
        if macro_recall is not None
        else
        "Macro Recall: Not available"
    )

    print(
        f"Macro F1-score: "
        f"{macro_f1 * 100:.2f}%"
        if macro_f1 is not None
        else
        "Macro F1-score: Not available"
    )

    print(
        f"Weighted F1-score: "
        f"{weighted_f1 * 100:.2f}%"
        if weighted_f1 is not None
        else
        "Weighted F1-score: Not available"
    )

    print()

    print(
        f"ECE: "
        f"{ece * 100:.2f}%"
    )

    print(
        f"MCE: "
        f"{mce * 100:.2f}%"
    )

    print(
        f"Brier Score: "
        f"{brier_score:.6f}"
    )

    print(
        f"High-confidence errors: "
        f"{high_confidence_errors}"
    )

    print(
        f"Calibration behavior: "
        f"{calibration_behavior}"
    )


# =============================================================================
# CREATE FINAL RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    results
)


# =============================================================================
# SORT BY ACCURACY
# =============================================================================

results_df = results_df.sort_values(

    by="accuracy_percent",

    ascending=False
)


# =============================================================================
# CREATE RANKINGS
# =============================================================================

results_df[
    "accuracy_rank"
] = (

    results_df[
        "accuracy_percent"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


results_df[
    "macro_f1_rank"
] = (

    results_df[
        "macro_f1_percent"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype("Int64")
)


results_df[
    "ece_rank"
] = (

    results_df[
        "ece_percent"
    ]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)


results_df[
    "brier_score_rank"
] = (

    results_df[
        "brier_score"
    ]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)


# =============================================================================
# SAVE FINAL RESULTS
# =============================================================================

output_file = (

    OUTPUT_DIR /

    "unified_unseen_source_model_comparison.csv"
)


results_df.to_csv(

    output_file,

    index=False
)


# =============================================================================
# DISPLAY FINAL RESULTS
# =============================================================================

print()

print("=" * 100)

print(
    "FINAL UNIFIED "
    "UNSEEN-SOURCE MODEL COMPARISON"
)

print("=" * 100)

print()

print(
    results_df.to_string(
        index=False
    )
)


print()

print("=" * 100)

print(
    "UNIFIED ANALYSIS "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 100)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Final comparison file:"
)

print(
    output_file
)

