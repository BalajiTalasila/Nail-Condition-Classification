import pandas as pd
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

CALIBRATION_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_calibration_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_comprehensive_analysis"
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
# HELPER FUNCTION
# =============================================================================

def get_metric(
    df,
    metric_name,
    preferred_rows
):

    metric_name = str(metric_name)

    for row_name in preferred_rows:

        matching_rows = df[
            df.iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            row_name.lower()
        ]

        if not matching_rows.empty:

            row = matching_rows.iloc[0]

            for column in df.columns[1:]:

                if (
                    metric_name.lower()
                    in str(column).lower()
                ):

                    try:

                        return float(
                            row[column]
                        )

                    except Exception:

                        pass

    return None


# =============================================================================
# LOAD CALIBRATION SUMMARY
# =============================================================================

calibration_file = (
    CALIBRATION_DIR /
    "unseen_source_calibration_summary.csv"
)

if not calibration_file.exists():

    raise FileNotFoundError(
        f"Calibration summary not found: "
        f"{calibration_file}"
    )

calibration_df = pd.read_csv(
    calibration_file
)


# =============================================================================
# ANALYZE EACH MODEL
# =============================================================================

results = []

print("=" * 100)
print("COMPREHENSIVE UNSEEN-SOURCE MODEL ANALYSIS")
print("=" * 100)


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

    report_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_classification_report.csv"
    )

    predictions_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    # -------------------------------------------------------------------------
    # VERIFY FILES
    # -------------------------------------------------------------------------

    if not report_file.exists():

        raise FileNotFoundError(
            f"Classification report not found: "
            f"{report_file}"
        )

    if not predictions_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{predictions_file}"
        )


    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    report_df = pd.read_csv(
        report_file
    )

    prediction_df = pd.read_csv(
        predictions_file
    )


    # -------------------------------------------------------------------------
    # STANDARDIZE CORRECTNESS
    # -------------------------------------------------------------------------

    if "correct" not in prediction_df.columns:

        prediction_df["correct"] = (
            prediction_df["true_class"]
            ==
            prediction_df["predicted_class"]
        )

    else:

        if prediction_df["correct"].dtype == object:

            prediction_df["correct"] = (
                prediction_df["correct"]
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


    # -------------------------------------------------------------------------
    # BASIC METRICS
    # -------------------------------------------------------------------------

    total_images = len(
        prediction_df
    )

    correct_predictions = int(
        prediction_df["correct"].sum()
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
        prediction_df[
            "confidence"
        ].mean()
    )

    confidence_accuracy_gap = (
        average_confidence -
        accuracy
    )


    # -------------------------------------------------------------------------
    # HIGH-CONFIDENCE ERRORS
    # -------------------------------------------------------------------------

    high_confidence_errors = int(
        (
            (~prediction_df["correct"])
            &
            (
                prediction_df[
                    "confidence"
                ] >= 0.90
            )
        ).sum()
    )


    # -------------------------------------------------------------------------
    # CLASSIFICATION REPORT METRICS
    # -------------------------------------------------------------------------

    first_column = (
        report_df.columns[0]
    )

    report_df = report_df.rename(
        columns={
            first_column:
            "metric_row"
        }
    )


    macro_precision = get_metric(
        report_df,
        "precision",
        [
            "macro avg",
            "macro_average",
            "macro"
        ]
    )

    macro_recall = get_metric(
        report_df,
        "recall",
        [
            "macro avg",
            "macro_average",
            "macro"
        ]
    )

    macro_f1 = get_metric(
        report_df,
        "f1",
        [
            "macro avg",
            "macro_average",
            "macro"
        ]
    )

    weighted_f1 = get_metric(
        report_df,
        "f1",
        [
            "weighted avg",
            "weighted_average",
            "weighted"
        ]
    )


    # -------------------------------------------------------------------------
    # CALIBRATION METRICS
    # -------------------------------------------------------------------------

    calibration_row = calibration_df[
        calibration_df["model"]
        ==
        model_name
    ]

    if calibration_row.empty:

        raise ValueError(
            f"Calibration data not found "
            f"for {model_name}"
        )

    calibration_row = (
        calibration_row.iloc[0]
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
                else None
            ),

        "macro_recall_percent":
            (
                macro_recall * 100
                if macro_recall is not None
                else None
            ),

        "macro_f1_percent":
            (
                macro_f1 * 100
                if macro_f1 is not None
                else None
            ),

        "weighted_f1_percent":
            (
                weighted_f1 * 100
                if weighted_f1 is not None
                else None
            ),

        "average_confidence_percent":
            average_confidence * 100,

        "confidence_accuracy_gap_percent":
            confidence_accuracy_gap * 100,

        "ece_percent":
            calibration_row[
                "ece_percent"
            ],

        "mce_percent":
            calibration_row[
                "mce_percent"
            ],

        "brier_score":
            calibration_row[
                "brier_score"
            ],

        "high_confidence_errors":
            high_confidence_errors,

        "calibration_behavior":
            calibration_row[
                "calibration_behavior"
            ]
    }


    results.append(
        result
    )


    # -------------------------------------------------------------------------
    # DISPLAY RESULTS
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
        f"{calibration_row['ece_percent']:.2f}%"
    )

    print(
        f"Brier Score: "
        f"{calibration_row['brier_score']:.6f}"
    )

    print(
        f"High-confidence errors: "
        f"{high_confidence_errors}"
    )


# =============================================================================
# CREATE FINAL DATAFRAME
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
# SAVE RESULTS
# =============================================================================

output_file = (
    OUTPUT_DIR /
    "unseen_source_comprehensive_model_comparison.csv"
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
    "FINAL COMPREHENSIVE "
    "UNSEEN-SOURCE COMPARISON"
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
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 100)

print()
print("Output file:")
print(output_file)

