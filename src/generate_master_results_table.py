from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


CALIBRATION_FILE = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "confidence_calibration_analysis" /
    "model_calibration_summary.csv"
)


STATISTICAL_FILE = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "publication_statistical_analysis" /
    "publication_statistical_summary.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "master_results"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_DISPLAY_NAMES = {

    "convnextv2_tiny":
    "ConvNeXtV2-Tiny",

    "densenet121":
    "DenseNet121",

    "efficientnet_b0":
    "EfficientNet-B0",

    "proposed_attention_efficientnet_b0":
    "Proposed Attention EfficientNet-B0"

}


print("=" * 80)
print("MASTER PUBLICATION RESULTS TABLE")
print("=" * 80)


# ============================================================
# LOAD CALIBRATION RESULTS
# ============================================================

if not CALIBRATION_FILE.exists():

    raise FileNotFoundError(
        f"Calibration file not found:\n"
        f"{CALIBRATION_FILE}"
    )


calibration_dataframe = pd.read_csv(
    CALIBRATION_FILE
)


print("\nCalibration results loaded:")
print(
    CALIBRATION_FILE
)


# ============================================================
# LOAD STATISTICAL RESULTS
# ============================================================

if not STATISTICAL_FILE.exists():

    raise FileNotFoundError(
        f"Statistical analysis file not found:\n"
        f"{STATISTICAL_FILE}"
    )


statistical_dataframe = pd.read_csv(
    STATISTICAL_FILE
)


print("\nStatistical results loaded:")
print(
    STATISTICAL_FILE
)


# ============================================================
# CREATE MAIN PERFORMANCE TABLE
# ============================================================

performance_columns = [

    "model",

    "total_images",

    "accuracy_percent",

    "average_confidence_percent",

    "confidence_accuracy_difference_percent",

    "ece_percent",

    "mce_percent",

    "brier_score"

]


performance_dataframe = calibration_dataframe[
    performance_columns
].copy()


performance_dataframe[
    "model_display"
] = performance_dataframe[
    "model"
].map(
    MODEL_DISPLAY_NAMES
)


performance_dataframe = performance_dataframe[
    [

        "model_display",

        "total_images",

        "accuracy_percent",

        "average_confidence_percent",

        "confidence_accuracy_difference_percent",

        "ece_percent",

        "mce_percent",

        "brier_score"

    ]
]


performance_dataframe = performance_dataframe.rename(
    columns={

        "model_display":
        "Model",

        "total_images":
        "Test Images",

        "accuracy_percent":
        "Accuracy (%)",

        "average_confidence_percent":
        "Average Confidence (%)",

        "confidence_accuracy_difference_percent":
        "Confidence - Accuracy (%)",

        "ece_percent":
        "ECE (%)",

        "mce_percent":
        "MCE (%)",

        "brier_score":
        "Brier Score"

    }
)


performance_dataframe = performance_dataframe.sort_values(
    by="Accuracy (%)",
    ascending=False
)


print("\n" + "=" * 80)
print("TABLE 1: OVERALL MODEL PERFORMANCE")
print("=" * 80)

print(
    performance_dataframe.to_string(
        index=False
    )
)


# ============================================================
# CREATE STATISTICAL COMPARISON TABLE
# ============================================================

statistical_columns = [

    "baseline_model",

    "proposed_model",

    "baseline_accuracy_percent",

    "proposed_accuracy_percent",

    "accuracy_difference_percent",

    "mcnemar_p_value",

    "mcnemar_significance",

    "wilcoxon_p_value",

    "wilcoxon_significance",

    "rank_biserial_correlation",

    "rank_biserial_interpretation",

    "cohens_dz",

    "cohens_dz_interpretation"

]


statistical_table = statistical_dataframe[
    statistical_columns
].copy()


statistical_table[
    "baseline_model"
] = statistical_table[
    "baseline_model"
].map(
    MODEL_DISPLAY_NAMES
)


statistical_table[
    "proposed_model"
] = statistical_table[
    "proposed_model"
].map(
    MODEL_DISPLAY_NAMES
)


statistical_table = statistical_table.rename(
    columns={

        "baseline_model":
        "Baseline Model",

        "proposed_model":
        "Proposed Model",

        "baseline_accuracy_percent":
        "Baseline Accuracy (%)",

        "proposed_accuracy_percent":
        "Proposed Accuracy (%)",

        "accuracy_difference_percent":
        "Accuracy Difference (%)",

        "mcnemar_p_value":
        "McNemar p-value",

        "mcnemar_significance":
        "McNemar Result",

        "wilcoxon_p_value":
        "Wilcoxon p-value",

        "wilcoxon_significance":
        "Wilcoxon Result",

        "rank_biserial_correlation":
        "Rank-Biserial Correlation",

        "rank_biserial_interpretation":
        "Effect Size",

        "cohens_dz":
        "Cohen's dz",

        "cohens_dz_interpretation":
        "Cohen's dz Interpretation"

    }
)


print("\n" + "=" * 80)
print("TABLE 2: PROPOSED MODEL STATISTICAL COMPARISON")
print("=" * 80)

print(
    statistical_table.to_string(
        index=False
    )
)


# ============================================================
# CREATE FORMATTED PUBLICATION TABLE
# ============================================================

publication_table = performance_dataframe.copy()


numeric_columns = [

    "Accuracy (%)",

    "Average Confidence (%)",

    "Confidence - Accuracy (%)",

    "ECE (%)",

    "MCE (%)",

    "Brier Score"

]


for column in numeric_columns:

    publication_table[
        column
    ] = publication_table[
        column
    ].astype(
        float
    )


publication_table[
    "Accuracy (%)"
] = publication_table[
    "Accuracy (%)"
].map(
    lambda value:
    f"{value:.2f}"
)


publication_table[
    "Average Confidence (%)"
] = publication_table[
    "Average Confidence (%)"
].map(
    lambda value:
    f"{value:.2f}"
)


publication_table[
    "Confidence - Accuracy (%)"
] = publication_table[
    "Confidence - Accuracy (%)"
].map(
    lambda value:
    f"{value:+.2f}"
)


publication_table[
    "ECE (%)"
] = publication_table[
    "ECE (%)"
].map(
    lambda value:
    f"{value:.2f}"
)


publication_table[
    "MCE (%)"
] = publication_table[
    "MCE (%)"
].map(
    lambda value:
    f"{value:.2f}"
)


publication_table[
    "Brier Score"
] = publication_table[
    "Brier Score"
].map(
    lambda value:
    f"{value:.4f}"
)


print("\n" + "=" * 80)
print("TABLE 3: PUBLICATION-READY RESULTS TABLE")
print("=" * 80)

print(
    publication_table.to_string(
        index=False
    )
)


# ============================================================
# SAVE TABLES
# ============================================================

performance_file = (
    OUTPUT_DIR /
    "overall_model_performance.csv"
)


statistical_file = (
    OUTPUT_DIR /
    "proposed_model_statistical_comparison.csv"
)


publication_file = (
    OUTPUT_DIR /
    "publication_ready_results_table.csv"
)


performance_dataframe.to_csv(
    performance_file,
    index=False
)


statistical_table.to_csv(
    statistical_file,
    index=False
)


publication_table.to_csv(
    publication_file,
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    performance_file
)

print(
    statistical_file
)

print(
    publication_file
)


print("\n" + "=" * 80)
print("MASTER RESULTS GENERATION COMPLETED")
print("=" * 80)
