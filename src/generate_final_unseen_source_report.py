import pandas as pd
from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path.cwd()

UNIFIED_ANALYSIS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_unified_analysis"
)

STATISTICAL_ANALYSIS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_statistical_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "final_unseen_source_report"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# INPUT FILES
# =============================================================================

unified_file = (
    UNIFIED_ANALYSIS_DIR /
    "unified_unseen_source_model_comparison.csv"
)

pairwise_file = (
    STATISTICAL_ANALYSIS_DIR /
    "unseen_source_pairwise_mcnemar_tests.csv"
)

accuracy_file = (
    STATISTICAL_ANALYSIS_DIR /
    "unseen_source_accuracy_statistical_summary.csv"
)


# =============================================================================
# VERIFY INPUT FILES
# =============================================================================

required_files = [
    unified_file,
    pairwise_file,
    accuracy_file
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{file_path}"
        )


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 100)
print("FINAL UNSEEN-SOURCE RESEARCH REPORT")
print("=" * 100)

print()
print("Loading analysis results...")

unified_df = pd.read_csv(
    unified_file
)

pairwise_df = pd.read_csv(
    pairwise_file
)

accuracy_df = pd.read_csv(
    accuracy_file
)


print()
print(
    f"Models loaded: "
    f"{len(unified_df)}"
)

print(
    f"Pairwise comparisons loaded: "
    f"{len(pairwise_df)}"
)


# =============================================================================
# STANDARDIZE MODEL ORDER
# =============================================================================

final_df = unified_df.copy()


# =============================================================================
# PERFORMANCE RANKING
# =============================================================================

final_df[
    "accuracy_rank"
] = (
    final_df[
        "accuracy_percent"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


final_df[
    "macro_f1_rank"
] = (
    final_df[
        "macro_f1_percent"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


final_df[
    "ece_rank"
] = (
    final_df[
        "ece_percent"
    ]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)


final_df[
    "brier_score_rank"
] = (
    final_df[
        "brier_score"
    ]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)


# =============================================================================
# OVERALL SCORE
# =============================================================================
#
# The overall score combines model ranks.
#
# Lower score = stronger overall performance.
#
# Accuracy and Macro F1 receive higher importance.
#

final_df[
    "overall_rank_score"
] = (

    0.35 *
    final_df[
        "accuracy_rank"
    ]

    +

    0.35 *
    final_df[
        "macro_f1_rank"
    ]

    +

    0.15 *
    final_df[
        "ece_rank"
    ]

    +

    0.15 *
    final_df[
        "brier_score_rank"
    ]
)


final_df[
    "overall_rank"
] = (
    final_df[
        "overall_rank_score"
    ]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)


final_df = (
    final_df
    .sort_values(
        by=[
            "overall_rank",
            "accuracy_percent",
            "macro_f1_percent"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# PERFORMANCE TABLE
# =============================================================================

performance_columns = [

    "overall_rank",

    "model_display",

    "total_images",

    "correct_predictions",

    "total_errors",

    "accuracy_percent",

    "macro_precision_percent",

    "macro_recall_percent",

    "macro_f1_percent",

    "weighted_f1_percent"
]


performance_df = (
    final_df[
        performance_columns
    ]
    .copy()
)


# =============================================================================
# CALIBRATION TABLE
# =============================================================================

calibration_columns = [

    "overall_rank",

    "model_display",

    "average_confidence_percent",

    "confidence_accuracy_gap_percent",

    "ece_percent",

    "mce_percent",

    "brier_score",

    "high_confidence_errors",

    "calibration_behavior"
]


calibration_df = (
    final_df[
        calibration_columns
    ]
    .copy()
)


# =============================================================================
# STATISTICAL SIGNIFICANCE SUMMARY
# =============================================================================

statistical_df = (
    pairwise_df
    .copy()
)


# Detect the significance column automatically.

significance_column = None

for column in statistical_df.columns:

    if (
        "significant"
        in column.lower()
    ):

        significance_column = column
        break


if significance_column is not None:

    significant_comparisons = int(
        statistical_df[
            significance_column
        ].astype(bool).sum()
    )

else:

    significant_comparisons = 0


total_comparisons = len(
    statistical_df
)


# =============================================================================
# IDENTIFY BEST MODEL
# =============================================================================

best_model = (
    final_df.iloc[0]
)


best_model_name = (
    best_model[
        "model_display"
    ]
)


# =============================================================================
# GENERATE AUTOMATED INTERPRETATION
# =============================================================================

interpretation_lines = []


interpretation_lines.append(
    "FINAL UNSEEN-SOURCE EVALUATION INTERPRETATION"
)

interpretation_lines.append(
    "=" * 70
)

interpretation_lines.append("")


interpretation_lines.append(
    "Dataset and evaluation setup:"
)

interpretation_lines.append(
    f"- Number of evaluated models: "
    f"{len(final_df)}"
)

interpretation_lines.append(
    f"- Number of unseen-source test images: "
    f"{int(best_model['total_images'])}"
)

interpretation_lines.append(
    "- All models were evaluated on the same "
    "aligned image set."
)

interpretation_lines.append("")


interpretation_lines.append(
    "Best overall model:"
)

interpretation_lines.append(
    f"- {best_model_name}"
)

interpretation_lines.append(
    f"- Accuracy: "
    f"{best_model['accuracy_percent']:.2f}%"
)

interpretation_lines.append(
    f"- Macro F1-score: "
    f"{best_model['macro_f1_percent']:.2f}%"
)

interpretation_lines.append(
    f"- Expected Calibration Error: "
    f"{best_model['ece_percent']:.2f}%"
)

interpretation_lines.append(
    f"- Brier Score: "
    f"{best_model['brier_score']:.6f}"
)

interpretation_lines.append(
    f"- Calibration behavior: "
    f"{best_model['calibration_behavior']}"
)

interpretation_lines.append("")


interpretation_lines.append(
    "Statistical significance analysis:"
)

interpretation_lines.append(
    f"- Pairwise McNemar comparisons: "
    f"{total_comparisons}"
)

interpretation_lines.append(
    f"- Statistically significant comparisons "
    f"detected: {significant_comparisons}"
)


if significant_comparisons == 0:

    interpretation_lines.append(
        "- No statistically significant accuracy "
        "differences were observed between the "
        "evaluated models on the 71-image "
        "unseen-source test set."
    )

    interpretation_lines.append(
        "- Therefore, differences in observed "
        "accuracy should be interpreted carefully, "
        "particularly because of the limited "
        "unseen-source sample size."
    )


interpretation_lines.append("")


interpretation_lines.append(
    "Overall interpretation:"
)

interpretation_lines.append(
    f"- {best_model_name} achieved the strongest "
    "overall balance according to the weighted "
    "ranking framework."
)

interpretation_lines.append(
    "- The final ranking considers predictive "
    "performance together with confidence "
    "calibration."
)

interpretation_lines.append(
    "- Accuracy and Macro F1-score receive higher "
    "weight than calibration metrics in the "
    "overall ranking."
)


# =============================================================================
# GENERATE BEST-MODEL RATIONALE
# =============================================================================

rationale_lines = []


rationale_lines.append(
    "BEST MODEL SELECTION RATIONALE"
)

rationale_lines.append(
    "=" * 70
)

rationale_lines.append("")


rationale_lines.append(
    f"Selected model: "
    f"{best_model_name}"
)

rationale_lines.append("")


rationale_lines.append(
    "Selection criteria:"
)

rationale_lines.append(
    "1. Predictive accuracy"
)

rationale_lines.append(
    "2. Macro F1-score"
)

rationale_lines.append(
    "3. Expected Calibration Error"
)

rationale_lines.append(
    "4. Brier Score"
)

rationale_lines.append(
    "5. Overall weighted ranking"
)

rationale_lines.append("")


rationale_lines.append(
    "Model characteristics:"
)

rationale_lines.append(
    f"- Accuracy: "
    f"{best_model['accuracy_percent']:.2f}%"
)

rationale_lines.append(
    f"- Macro Precision: "
    f"{best_model['macro_precision_percent']:.2f}%"
)

rationale_lines.append(
    f"- Macro Recall: "
    f"{best_model['macro_recall_percent']:.2f}%"
)

rationale_lines.append(
    f"- Macro F1-score: "
    f"{best_model['macro_f1_percent']:.2f}%"
)

rationale_lines.append(
    f"- Weighted F1-score: "
    f"{best_model['weighted_f1_percent']:.2f}%"
)

rationale_lines.append(
    f"- ECE: "
    f"{best_model['ece_percent']:.2f}%"
)

rationale_lines.append(
    f"- Brier Score: "
    f"{best_model['brier_score']:.6f}"
)

rationale_lines.append(
    f"- Calibration behavior: "
    f"{best_model['calibration_behavior']}"
)

rationale_lines.append("")


rationale_lines.append(
    "Scientific interpretation:"
)

rationale_lines.append(
    f"{best_model_name} was selected as the "
    "strongest overall model based on the "
    "combined ranking framework."
)

rationale_lines.append(
    "This selection represents the best balance "
    "between predictive performance and confidence "
    "calibration among the evaluated models."
)

rationale_lines.append(
    "However, the McNemar statistical significance "
    "analysis should be considered when making "
    "claims about superiority in accuracy."
)


# =============================================================================
# SAVE CSV FILES
# =============================================================================

final_file = (
    OUTPUT_DIR /
    "final_unseen_source_model_ranking.csv"
)

performance_file = (
    OUTPUT_DIR /
    "final_unseen_source_performance_table.csv"
)

calibration_file = (
    OUTPUT_DIR /
    "final_unseen_source_calibration_table.csv"
)

statistical_file = (
    OUTPUT_DIR /
    "final_unseen_source_statistical_summary.csv"
)


final_df.to_csv(
    final_file,
    index=False
)

performance_df.to_csv(
    performance_file,
    index=False
)

calibration_df.to_csv(
    calibration_file,
    index=False
)


statistical_summary_df = pd.DataFrame([
    {

        "number_of_models":
            len(final_df),

        "pairwise_comparisons":
            total_comparisons,

        "significant_comparisons":
            significant_comparisons,

        "best_overall_model":
            best_model_name,

        "best_model_accuracy_percent":
            best_model[
                "accuracy_percent"
            ],

        "best_model_macro_f1_percent":
            best_model[
                "macro_f1_percent"
            ]
    }
])


statistical_summary_df.to_csv(
    statistical_file,
    index=False
)


# =============================================================================
# SAVE TEXT REPORTS
# =============================================================================

interpretation_file = (
    OUTPUT_DIR /
    "final_unseen_source_interpretation.txt"
)

rationale_file = (
    OUTPUT_DIR /
    "best_model_selection_rationale.txt"
)


interpretation_file.write_text(
    "\n".join(
        interpretation_lines
    ),
    encoding="utf-8"
)


rationale_file.write_text(
    "\n".join(
        rationale_lines
    ),
    encoding="utf-8"
)


# =============================================================================
# DISPLAY FINAL RANKING
# =============================================================================

print()
print("=" * 100)
print("FINAL MODEL RANKING")
print("=" * 100)
print()


print(
    final_df[
        [
            "overall_rank",
            "model_display",
            "accuracy_percent",
            "macro_f1_percent",
            "ece_percent",
            "brier_score",
            "overall_rank_score"
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("FINAL REPORT GENERATED SUCCESSFULLY")
print("=" * 100)


print()
print(
    f"Best overall model: "
    f"{best_model_name}"
)


print()
print("Output directory:")
print(OUTPUT_DIR)


print()
print("Generated files:")

for file_path in [

    final_file,
    performance_file,
    calibration_file,
    statistical_file,
    interpretation_file,
    rationale_file

]:

    print()
    print(file_path)
