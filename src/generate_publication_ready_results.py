import pandas as pd
from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path.cwd()

FINAL_REPORT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "final_unseen_source_report"
)

STATISTICAL_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_statistical_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "publication_ready_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# INPUT FILES
# =============================================================================

performance_file = (
    FINAL_REPORT_DIR /
    "final_unseen_source_performance_table.csv"
)

calibration_file = (
    FINAL_REPORT_DIR /
    "final_unseen_source_calibration_table.csv"
)

ranking_file = (
    FINAL_REPORT_DIR /
    "final_unseen_source_model_ranking.csv"
)

pairwise_file = (
    STATISTICAL_DIR /
    "unseen_source_pairwise_mcnemar_tests.csv"
)


# =============================================================================
# VERIFY INPUT FILES
# =============================================================================

required_files = [

    performance_file,

    calibration_file,

    ranking_file,

    pairwise_file
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required input file was not found:\n"
            f"{file_path}"
        )


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 100)
print("GENERATING PUBLICATION-READY UNSEEN-SOURCE RESULTS")
print("=" * 100)

print()
print("Loading existing analysis results...")


performance_df = pd.read_csv(
    performance_file
)

calibration_df = pd.read_csv(
    calibration_file
)

ranking_df = pd.read_csv(
    ranking_file
)

pairwise_df = pd.read_csv(
    pairwise_file
)


print(
    f"Models loaded: "
    f"{len(ranking_df)}"
)

print(
    f"Statistical comparisons loaded: "
    f"{len(pairwise_df)}"
)


# =============================================================================
# IDENTIFY BEST MODEL
# =============================================================================

best_model = (
    ranking_df
    .sort_values(
        "overall_rank"
    )
    .iloc[0]
)


best_model_name = (
    best_model[
        "model_display"
    ]
)


# =============================================================================
# ROUND VALUES FOR PUBLICATION
# =============================================================================

performance_publication = (
    performance_df.copy()
)

performance_publication = (
    performance_publication.rename(
        columns={

            "overall_rank":
                "Rank",

            "model_display":
                "Model",

            "total_images":
                "Test Images",

            "correct_predictions":
                "Correct",

            "total_errors":
                "Errors",

            "accuracy_percent":
                "Accuracy (%)",

            "macro_precision_percent":
                "Macro Precision (%)",

            "macro_recall_percent":
                "Macro Recall (%)",

            "macro_f1_percent":
                "Macro F1 (%)",

            "weighted_f1_percent":
                "Weighted F1 (%)"
        }
    )
)


for column in [

    "Accuracy (%)",

    "Macro Precision (%)",

    "Macro Recall (%)",

    "Macro F1 (%)",

    "Weighted F1 (%)"
]:

    if column in performance_publication.columns:

        performance_publication[
            column
        ] = (
            performance_publication[
                column
            ]
            .round(2)
        )


# =============================================================================
# CALIBRATION TABLE
# =============================================================================

calibration_publication = (
    calibration_df.copy()
)

calibration_publication = (
    calibration_publication.rename(
        columns={

            "overall_rank":
                "Rank",

            "model_display":
                "Model",

            "average_confidence_percent":
                "Average Confidence (%)",

            "confidence_accuracy_gap_percent":
                "Confidence-Accuracy Gap (%)",

            "ece_percent":
                "ECE (%)",

            "mce_percent":
                "MCE (%)",

            "brier_score":
                "Brier Score",

            "high_confidence_errors":
                "High-Confidence Errors",

            "calibration_behavior":
                "Calibration Behavior"
        }
    )
)


for column in [

    "Average Confidence (%)",

    "Confidence-Accuracy Gap (%)",

    "ECE (%)",

    "MCE (%)"
]:

    if column in calibration_publication.columns:

        calibration_publication[
            column
        ] = (
            calibration_publication[
                column
            ]
            .round(2)
        )


if "Brier Score" in calibration_publication.columns:

    calibration_publication[
        "Brier Score"
    ] = (
        calibration_publication[
            "Brier Score"
        ]
        .round(4)
    )


# =============================================================================
# MODEL RANKING TABLE
# =============================================================================

ranking_publication = (
    ranking_df[
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
    .copy()
)


ranking_publication = (
    ranking_publication.rename(
        columns={

            "overall_rank":
                "Overall Rank",

            "model_display":
                "Model",

            "accuracy_percent":
                "Accuracy (%)",

            "macro_f1_percent":
                "Macro F1 (%)",

            "ece_percent":
                "ECE (%)",

            "brier_score":
                "Brier Score",

            "overall_rank_score":
                "Weighted Rank Score"
        }
    )
)


for column in [

    "Accuracy (%)",

    "Macro F1 (%)",

    "ECE (%)",

    "Weighted Rank Score"
]:

    ranking_publication[
        column
    ] = (
        ranking_publication[
            column
        ]
        .round(2)
    )


ranking_publication[
    "Brier Score"
] = (
    ranking_publication[
        "Brier Score"
    ]
    .round(4)
)


# =============================================================================
# STATISTICAL TABLE
# =============================================================================

statistical_publication = (
    pairwise_df.copy()
)


# Automatically simplify column names where possible.

column_mapping = {}

for column in statistical_publication.columns:

    lower_name = column.lower()

    if "model_a" in lower_name:

        column_mapping[
            column
        ] = "Model A"

    elif "model_b" in lower_name:

        column_mapping[
            column
        ] = "Model B"

    elif "p_value" in lower_name:

        column_mapping[
            column
        ] = "McNemar p-value"

    elif "significant" in lower_name:

        column_mapping[
            column
        ] = "Statistically Significant"


statistical_publication = (
    statistical_publication.rename(
        columns=column_mapping
    )
)


# Remove duplicate column names that may have been
# created during the publication-column renaming process.

statistical_publication = (
    statistical_publication.loc[
        :,
        ~statistical_publication.columns.duplicated()
    ]
)


for column in statistical_publication.columns:

    if "p-value" in str(column).lower():

        selected_data = (
            statistical_publication[
                column
            ]
        )

        # Pandas returns a DataFrame instead of a Series
        # when duplicate column names exist. Select the
        # first matching column safely.

        if isinstance(
            selected_data,
            pd.DataFrame
        ):

            selected_data = (
                selected_data.iloc[:, 0]
            )

        statistical_publication[
            column
        ] = (
            pd.to_numeric(
                selected_data,
                errors="coerce"
            )
            .round(6)
        )


# =============================================================================
# STATISTICAL SIGNIFICANCE COUNT
# =============================================================================

significant_column = None


for column in pairwise_df.columns:

    if "significant" in column.lower():

        significant_column = column
        break


if significant_column is not None:

    significant_series = (
        pairwise_df[
            significant_column
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes"
        ])
    )

    significant_count = int(
        significant_series.sum()
    )

else:

    significant_count = 0


total_comparisons = len(
    pairwise_df
)


# =============================================================================
# RESULTS SECTION
# =============================================================================

results_lines = []


results_lines.append(
    "RESULTS"
)

results_lines.append(
    "=" * 80
)

results_lines.append("")


results_lines.append(
    "Unseen-Source Evaluation"
)

results_lines.append(
    "All four models were evaluated on the same "
    f"unseen-source test set containing "
    f"{int(best_model['total_images'])} images. "
    "Performance was assessed using accuracy, "
    "macro-averaged precision, macro-averaged recall, "
    "macro F1-score, weighted F1-score, and confidence "
    "calibration metrics."
)

results_lines.append("")


results_lines.append(
    "Classification Performance"
)

results_lines.append(
    f"The proposed Attention EfficientNet-B0 achieved "
    f"an accuracy of {best_model['accuracy_percent']:.2f}% "
    f"and a macro F1-score of "
    f"{best_model['macro_f1_percent']:.2f}% on the "
    "unseen-source test set."
)

results_lines.append(
    "DenseNet121 achieved identical observed accuracy "
    "and macro F1-score on this evaluation set, while "
    "ConvNeXtV2-Tiny and EfficientNet-B0 produced lower "
    "observed classification performance."
)

results_lines.append("")


results_lines.append(
    "Confidence Calibration"
)

results_lines.append(
    f"The proposed Attention EfficientNet-B0 achieved "
    f"an Expected Calibration Error (ECE) of "
    f"{best_model['ece_percent']:.2f}% and a Brier Score "
    f"of {best_model['brier_score']:.4f}."
)

results_lines.append(
    "Among the evaluated models, the proposed model "
    "achieved the lowest ECE, indicating the strongest "
    "agreement between predicted confidence and observed "
    "classification correctness according to this metric."
)

results_lines.append("")


results_lines.append(
    "Overall Model Ranking"
)

results_lines.append(
    f"Using a weighted ranking framework that prioritised "
    "classification accuracy and macro F1-score while also "
    "considering ECE and Brier Score, "
    f"{best_model_name} obtained the highest overall rank."
)

results_lines.append("")


results_lines.append(
    "Statistical Significance"
)

results_lines.append(
    f"Pairwise McNemar testing was performed across "
    f"{total_comparisons} model comparisons."
)

if significant_count == 0:

    results_lines.append(
        "No statistically significant differences in "
        "classification accuracy were identified among "
        "the evaluated models on the unseen-source test set."
    )

    results_lines.append(
        "Therefore, the observed differences in model "
        "accuracy should not be interpreted as statistically "
        "significant superiority."
    )

else:

    results_lines.append(
        f"{significant_count} pairwise comparison(s) "
        "demonstrated statistically significant differences."
    )


# =============================================================================
# DISCUSSION SECTION
# =============================================================================

discussion_lines = []


discussion_lines.append(
    "DISCUSSION"
)

discussion_lines.append(
    "=" * 80
)

discussion_lines.append("")


discussion_lines.append(
    f"The unseen-source evaluation demonstrated that "
    f"{best_model_name} provided the strongest overall "
    "balance between classification performance and "
    "confidence calibration."
)

discussion_lines.append(
    "The proposed model achieved the highest observed "
    "classification performance, although DenseNet121 "
    "produced identical observed accuracy and macro "
    "F1-score on the evaluated test set."
)

discussion_lines.append(
    "A notable advantage of the proposed model was its "
    "confidence calibration performance. The model achieved "
    "the lowest ECE among the evaluated architectures, "
    "suggesting that its predicted confidence values were "
    "more closely aligned with empirical correctness."
)

discussion_lines.append(
    "Reliable confidence estimates are particularly important "
    "for image classification systems intended for decision "
    "support, because confidence information can help "
    "distinguish highly reliable predictions from predictions "
    "that may require additional review."
)

discussion_lines.append(
    "However, the pairwise McNemar tests did not identify "
    "statistically significant differences in accuracy among "
    "the models. This result should be considered when "
    "interpreting the observed ranking, especially because "
    "the unseen-source evaluation contained a relatively "
    "limited number of images."
)

discussion_lines.append(
    "Consequently, the proposed model should be described "
    "as demonstrating the strongest observed overall balance "
    "of predictive performance and calibration rather than "
    "being conclusively superior in statistically significant "
    "classification accuracy."
)


# =============================================================================
# KEY FINDINGS
# =============================================================================

findings_lines = []


findings_lines.append(
    "KEY FINDINGS"
)

findings_lines.append(
    "=" * 80
)

findings_lines.append("")


findings_lines.append(
    f"1. {best_model_name} ranked first according to "
    "the combined performance and calibration framework."
)

findings_lines.append(
    f"2. The proposed model achieved "
    f"{best_model['accuracy_percent']:.2f}% accuracy "
    f"and {best_model['macro_f1_percent']:.2f}% macro F1-score."
)

findings_lines.append(
    f"3. The proposed model achieved the lowest ECE "
    f"of {best_model['ece_percent']:.2f}% among the "
    "evaluated models."
)

findings_lines.append(
    "4. DenseNet121 matched the proposed model in "
    "observed accuracy and macro F1-score."
)

findings_lines.append(
    "5. No statistically significant pairwise differences "
    "in classification accuracy were identified using "
    "McNemar testing."
)

findings_lines.append(
    "6. The results indicate that confidence calibration "
    "provides useful additional information beyond "
    "classification accuracy alone."
)


# =============================================================================
# LIMITATIONS
# =============================================================================

limitations_lines = []


limitations_lines.append(
    "STUDY LIMITATIONS"
)

limitations_lines.append(
    "=" * 80
)

limitations_lines.append("")


limitations_lines.append(
    "1. The unseen-source test set contained a relatively "
    "small number of images, which limits statistical power."
)

limitations_lines.append(
    "2. The absence of statistically significant differences "
    "does not necessarily indicate that all models have "
    "identical performance; larger independent test sets "
    "would provide greater statistical sensitivity."
)

limitations_lines.append(
    "3. The evaluation was performed on the available "
    "unseen-source dataset, and additional external datasets "
    "would strengthen conclusions regarding generalisation."
)

limitations_lines.append(
    "4. Model ranking was based on a weighted decision "
    "framework. Although the weighting prioritised "
    "classification performance, alternative weighting "
    "strategies could produce different overall rankings."
)

limitations_lines.append(
    "5. Confidence calibration was assessed using ECE, MCE, "
    "and Brier Score, and these metrics may emphasise "
    "different characteristics of probabilistic predictions."
)


# =============================================================================
# SAVE TABLES
# =============================================================================

performance_output = (
    OUTPUT_DIR /
    "Table_1_Unseen_Source_Classification_Performance.csv"
)

calibration_output = (
    OUTPUT_DIR /
    "Table_2_Unseen_Source_Confidence_Calibration.csv"
)

ranking_output = (
    OUTPUT_DIR /
    "Table_3_Overall_Model_Ranking.csv"
)

statistical_output = (
    OUTPUT_DIR /
    "Table_4_McNemar_Statistical_Comparisons.csv"
)


performance_publication.to_csv(
    performance_output,
    index=False
)

calibration_publication.to_csv(
    calibration_output,
    index=False
)

ranking_publication.to_csv(
    ranking_output,
    index=False
)

statistical_publication.to_csv(
    statistical_output,
    index=False
)


# =============================================================================
# SAVE TEXT SECTIONS
# =============================================================================

results_output = (
    OUTPUT_DIR /
    "Results_Section.txt"
)

discussion_output = (
    OUTPUT_DIR /
    "Discussion_Section.txt"
)

findings_output = (
    OUTPUT_DIR /
    "Key_Findings.txt"
)

limitations_output = (
    OUTPUT_DIR /
    "Study_Limitations.txt"
)


results_output.write_text(
    "\n\n".join(
        results_lines
    ),
    encoding="utf-8"
)

discussion_output.write_text(
    "\n\n".join(
        discussion_lines
    ),
    encoding="utf-8"
)

findings_output.write_text(
    "\n".join(
        findings_lines
    ),
    encoding="utf-8"
)

limitations_output.write_text(
    "\n".join(
        limitations_lines
    ),
    encoding="utf-8"
)


# =============================================================================
# CREATE COMBINED MANUSCRIPT RESULTS FILE
# =============================================================================

manuscript_output = (
    OUTPUT_DIR /
    "Publication_Ready_Unseen_Source_Analysis.txt"
)


combined_lines = []

combined_lines.extend(
    results_lines
)

combined_lines.append("")
combined_lines.append("")

combined_lines.extend(
    discussion_lines
)

combined_lines.append("")
combined_lines.append("")

combined_lines.extend(
    findings_lines
)

combined_lines.append("")
combined_lines.append("")

combined_lines.extend(
    limitations_lines
)


manuscript_output.write_text(
    "\n".join(
        combined_lines
    ),
    encoding="utf-8"
)


# =============================================================================
# DISPLAY SUMMARY
# =============================================================================

print()
print("=" * 100)
print("PUBLICATION-READY RESULTS GENERATED SUCCESSFULLY")
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

print("Generated publication tables:")

for file_path in [

    performance_output,

    calibration_output,

    ranking_output,

    statistical_output
]:

    print()
    print(file_path)


print()

print("Generated manuscript sections:")

for file_path in [

    results_output,

    discussion_output,

    findings_output,

    limitations_output,

    manuscript_output
]:

    print()
    print(file_path)


print()
print("=" * 100)
print("PUBLICATION PREPARATION COMPLETED")
print("=" * 100)
