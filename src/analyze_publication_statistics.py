from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    binomtest,
    wilcoxon
)


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
    "publication_statistical_analysis"
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


PROPOSED_MODEL = (
    "proposed_attention_efficientnet_b0"
)


SIGNIFICANCE_LEVEL = 0.05


print("=" * 80)
print("PUBLICATION-QUALITY STATISTICAL ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD PREDICTION FILES
# ============================================================

prediction_dataframes = {}


for model_name in MODEL_NAMES:

    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{prediction_file}"
        )


    dataframe = pd.read_csv(
        prediction_file
    )


    required_columns = [

        "image_path",

        "true_class",

        "predicted_class",

        "confidence"
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in dataframe.columns
    ]


    if missing_columns:

        raise RuntimeError(
            f"{model_name} is missing columns: "
            f"{missing_columns}"
        )


    prediction_dataframes[
        model_name
    ] = dataframe


    print(
        f"\nLoaded {model_name}: "
        f"{len(dataframe)} images"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def significance_label(
    p_value
):

    if pd.isna(p_value):

        return "Not available"


    if p_value < 0.001:

        return "*** (p < 0.001)"


    elif p_value < 0.01:

        return "** (p < 0.01)"


    elif p_value < 0.05:

        return "* (p < 0.05)"


    else:

        return "Not significant"


def interpret_effect_size(
    value
):

    if pd.isna(value):

        return "Not available"


    absolute_value = abs(
        value
    )


    if absolute_value < 0.20:

        return "Negligible"


    elif absolute_value < 0.50:

        return "Small"


    elif absolute_value < 0.80:

        return "Medium"


    else:

        return "Large"


# ============================================================
# EXACT MCNEMAR TEST
# ============================================================

def exact_mcnemar_test(
    baseline_correct_proposed_wrong,
    baseline_wrong_proposed_correct
):

    discordant_pairs = (

        baseline_correct_proposed_wrong

        +

        baseline_wrong_proposed_correct
    )


    if discordant_pairs == 0:

        return (
            np.nan,
            "No discordant pairs"
        )


    # Under the null hypothesis, each discordant
    # pair has equal probability of favoring either model.

    result = binomtest(

        baseline_wrong_proposed_correct,

        n=discordant_pairs,

        p=0.5,

        alternative="two-sided"
    )


    return (

        result.pvalue,

        significance_label(
            result.pvalue
        )
    )


# ============================================================
# WILCOXON EFFECT SIZE
# ============================================================

def calculate_rank_biserial(
    differences
):

    non_zero_differences = (
        differences[
            differences != 0
        ]
    )


    if len(non_zero_differences) == 0:

        return np.nan


    positive_count = (
        non_zero_differences > 0
    ).sum()


    negative_count = (
        non_zero_differences < 0
    ).sum()


    rank_biserial = (

        positive_count

        -

        negative_count

    ) / len(
        non_zero_differences
    )


    return rank_biserial


def calculate_cohens_dz(
    differences
):

    if len(differences) < 2:

        return np.nan


    standard_deviation = (
        differences.std(
            ddof=1
        )
    )


    if standard_deviation == 0:

        return np.nan


    cohens_dz = (

        differences.mean()

        /

        standard_deviation
    )


    return cohens_dz


# ============================================================
# PAIRED MODEL COMPARISON
# ============================================================

def compare_models(
    baseline_name,
    proposed_name
):

    baseline_dataframe = (
        prediction_dataframes[
            baseline_name
        ].copy()
    )


    proposed_dataframe = (
        prediction_dataframes[
            proposed_name
        ].copy()
    )


    # --------------------------------------------------------
    # VERIFY IMAGE ALIGNMENT
    # --------------------------------------------------------

    if not np.array_equal(

        baseline_dataframe[
            "image_path"
        ].to_numpy(),

        proposed_dataframe[
            "image_path"
        ].to_numpy()
    ):

        raise RuntimeError(
            f"Image order mismatch between "
            f"{baseline_name} and "
            f"{proposed_name}"
        )


    comparison_dataframe = pd.DataFrame({

        "image_path":
        baseline_dataframe[
            "image_path"
        ],

        "true_class":
        baseline_dataframe[
            "true_class"
        ],

        "baseline_prediction":
        baseline_dataframe[
            "predicted_class"
        ],

        "proposed_prediction":
        proposed_dataframe[
            "predicted_class"
        ],

        "baseline_confidence":
        baseline_dataframe[
            "confidence"
        ],

        "proposed_confidence":
        proposed_dataframe[
            "confidence"
        ]
    })


    # --------------------------------------------------------
    # CORRECTNESS
    # --------------------------------------------------------

    comparison_dataframe[
        "baseline_correct"
    ] = (

        comparison_dataframe[
            "baseline_prediction"
        ]

        ==

        comparison_dataframe[
            "true_class"
        ]
    )


    comparison_dataframe[
        "proposed_correct"
    ] = (

        comparison_dataframe[
            "proposed_prediction"
        ]

        ==

        comparison_dataframe[
            "true_class"
        ]
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    baseline_accuracy = (
        comparison_dataframe[
            "baseline_correct"
        ].mean()
    )


    proposed_accuracy = (
        comparison_dataframe[
            "proposed_correct"
        ].mean()
    )


    accuracy_difference = (

        proposed_accuracy

        -

        baseline_accuracy
    )


    # --------------------------------------------------------
    # PREDICTION AGREEMENT
    # --------------------------------------------------------

    comparison_dataframe[
        "same_prediction"
    ] = (

        comparison_dataframe[
            "baseline_prediction"
        ]

        ==

        comparison_dataframe[
            "proposed_prediction"
        ]
    )


    same_predictions = (
        comparison_dataframe[
            "same_prediction"
        ].sum()
    )


    # --------------------------------------------------------
    # MCNEMAR CONTINGENCY VALUES
    # --------------------------------------------------------

    both_correct = (

        comparison_dataframe[
            "baseline_correct"
        ]

        &

        comparison_dataframe[
            "proposed_correct"
        ]

    ).sum()


    baseline_correct_proposed_wrong = (

        comparison_dataframe[
            "baseline_correct"
        ]

        &

        ~

        comparison_dataframe[
            "proposed_correct"
        ]

    ).sum()


    baseline_wrong_proposed_correct = (

        ~

        comparison_dataframe[
            "baseline_correct"
        ]

        &

        comparison_dataframe[
            "proposed_correct"
        ]

    ).sum()


    both_wrong = (

        ~

        comparison_dataframe[
            "baseline_correct"
        ]

        &

        ~

        comparison_dataframe[
            "proposed_correct"
        ]

    ).sum()


    # --------------------------------------------------------
    # EXACT MCNEMAR TEST
    # --------------------------------------------------------

    mcnemar_p_value, mcnemar_result = (
        exact_mcnemar_test(

            baseline_correct_proposed_wrong,

            baseline_wrong_proposed_correct
        )
    )


    # --------------------------------------------------------
    # CONFIDENCE DIFFERENCES
    # --------------------------------------------------------

    comparison_dataframe[
        "confidence_difference"
    ] = (

        comparison_dataframe[
            "proposed_confidence"
        ]

        -

        comparison_dataframe[
            "baseline_confidence"
        ]
    )


    confidence_differences = (
        comparison_dataframe[
            "confidence_difference"
        ]
    )


    # --------------------------------------------------------
    # WILCOXON SIGNED-RANK TEST
    # --------------------------------------------------------

    non_zero_differences = (
        confidence_differences[
            confidence_differences != 0
        ]
    )


    if len(non_zero_differences) > 0:

        try:

            (
                wilcoxon_statistic,
                wilcoxon_p_value
            ) = wilcoxon(

                comparison_dataframe[
                    "proposed_confidence"
                ],

                comparison_dataframe[
                    "baseline_confidence"
                ],

                alternative="two-sided"
            )


        except ValueError:

            wilcoxon_statistic = np.nan

            wilcoxon_p_value = np.nan


    else:

        wilcoxon_statistic = np.nan

        wilcoxon_p_value = np.nan


    # --------------------------------------------------------
    # EFFECT SIZES
    # --------------------------------------------------------

    cohens_dz = (
        calculate_cohens_dz(
            confidence_differences
        )
    )


    rank_biserial = (
        calculate_rank_biserial(
            confidence_differences
        )
    )


    # --------------------------------------------------------
    # CONFIDENCE COUNTS
    # --------------------------------------------------------

    proposed_more_confident = (

        confidence_differences > 0

    ).sum()


    baseline_more_confident = (

        confidence_differences < 0

    ).sum()


    equal_confidence = (

        confidence_differences == 0

    ).sum()


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "baseline_model":
        baseline_name,

        "proposed_model":
        proposed_name,

        "total_images":
        len(comparison_dataframe),

        "baseline_accuracy":
        baseline_accuracy,

        "proposed_accuracy":
        proposed_accuracy,

        "accuracy_difference":
        accuracy_difference,

        "baseline_accuracy_percent":
        baseline_accuracy * 100,

        "proposed_accuracy_percent":
        proposed_accuracy * 100,

        "accuracy_difference_percent":
        accuracy_difference * 100,

        "same_predictions":
        same_predictions,

        "prediction_agreement_percent":
        (

            same_predictions

            /

            len(comparison_dataframe)

            *

            100
        ),

        "both_correct":
        both_correct,

        "baseline_correct_proposed_wrong":
        baseline_correct_proposed_wrong,

        "baseline_wrong_proposed_correct":
        baseline_wrong_proposed_correct,

        "both_wrong":
        both_wrong,

        "mcnemar_p_value":
        mcnemar_p_value,

        "mcnemar_significance":
        mcnemar_result,

        "baseline_average_confidence":
        comparison_dataframe[
            "baseline_confidence"
        ].mean(),

        "proposed_average_confidence":
        comparison_dataframe[
            "proposed_confidence"
        ].mean(),

        "mean_confidence_difference":
        confidence_differences.mean(),

        "median_confidence_difference":
        confidence_differences.median(),

        "proposed_more_confident":
        proposed_more_confident,

        "baseline_more_confident":
        baseline_more_confident,

        "equal_confidence":
        equal_confidence,

        "wilcoxon_statistic":
        wilcoxon_statistic,

        "wilcoxon_p_value":
        wilcoxon_p_value,

        "wilcoxon_significance":
        significance_label(
            wilcoxon_p_value
        ),

        "rank_biserial_correlation":
        rank_biserial,

        "rank_biserial_interpretation":
        interpret_effect_size(
            rank_biserial
        ),

        "cohens_dz":
        cohens_dz,

        "cohens_dz_interpretation":
        interpret_effect_size(
            cohens_dz
        )
    }


    return (

        result,

        comparison_dataframe
    )


# ============================================================
# RUN ALL COMPARISONS
# ============================================================

print("\n" + "=" * 80)
print("PROPOSED MODEL VS BASELINES")
print("=" * 80)


comparison_results = []


detailed_results = {}


for baseline_model in MODEL_NAMES:

    if baseline_model == PROPOSED_MODEL:

        continue


    print("\n" + "-" * 80)

    print(
        f"COMPARING PROPOSED MODEL WITH "
        f"{baseline_model}"
    )

    print("-" * 80)


    result, detailed_dataframe = (
        compare_models(

            baseline_model,

            PROPOSED_MODEL
        )
    )


    comparison_results.append(
        result
    )


    detailed_results[
        baseline_model
    ] = detailed_dataframe


    print(
        f"\nAccuracy:"
    )


    print(
        f"Baseline: "
        f"{result['baseline_accuracy_percent']:.2f}%"
    )


    print(
        f"Proposed: "
        f"{result['proposed_accuracy_percent']:.2f}%"
    )


    print(
        f"Difference: "
        f"{result['accuracy_difference_percent']:.2f}%"
    )


    print(
        f"\nMcNemar Test:"
    )


    print(
        f"Baseline correct / Proposed wrong: "
        f"{result['baseline_correct_proposed_wrong']}"
    )


    print(
        f"Baseline wrong / Proposed correct: "
        f"{result['baseline_wrong_proposed_correct']}"
    )


    print(
        f"p-value: "
        f"{result['mcnemar_p_value']}"
    )


    print(
        f"Result: "
        f"{result['mcnemar_significance']}"
    )


    print(
        f"\nConfidence Comparison:"
    )


    print(
        f"Mean difference: "
        f"{result['mean_confidence_difference'] * 100:.2f}%"
    )


    print(
        f"Median difference: "
        f"{result['median_confidence_difference'] * 100:.2f}%"
    )


    print(
        f"Proposed more confident: "
        f"{result['proposed_more_confident']}"
    )


    print(
        f"Baseline more confident: "
        f"{result['baseline_more_confident']}"
    )


    print(
        f"\nWilcoxon Test:"
    )


    print(
        f"Statistic: "
        f"{result['wilcoxon_statistic']}"
    )


    print(
        f"p-value: "
        f"{result['wilcoxon_p_value']}"
    )


    print(
        f"Result: "
        f"{result['wilcoxon_significance']}"
    )


    print(
        f"\nEffect Sizes:"
    )


    print(
        f"Rank-biserial correlation: "
        f"{result['rank_biserial_correlation']:.4f}"
    )


    print(
        f"Interpretation: "
        f"{result['rank_biserial_interpretation']}"
    )


    print(
        f"Cohen's dz: "
        f"{result['cohens_dz']:.4f}"
    )


    print(
        f"Interpretation: "
        f"{result['cohens_dz_interpretation']}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_dataframe = pd.DataFrame(
    comparison_results
)


print("\n" + "=" * 80)
print("FINAL PUBLICATION STATISTICAL SUMMARY")
print("=" * 80)


print("\n")

print(
    summary_dataframe.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

summary_file = (

    OUTPUT_DIR /

    "publication_statistical_summary.csv"
)


summary_dataframe.to_csv(

    summary_file,

    index=False
)


for baseline_model, dataframe in (
    detailed_results.items()
):

    detailed_file = (

        OUTPUT_DIR /

        f"proposed_vs_{baseline_model}_paired_analysis.csv"
    )


    dataframe.to_csv(

        detailed_file,

        index=False
    )


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    summary_file
)


for baseline_model in detailed_results:

    print(

        OUTPUT_DIR /

        f"proposed_vs_{baseline_model}_paired_analysis.csv"
    )


print("\n" + "=" * 80)
print("PUBLICATION STATISTICAL ANALYSIS COMPLETED")
print("=" * 80)
