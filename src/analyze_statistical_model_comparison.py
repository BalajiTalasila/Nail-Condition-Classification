from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


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
    "statistical_model_comparison"
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


print("=" * 80)
print("STATISTICAL MODEL COMPARISON")
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


    prediction_dataframes[
        model_name
    ] = dataframe


    print(
        f"\nLoaded {model_name}: "
        f"{len(dataframe)} images"
    )


# ============================================================
# CALCULATE ACCURACY FOR EACH MODEL
# ============================================================

print("\n" + "=" * 80)
print("MODEL ACCURACY SUMMARY")
print("=" * 80)


accuracy_results = []


for model_name, dataframe in (
    prediction_dataframes.items()
):

    accuracy = (

        dataframe[
            "true_class"
        ]

        ==

        dataframe[
            "predicted_class"
        ]

    ).mean()


    accuracy_results.append({

        "model":
        model_name,

        "total_images":
        len(dataframe),

        "accuracy":
        accuracy,

        "accuracy_percent":
        accuracy * 100
    })


accuracy_dataframe = pd.DataFrame(
    accuracy_results
)


accuracy_dataframe = (
    accuracy_dataframe
    .sort_values(
        "accuracy",
        ascending=False
    )
)


print("\n")

print(
    accuracy_dataframe.to_string(
        index=False
    )
)


# ============================================================
# PAIRED COMPARISON FUNCTION
# ============================================================

def compare_models(
    baseline_name,
    proposed_name
):

    baseline_dataframe = (
        prediction_dataframes[
            baseline_name
        ]
        .copy()
    )


    proposed_dataframe = (
        prediction_dataframes[
            proposed_name
        ]
        .copy()
    )


    merged_dataframe = pd.DataFrame({

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

    merged_dataframe[
        "baseline_correct"
    ] = (

        merged_dataframe[
            "baseline_prediction"
        ]

        ==

        merged_dataframe[
            "true_class"
        ]
    )


    merged_dataframe[
        "proposed_correct"
    ] = (

        merged_dataframe[
            "proposed_prediction"
        ]

        ==

        merged_dataframe[
            "true_class"
        ]
    )


    # --------------------------------------------------------
    # CONFIDENCE DIFFERENCE
    # --------------------------------------------------------

    merged_dataframe[
        "confidence_difference"
    ] = (

        merged_dataframe[
            "proposed_confidence"
        ]

        -

        merged_dataframe[
            "baseline_confidence"
        ]
    )


    confidence_difference = (
        merged_dataframe[
            "confidence_difference"
        ]
    )


    # --------------------------------------------------------
    # PREDICTION AGREEMENT
    # --------------------------------------------------------

    same_predictions = (

        merged_dataframe[
            "baseline_prediction"
        ]

        ==

        merged_dataframe[
            "proposed_prediction"
        ]
    )


    prediction_agreement_count = (
        same_predictions.sum()
    )


    prediction_agreement_percent = (

        prediction_agreement_count

        /

        len(merged_dataframe)

        *

        100
    )


    # --------------------------------------------------------
    # CORRECTNESS CONTINGENCY TABLE
    # --------------------------------------------------------

    both_correct = (

        merged_dataframe[
            "baseline_correct"
        ]

        &

        merged_dataframe[
            "proposed_correct"
        ]

    ).sum()


    baseline_correct_proposed_wrong = (

        merged_dataframe[
            "baseline_correct"
        ]

        &

        ~

        merged_dataframe[
            "proposed_correct"
        ]

    ).sum()


    baseline_wrong_proposed_correct = (

        ~

        merged_dataframe[
            "baseline_correct"
        ]

        &

        merged_dataframe[
            "proposed_correct"
        ]

    ).sum()


    both_wrong = (

        ~

        merged_dataframe[
            "baseline_correct"
        ]

        &

        ~

        merged_dataframe[
            "proposed_correct"
        ]

    ).sum()


    # --------------------------------------------------------
    # WILCOXON SIGNED-RANK TEST
    # --------------------------------------------------------

    non_zero_differences = (
        confidence_difference[
            confidence_difference != 0
        ]
    )


    if len(non_zero_differences) > 0:

        try:

            wilcoxon_statistic, wilcoxon_p_value = (
                wilcoxon(
                    merged_dataframe[
                        "proposed_confidence"
                    ],

                    merged_dataframe[
                        "baseline_confidence"
                    ],

                    alternative="two-sided"
                )
            )

        except ValueError:

            wilcoxon_statistic = np.nan

            wilcoxon_p_value = np.nan

    else:

        wilcoxon_statistic = np.nan

        wilcoxon_p_value = np.nan


    # --------------------------------------------------------
    # EFFECT SIZE
    # --------------------------------------------------------

    mean_difference = (
        confidence_difference.mean()
    )


    median_difference = (
        confidence_difference.median()
    )


    standard_deviation_difference = (
        confidence_difference.std()
    )


    if (
        standard_deviation_difference > 0
    ):

        cohens_d = (

            mean_difference

            /

            standard_deviation_difference
        )

    else:

        cohens_d = np.nan


    # --------------------------------------------------------
    # MORE CONFIDENT COUNTS
    # --------------------------------------------------------

    proposed_more_confident = (

        confidence_difference > 0

    ).sum()


    baseline_more_confident = (

        confidence_difference < 0

    ).sum()


    equal_confidence = (

        confidence_difference == 0

    ).sum()


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    baseline_accuracy = (
        merged_dataframe[
            "baseline_correct"
        ].mean()
    )


    proposed_accuracy = (
        merged_dataframe[
            "proposed_correct"
        ].mean()
    )


    accuracy_difference = (

        proposed_accuracy

        -

        baseline_accuracy
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    comparison_result = {

        "baseline_model":
        baseline_name,

        "proposed_model":
        proposed_name,

        "total_images":
        len(merged_dataframe),

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
        prediction_agreement_count,

        "prediction_agreement_percent":
        prediction_agreement_percent,

        "both_correct":
        both_correct,

        "baseline_correct_proposed_wrong":
        baseline_correct_proposed_wrong,

        "baseline_wrong_proposed_correct":
        baseline_wrong_proposed_correct,

        "both_wrong":
        both_wrong,

        "baseline_average_confidence":
        merged_dataframe[
            "baseline_confidence"
        ].mean(),

        "proposed_average_confidence":
        merged_dataframe[
            "proposed_confidence"
        ].mean(),

        "mean_confidence_difference":
        mean_difference,

        "median_confidence_difference":
        median_difference,

        "standard_deviation_difference":
        standard_deviation_difference,

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

        "cohens_d":
        cohens_d
    }


    return (
        comparison_result,
        merged_dataframe
    )


# ============================================================
# COMPARE PROPOSED MODEL WITH BASELINES
# ============================================================

print("\n" + "=" * 80)
print("PROPOSED MODEL VS BASELINE MODELS")
print("=" * 80)


comparison_results = []


detailed_comparisons = {}


baseline_models = [

    model_name

    for model_name in MODEL_NAMES

    if model_name != PROPOSED_MODEL
]


for baseline_model in baseline_models:

    print("\n" + "-" * 80)

    print(
        f"COMPARISON: "
        f"{PROPOSED_MODEL} "
        f"VS "
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


    detailed_comparisons[
        baseline_model
    ] = detailed_dataframe


    print(
        f"\nBaseline accuracy: "
        f"{result['baseline_accuracy_percent']:.2f}%"
    )


    print(
        f"Proposed accuracy: "
        f"{result['proposed_accuracy_percent']:.2f}%"
    )


    print(
        f"Accuracy difference: "
        f"{result['accuracy_difference_percent']:.2f}%"
    )


    print(
        f"\nSame predictions: "
        f"{result['same_predictions']} / "
        f"{result['total_images']}"
    )


    print(
        f"Prediction agreement: "
        f"{result['prediction_agreement_percent']:.2f}%"
    )


    print(
        f"\nBaseline correct / Proposed wrong: "
        f"{result['baseline_correct_proposed_wrong']}"
    )


    print(
        f"Baseline wrong / Proposed correct: "
        f"{result['baseline_wrong_proposed_correct']}"
    )


    print(
        f"\nBaseline average confidence: "
        f"{result['baseline_average_confidence'] * 100:.2f}%"
    )


    print(
        f"Proposed average confidence: "
        f"{result['proposed_average_confidence'] * 100:.2f}%"
    )


    print(
        f"Mean confidence difference: "
        f"{result['mean_confidence_difference'] * 100:.2f}%"
    )


    print(
        f"Median confidence difference: "
        f"{result['median_confidence_difference'] * 100:.2f}%"
    )


    print(
        f"\nProposed more confident: "
        f"{result['proposed_more_confident']}"
    )


    print(
        f"Baseline more confident: "
        f"{result['baseline_more_confident']}"
    )


    print(
        f"Equal confidence: "
        f"{result['equal_confidence']}"
    )


    print(
        f"\nWilcoxon statistic: "
        f"{result['wilcoxon_statistic']}"
    )


    print(
        f"Wilcoxon p-value: "
        f"{result['wilcoxon_p_value']}"
    )


    print(
        f"Cohen's d: "
        f"{result['cohens_d']:.4f}"
    )


    if (
        pd.notna(
            result[
                "wilcoxon_p_value"
            ]
        )
    ):

        if (
            result[
                "wilcoxon_p_value"
            ]

            <

            0.05
        ):

            print(
                "\nResult: "
                "Statistically significant "
                "confidence difference "
                "(p < 0.05)"
            )

        else:

            print(
                "\nResult: "
                "No statistically significant "
                "confidence difference "
                "(p >= 0.05)"
            )


# ============================================================
# CREATE FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL STATISTICAL COMPARISON SUMMARY")
print("=" * 80)


comparison_summary_dataframe = pd.DataFrame(
    comparison_results
)


print("\n")

print(
    comparison_summary_dataframe.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

summary_file = (

    OUTPUT_DIR /

    "proposed_vs_baseline_statistical_comparison.csv"
)


comparison_summary_dataframe.to_csv(

    summary_file,

    index=False
)


print("\n" + "=" * 80)
print("SAVING DETAILED COMPARISONS")
print("=" * 80)


for baseline_model, detailed_dataframe in (
    detailed_comparisons.items()
):

    detailed_file = (

        OUTPUT_DIR /

        f"proposed_vs_{baseline_model}_detailed.csv"
    )


    detailed_dataframe.to_csv(

        detailed_file,

        index=False
    )


    print(
        detailed_file
    )


print("\n" + "=" * 80)
print("SUMMARY FILE")
print("=" * 80)


print(
    summary_file
)


print("\n" + "=" * 80)
print("STATISTICAL MODEL COMPARISON COMPLETED")
print("=" * 80)
