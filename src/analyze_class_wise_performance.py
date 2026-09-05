from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
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
    "class_wise_performance_analysis"
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
print("CLASS-WISE MODEL PERFORMANCE ANALYSIS")
print("=" * 80)


all_model_results = []

overall_summary = []


for model_name in MODEL_NAMES:

    print("\n" + "=" * 80)

    print(
        f"ANALYZING: "
        f"{MODEL_DISPLAY_NAMES[model_name]}"
    )

    print("=" * 80)


    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        print(
            "\nWARNING: Prediction file not found:"
        )

        print(
            prediction_file
        )

        continue


    dataframe = pd.read_csv(
        prediction_file
    )


    print(
        f"\nTotal images: "
        f"{len(dataframe)}"
    )


    required_columns = [

        "true_class",

        "predicted_class"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in dataframe.columns

    ]


    if missing_columns:

        raise RuntimeError(

            f"Missing required columns for "
            f"{model_name}: "
            f"{missing_columns}"

        )


    true_labels = dataframe[
        "true_class"
    ]


    predicted_labels = dataframe[
        "predicted_class"
    ]


    class_names = sorted(
        true_labels.unique()
    )


    # ============================================================
    # CLASS-WISE METRICS
    # ============================================================

    precision, recall, f1_score, support = (

        precision_recall_fscore_support(

            true_labels,

            predicted_labels,

            labels=class_names,

            zero_division=0

        )

    )


    class_results = []


    for index, class_name in enumerate(
        class_names
    ):

        class_mask = (
            true_labels == class_name
        )


        class_total = int(
            class_mask.sum()
        )


        class_correct = int(

            (
                predicted_labels[
                    class_mask
                ]

                ==

                true_labels[
                    class_mask
                ]

            ).sum()

        )


        class_incorrect = (

            class_total

            -

            class_correct

        )


        class_accuracy = (

            class_correct

            /

            class_total

        )


        class_results.append({

            "model":
            model_name,

            "model_display":
            MODEL_DISPLAY_NAMES[
                model_name
            ],

            "class":
            class_name,

            "support":
            class_total,

            "correct_predictions":
            class_correct,

            "incorrect_predictions":
            class_incorrect,

            "class_accuracy":
            class_accuracy,

            "class_accuracy_percent":
            class_accuracy * 100,

            "precision":
            precision[index],

            "precision_percent":
            precision[index] * 100,

            "recall":
            recall[index],

            "recall_percent":
            recall[index] * 100,

            "f1_score":
            f1_score[index],

            "f1_score_percent":
            f1_score[index] * 100

        })


    class_dataframe = pd.DataFrame(
        class_results
    )


    all_model_results.append(
        class_dataframe
    )


    print(
        "\nCLASS-WISE PERFORMANCE:"
    )


    print(

        class_dataframe[

            [

                "class",

                "support",

                "correct_predictions",

                "incorrect_predictions",

                "precision_percent",

                "recall_percent",

                "f1_score_percent"

            ]

        ].to_string(

            index=False

        )

    )


    # ============================================================
    # OVERALL / MACRO / WEIGHTED METRICS
    # ============================================================

    macro_precision, macro_recall, macro_f1, _ = (

        precision_recall_fscore_support(

            true_labels,

            predicted_labels,

            average="macro",

            zero_division=0

        )

    )


    weighted_precision, weighted_recall, weighted_f1, _ = (

        precision_recall_fscore_support(

            true_labels,

            predicted_labels,

            average="weighted",

            zero_division=0

        )

    )


    accuracy = accuracy_score(

        true_labels,

        predicted_labels

    )


    overall_summary.append({

        "model":
        model_name,

        "model_display":
        MODEL_DISPLAY_NAMES[
            model_name
        ],

        "total_images":
        len(dataframe),

        "accuracy":
        accuracy,

        "accuracy_percent":
        accuracy * 100,

        "macro_precision":
        macro_precision,

        "macro_precision_percent":
        macro_precision * 100,

        "macro_recall":
        macro_recall,

        "macro_recall_percent":
        macro_recall * 100,

        "macro_f1":
        macro_f1,

        "macro_f1_percent":
        macro_f1 * 100,

        "weighted_precision":
        weighted_precision,

        "weighted_precision_percent":
        weighted_precision * 100,

        "weighted_recall":
        weighted_recall,

        "weighted_recall_percent":
        weighted_recall * 100,

        "weighted_f1":
        weighted_f1,

        "weighted_f1_percent":
        weighted_f1 * 100

    })


# ============================================================
# COMBINE ALL CLASS RESULTS
# ============================================================

combined_class_dataframe = pd.concat(

    all_model_results,

    ignore_index=True

)


overall_summary_dataframe = pd.DataFrame(
    overall_summary
)


# ============================================================
# CREATE CLASS COMPARISON TABLE
# ============================================================

comparison_dataframe = combined_class_dataframe.pivot(

    index="class",

    columns="model_display",

    values="f1_score_percent"

)


comparison_dataframe = comparison_dataframe.round(
    2
)


print("\n" + "=" * 80)
print("CLASS-WISE F1-SCORE COMPARISON")
print("=" * 80)


print(
    comparison_dataframe.to_string()
)


# ============================================================
# FINAL OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OVERALL MACRO AND WEIGHTED PERFORMANCE")
print("=" * 80)


summary_display_columns = [

    "model_display",

    "accuracy_percent",

    "macro_precision_percent",

    "macro_recall_percent",

    "macro_f1_percent",

    "weighted_f1_percent"

]


print(

    overall_summary_dataframe[

        summary_display_columns

    ].to_string(

        index=False

    )

)


# ============================================================
# SAVE RESULTS
# ============================================================

class_results_file = (

    OUTPUT_DIR /

    "class_wise_performance_all_models.csv"

)


overall_summary_file = (

    OUTPUT_DIR /

    "overall_macro_weighted_performance.csv"

)


f1_comparison_file = (

    OUTPUT_DIR /

    "class_wise_f1_comparison.csv"

)


combined_class_dataframe.to_csv(

    class_results_file,

    index=False

)


overall_summary_dataframe.to_csv(

    overall_summary_file,

    index=False

)


comparison_dataframe.to_csv(

    f1_comparison_file

)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    class_results_file
)


print(
    overall_summary_file
)


print(
    f1_comparison_file
)


print("\n" + "=" * 80)
print("CLASS-WISE PERFORMANCE ANALYSIS COMPLETED")
print("=" * 80)

