import pandas as pd
import numpy as np

from pathlib import Path

from itertools import combinations

from scipy.stats import binomtest

import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path.cwd()

PREDICTION_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_detailed_results"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_statistical_analysis"
)

FIGURES_DIR = (
    PROJECT_ROOT /
    "results" /
    "figures" /
    "unseen_source_statistical_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
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
# LOAD AND STANDARDIZE PREDICTIONS
# =============================================================================

def load_predictions(
    model_name
):

    file_path = (
        PREDICTION_DIR /
        f"{model_name}_image_predictions.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{file_path}"
        )


    df = pd.read_csv(
        file_path
    )


    required_columns = [
        "image_path",
        "true_class",
        "predicted_class"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing required columns for "
            f"{model_name}: "
            f"{missing_columns}"
        )


    df = df.copy()


    df["correct"] = (
        df["true_class"]
        ==
        df["predicted_class"]
    )


    df = df[
        [
            "image_path",
            "true_class",
            "predicted_class",
            "correct"
        ]
    ]


    df = df.rename(
        columns={
            "predicted_class":
                f"{model_name}_prediction",

            "correct":
                f"{model_name}_correct"
        }
    )


    return df


# =============================================================================
# LOAD ALL MODELS
# =============================================================================

print("=" * 100)
print("UNSEEN-SOURCE STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 100)

print()
print(
    "Loading predictions from all models..."
)


prediction_data = {}


for model_name in MODELS:

    df = load_predictions(
        model_name
    )

    prediction_data[
        model_name
    ] = df


    print()
    print(
        MODELS[
            model_name
        ]["display"]
    )

    print(
        f"Images loaded: "
        f"{len(df)}"
    )


# =============================================================================
# VERIFY IMAGE ALIGNMENT
# =============================================================================

reference_model = (
    list(MODELS.keys())[0]
)

reference_images = set(
    prediction_data[
        reference_model
    ]["image_path"]
)


for model_name, df in (
    prediction_data.items()
):

    current_images = set(
        df["image_path"]
    )


    if current_images != reference_images:

        missing = (
            reference_images -
            current_images
        )

        extra = (
            current_images -
            reference_images
        )


        raise ValueError(
            f"Image mismatch detected for "
            f"{model_name}\n"
            f"Missing: {len(missing)}\n"
            f"Extra: {len(extra)}"
        )


print()
print(
    "SUCCESS: All models were evaluated "
    "on the same image set."
)


# =============================================================================
# CREATE UNIFIED CORRECTNESS DATAFRAME
# =============================================================================

combined_df = (
    prediction_data[
        reference_model
    ][
        [
            "image_path",
            "true_class"
        ]
    ]
    .copy()
)


for model_name, df in (
    prediction_data.items()
):

    combined_df = (
        combined_df.merge(
            df[
                [
                    "image_path",
                    f"{model_name}_correct"
                ]
            ],
            on="image_path",
            how="inner"
        )
    )


print()
print(
    f"Total aligned images: "
    f"{len(combined_df)}"
)


# =============================================================================
# MCCNEMAR EXACT TEST
# =============================================================================

def perform_mcnemar_test(
    model_a,
    model_b
):

    correct_a = (
        combined_df[
            f"{model_a}_correct"
        ]
    )

    correct_b = (
        combined_df[
            f"{model_b}_correct"
        ]
    )


    both_correct = int(
        (
            correct_a &
            correct_b
        ).sum()
    )


    model_a_correct_b_wrong = int(
        (
            correct_a &
            (~correct_b)
        ).sum()
    )


    model_a_wrong_b_correct = int(
        (
            (~correct_a) &
            correct_b
        ).sum()
    )


    both_wrong = int(
        (
            (~correct_a) &
            (~correct_b)
        ).sum()
    )


    discordant_total = (
        model_a_correct_b_wrong +
        model_a_wrong_b_correct
    )


    if discordant_total == 0:

        p_value = 1.0

    else:

        test_result = binomtest(
            k=model_a_correct_b_wrong,
            n=discordant_total,
            p=0.5,
            alternative="two-sided"
        )

        p_value = (
            test_result.pvalue
        )


    accuracy_a = float(
        correct_a.mean() * 100
    )

    accuracy_b = float(
        correct_b.mean() * 100
    )


    accuracy_difference = (
        accuracy_a -
        accuracy_b
    )


    if p_value < 0.05:

        if accuracy_difference > 0:

            conclusion = (
                f"{MODELS[model_a]['display']} "
                f"significantly outperforms "
                f"{MODELS[model_b]['display']}"
            )

        elif accuracy_difference < 0:

            conclusion = (
                f"{MODELS[model_b]['display']} "
                f"significantly outperforms "
                f"{MODELS[model_a]['display']}"
            )

        else:

            conclusion = (
                "Statistically significant "
                "difference detected"
            )

    else:

        conclusion = (
            "No statistically significant "
            "difference"
        )


    return {

        "model_a":
            model_a,

        "model_a_display":
            MODELS[
                model_a
            ]["display"],

        "model_b":
            model_b,

        "model_b_display":
            MODELS[
                model_b
            ]["display"],

        "both_correct":
            both_correct,

        "model_a_correct_b_wrong":
            model_a_correct_b_wrong,

        "model_a_wrong_b_correct":
            model_a_wrong_b_correct,

        "both_wrong":
            both_wrong,

        "discordant_predictions":
            discordant_total,

        "model_a_accuracy_percent":
            accuracy_a,

        "model_b_accuracy_percent":
            accuracy_b,

        "accuracy_difference_percent":
            accuracy_difference,

        "p_value":
            p_value,

        "statistically_significant":
            p_value < 0.05,

        "conclusion":
            conclusion
    }


# =============================================================================
# PERFORM ALL PAIRWISE TESTS
# =============================================================================

print()
print("=" * 100)
print("PAIRWISE MCCNEMAR TEST RESULTS")
print("=" * 100)


results = []


for model_a, model_b in combinations(
    MODELS.keys(),
    2
):

    result = perform_mcnemar_test(
        model_a,
        model_b
    )


    results.append(
        result
    )


    print()
    print(
        f"{result['model_a_display']}"
    )

    print(
        "vs"
    )

    print(
        f"{result['model_b_display']}"
    )

    print()

    print(
        f"Both correct: "
        f"{result['both_correct']}"
    )

    print(
        f"Model A correct / "
        f"Model B wrong: "
        f"{result['model_a_correct_b_wrong']}"
    )

    print(
        f"Model A wrong / "
        f"Model B correct: "
        f"{result['model_a_wrong_b_correct']}"
    )

    print(
        f"Both wrong: "
        f"{result['both_wrong']}"
    )

    print(
        f"Discordant predictions: "
        f"{result['discordant_predictions']}"
    )

    print()

    print(
        f"Accuracy difference: "
        f"{result['accuracy_difference_percent']:.2f}%"
    )

    print(
        f"Exact McNemar p-value: "
        f"{result['p_value']:.6f}"
    )

    print(
        f"Conclusion: "
        f"{result['conclusion']}"
    )


# =============================================================================
# CREATE RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    results
)


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# BONFERRONI
# =============================================================================

number_of_tests = len(
    results_df
)


results_df[
    "bonferroni_adjusted_p_value"
] = np.minimum(
    results_df[
        "p_value"
    ] *
    number_of_tests,

    1.0
)


results_df[
    "significant_after_bonferroni"
] = (
    results_df[
        "bonferroni_adjusted_p_value"
    ] <
    0.05
)


# =============================================================================
# SAVE PAIRWISE RESULTS
# =============================================================================

output_file = (
    OUTPUT_DIR /
    "unseen_source_pairwise_mcnemar_tests.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


# =============================================================================
# CREATE P-VALUE MATRIX
# =============================================================================

model_names = list(
    MODELS.keys()
)


p_value_matrix = pd.DataFrame(

    np.ones(
        (
            len(model_names),
            len(model_names)
        )
    ),

    index=[
        MODELS[
            model
        ]["display"]
        for model in model_names
    ],

    columns=[
        MODELS[
            model
        ]["display"]
        for model in model_names
    ]
)


for _, row in results_df.iterrows():

    model_a = (
        row[
            "model_a_display"
        ]
    )

    model_b = (
        row[
            "model_b_display"
        ]
    )

    p_value = (
        row[
            "bonferroni_adjusted_p_value"
        ]
    )


    p_value_matrix.loc[
        model_a,
        model_b
    ] = p_value


    p_value_matrix.loc[
        model_b,
        model_a
    ] = p_value


p_value_file = (
    OUTPUT_DIR /
    "unseen_source_mcnemar_p_value_matrix.csv"
)


p_value_matrix.to_csv(
    p_value_file
)


# =============================================================================
# CREATE P-VALUE HEATMAP
# =============================================================================

plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    p_value_matrix.values,
    aspect="auto"
)


plt.colorbar(
    label="Bonferroni-adjusted p-value"
)


plt.xticks(

    np.arange(
        len(
            p_value_matrix.columns
        )
    ),

    p_value_matrix.columns,

    rotation=30,

    ha="right"
)


plt.yticks(

    np.arange(
        len(
            p_value_matrix.index
        )
    ),

    p_value_matrix.index
)


for i in range(
    len(
        p_value_matrix.index
    )
):

    for j in range(
        len(
            p_value_matrix.columns
        )
    ):

        plt.text(

            j,

            i,

            f"{p_value_matrix.iloc[i, j]:.3f}",

            ha="center",

            va="center"
        )


plt.title(
    "Pairwise McNemar Test P-values "
    "(Bonferroni Adjusted)"
)


plt.tight_layout()


heatmap_file = (
    FIGURES_DIR /
    "unseen_source_mcnemar_p_value_heatmap.png"
)


plt.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =============================================================================
# CREATE MODEL ACCURACY TABLE
# =============================================================================

accuracy_rows = []


for model_name in MODELS:

    correct_column = (
        f"{model_name}_correct"
    )


    correct_count = int(
        combined_df[
            correct_column
        ].sum()
    )


    total_images = len(
        combined_df
    )


    accuracy = (
        correct_count /
        total_images *
        100
    )


    accuracy_rows.append({

        "model":
            model_name,

        "model_display":
            MODELS[
                model_name
            ]["display"],

        "correct_predictions":
            correct_count,

        "total_images":
            total_images,

        "accuracy_percent":
            accuracy
    })


accuracy_df = pd.DataFrame(
    accuracy_rows
)


accuracy_df = (
    accuracy_df
    .sort_values(
        "accuracy_percent",
        ascending=False
    )
)


accuracy_file = (
    OUTPUT_DIR /
    "unseen_source_accuracy_statistical_summary.csv"
)


accuracy_df.to_csv(
    accuracy_file,
    index=False
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 100)
print("STATISTICAL ANALYSIS SUMMARY")
print("=" * 100)

print()

print(
    "Number of models:",
    len(MODELS)
)

print(
    "Pairwise comparisons:",
    number_of_tests
)

print()

print(
    "Comparisons significant "
    "before Bonferroni correction:",
    int(
        results_df[
            "statistically_significant"
        ].sum()
    )
)

print(
    "Comparisons significant "
    "after Bonferroni correction:",
    int(
        results_df[
            "significant_after_bonferroni"
        ].sum()
    )
)

print()
print(
    "Pairwise results:"
)
print(
    output_file
)

print()
print(
    "P-value matrix:"
)
print(
    p_value_file
)

print()
print(
    "Accuracy summary:"
)
print(
    accuracy_file
)

print()
print(
    "Heatmap:"
)
print(
    heatmap_file
)

print()
print("=" * 100)
print("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 100)

