from pathlib import Path

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
    "error_analysis"
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
print("COMPREHENSIVE ERROR ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD PREDICTION FILES
# ============================================================

model_data = {}


for model_name in MODEL_NAMES:

    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{prediction_file}"
        )


    dataframe = pd.read_csv(
        prediction_file
    )


    model_data[
        model_name
    ] = dataframe


    print(
        f"\nLoaded {MODEL_DISPLAY_NAMES[model_name]}: "
        f"{len(dataframe)} images"
    )


# ============================================================
# VERIFY IMAGE ALIGNMENT
# ============================================================

reference_model = MODEL_NAMES[0]

reference_dataframe = model_data[
    reference_model
]


image_identifier_columns = [
    column
    for column in [
        "image_path",
        "image_name",
        "filename"
    ]
    if column in reference_dataframe.columns
]


print("\n" + "=" * 80)
print("DATA ALIGNMENT VERIFICATION")
print("=" * 80)


print(
    f"\nReference model: "
    f"{MODEL_DISPLAY_NAMES[reference_model]}"
)


for model_name in MODEL_NAMES[1:]:

    current_dataframe = model_data[
        model_name
    ]


    same_order = True


    if "image_path" in reference_dataframe.columns:

        same_order = (
            reference_dataframe[
                "image_path"
            ].tolist()
            ==
            current_dataframe[
                "image_path"
            ].tolist()
        )


    print(
        f"{MODEL_DISPLAY_NAMES[model_name]} "
        f"same image order: "
        f"{same_order}"
    )


# ============================================================
# CREATE COMBINED PREDICTION TABLE
# ============================================================

combined_dataframe = pd.DataFrame()


if "image_path" in reference_dataframe.columns:

    combined_dataframe[
        "image_path"
    ] = reference_dataframe[
        "image_path"
    ]


combined_dataframe[
    "true_class"
] = reference_dataframe[
    "true_class"
]


for model_name in MODEL_NAMES:

    dataframe = model_data[
        model_name
    ]


    combined_dataframe[
        f"{model_name}_prediction"
    ] = dataframe[
        "predicted_class"
    ]


    combined_dataframe[
        f"{model_name}_confidence"
    ] = dataframe[
        "confidence"
    ]


    combined_dataframe[
        f"{model_name}_correct"
    ] = (
        dataframe[
            "true_class"
        ]
        ==
        dataframe[
            "predicted_class"
        ]
    )


# ============================================================
# OVERALL ERROR COUNTS
# ============================================================

print("\n" + "=" * 80)
print("OVERALL ERROR COUNTS")
print("=" * 80)


error_summary = []


for model_name in MODEL_NAMES:

    correct_column = (
        f"{model_name}_correct"
    )


    total_images = len(
        combined_dataframe
    )


    correct_count = int(
        combined_dataframe[
            correct_column
        ].sum()
    )


    incorrect_count = (
        total_images
        -
        correct_count
    )


    error_rate = (
        incorrect_count
        /
        total_images
        *
        100
    )


    print(
        f"\n{MODEL_DISPLAY_NAMES[model_name]}"
    )


    print(
        f"Correct predictions: "
        f"{correct_count}"
    )


    print(
        f"Incorrect predictions: "
        f"{incorrect_count}"
    )


    print(
        f"Error rate: "
        f"{error_rate:.2f}%"
    )


    error_summary.append({

        "model":
        MODEL_DISPLAY_NAMES[model_name],

        "total_images":
        total_images,

        "correct_predictions":
        correct_count,

        "incorrect_predictions":
        incorrect_count,

        "error_rate_percent":
        error_rate

    })


error_summary_dataframe = pd.DataFrame(
    error_summary
)


# ============================================================
# FIND IMAGES MISCLASSIFIED BY ALL MODELS
# ============================================================

print("\n" + "=" * 80)
print("COMMON MISCLASSIFICATIONS")
print("=" * 80)


correct_columns = [

    f"{model_name}_correct"

    for model_name in MODEL_NAMES
]


all_models_wrong_mask = (
    ~combined_dataframe[
        correct_columns
    ].any(
        axis=1
    )
)


all_models_wrong = (
    combined_dataframe[
        all_models_wrong_mask
    ]
)


print(
    f"\nImages misclassified by ALL models: "
    f"{len(all_models_wrong)}"
)


if len(all_models_wrong) > 0:

    print("\nCommon difficult cases:")

    print(
        all_models_wrong.to_string(
            index=False
        )
    )


# ============================================================
# FIND IMAGES CORRECTLY CLASSIFIED ONLY BY PROPOSED MODEL
# ============================================================

print("\n" + "=" * 80)
print("PROPOSED MODEL UNIQUE CORRECT PREDICTIONS")
print("=" * 80)


proposed_correct_column = (
    "proposed_attention_efficientnet_b0_correct"
)


baseline_correct_columns = [

    f"{model_name}_correct"

    for model_name in MODEL_NAMES

    if model_name
    !=
    "proposed_attention_efficientnet_b0"
]


proposed_unique_correct_mask = (

    combined_dataframe[
        proposed_correct_column
    ]

    &

    ~combined_dataframe[
        baseline_correct_columns
    ].any(
        axis=1
    )
)


proposed_unique_correct = (
    combined_dataframe[
        proposed_unique_correct_mask
    ]
)


print(
    f"\nImages correctly classified ONLY by "
    f"the proposed model: "
    f"{len(proposed_unique_correct)}"
)


if len(proposed_unique_correct) > 0:

    print(
        proposed_unique_correct.to_string(
            index=False
        )
    )


# ============================================================
# BASELINE CORRECT BUT PROPOSED WRONG
# ============================================================

print("\n" + "=" * 80)
print("BASELINE ADVANTAGE CASES")
print("=" * 80)


proposed_wrong_mask = (

    ~combined_dataframe[
        proposed_correct_column
    ]
)


any_baseline_correct_mask = (
    combined_dataframe[
        baseline_correct_columns
    ].any(
        axis=1
    )
)


baseline_advantage_cases = (
    combined_dataframe[

        proposed_wrong_mask

        &

        any_baseline_correct_mask
    ]
)


print(
    f"\nImages where proposed model was wrong "
    f"but at least one baseline was correct: "
    f"{len(baseline_advantage_cases)}"
)


if len(baseline_advantage_cases) > 0:

    print(
        baseline_advantage_cases.to_string(
            index=False
        )
    )


# ============================================================
# INDIVIDUAL MODEL ERROR TABLES
# ============================================================

print("\n" + "=" * 80)
print("INDIVIDUAL MODEL MISCLASSIFICATIONS")
print("=" * 80)


for model_name in MODEL_NAMES:

    print("\n" + "-" * 80)

    print(
        f"MISCLASSIFICATIONS: "
        f"{MODEL_DISPLAY_NAMES[model_name]}"
    )

    print("-" * 80)


    prediction_column = (
        f"{model_name}_prediction"
    )


    confidence_column = (
        f"{model_name}_confidence"
    )


    correct_column = (
        f"{model_name}_correct"
    )


    errors = combined_dataframe[

        ~combined_dataframe[
            correct_column
        ]

    ].copy()


    selected_columns = []


    if "image_path" in errors.columns:

        selected_columns.append(
            "image_path"
        )


    selected_columns.extend([

        "true_class",

        prediction_column,

        confidence_column

    ])


    errors = errors[
        selected_columns
    ]


    errors = errors.rename(
        columns={

            prediction_column:
            "predicted_class",

            confidence_column:
            "confidence"

        }
    )


    print(
        f"\nTotal errors: "
        f"{len(errors)}"
    )


    if len(errors) > 0:

        print(
            errors.to_string(
                index=False
            )
        )


    output_file = (

        OUTPUT_DIR /

        f"{model_name}_misclassifications.csv"
    )


    errors.to_csv(
        output_file,
        index=False
    )


# ============================================================
# ERROR PATTERN ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("ERROR PATTERN ANALYSIS")
print("=" * 80)


error_patterns = []


for model_name in MODEL_NAMES:

    prediction_column = (
        f"{model_name}_prediction"
    )


    correct_column = (
        f"{model_name}_correct"
    )


    errors = combined_dataframe[

        ~combined_dataframe[
            correct_column
        ]

    ]


    grouped_errors = errors.groupby(

        [
            "true_class",
            prediction_column
        ]

    ).size().reset_index(
        name="count"
    )


    grouped_errors = grouped_errors.rename(
        columns={

            prediction_column:
            "predicted_class"

        }
    )


    grouped_errors[
        "model"
    ] = MODEL_DISPLAY_NAMES[
        model_name
    ]


    error_patterns.append(
        grouped_errors
    )


all_error_patterns = pd.concat(

    error_patterns,

    ignore_index=True
)


print("\nError patterns:")


if len(all_error_patterns) > 0:

    print(

        all_error_patterns.sort_values(

            "count",

            ascending=False

        ).to_string(

            index=False

        )

    )


# ============================================================
# SAVE COMPREHENSIVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SAVING ERROR ANALYSIS RESULTS")
print("=" * 80)


error_summary_file = (

    OUTPUT_DIR /

    "model_error_summary.csv"
)


error_summary_dataframe.to_csv(

    error_summary_file,

    index=False
)


combined_file = (

    OUTPUT_DIR /

    "combined_model_predictions.csv"
)


combined_dataframe.to_csv(

    combined_file,

    index=False
)


common_errors_file = (

    OUTPUT_DIR /

    "common_misclassifications_all_models.csv"
)


all_models_wrong.to_csv(

    common_errors_file,

    index=False
)


proposed_unique_file = (

    OUTPUT_DIR /

    "proposed_unique_correct_predictions.csv"
)


proposed_unique_correct.to_csv(

    proposed_unique_file,

    index=False
)


baseline_advantage_file = (

    OUTPUT_DIR /

    "baseline_advantage_cases.csv"
)


baseline_advantage_cases.to_csv(

    baseline_advantage_file,

    index=False
)


error_patterns_file = (

    OUTPUT_DIR /

    "error_pattern_analysis.csv"
)


all_error_patterns.to_csv(

    error_patterns_file,

    index=False
)


print("\nSaved files:")

print(error_summary_file)

print(combined_file)

print(common_errors_file)

print(proposed_unique_file)

print(baseline_advantage_file)

print(error_patterns_file)


print("\n" + "=" * 80)
print("COMPREHENSIVE ERROR ANALYSIS COMPLETED")
print("=" * 80)
