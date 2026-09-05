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
    "cross_model_prediction_analysis"
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


print("=" * 80)
print("CROSS-MODEL PREDICTION ANALYSIS")
print("=" * 80)


prediction_dataframes = {}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

for model_name in MODEL_NAMES:

    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        print(
            f"\nWARNING: Prediction file not found for {model_name}"
        )

        print(
            prediction_file
        )

        continue


    dataframe = pd.read_csv(
        prediction_file
    )


    prediction_dataframes[
        model_name
    ] = dataframe


    print(
        f"\nLoaded {model_name}: "
        f"{len(dataframe)} predictions"
    )


if not prediction_dataframes:

    raise RuntimeError(
        "No prediction files were loaded."
    )


# ============================================================
# CREATE MASTER COMPARISON DATAFRAME
# ============================================================

first_model_name = list(
    prediction_dataframes.keys()
)[0]


master_df = prediction_dataframes[
    first_model_name
][

    [
        "image_path",
        "true_class_index",
        "true_class"
    ]

].copy()


for model_name, dataframe in (
    prediction_dataframes.items()
):

    master_df[
        f"{model_name}_prediction"
    ] = dataframe[
        "predicted_class"
    ].values


    master_df[
        f"{model_name}_correct"
    ] = (

        dataframe[
            "true_class"
        ].values

        ==

        dataframe[
            "predicted_class"
        ].values

    )


print("\n" + "=" * 80)
print("MODEL AGREEMENT ANALYSIS")
print("=" * 80)


# ============================================================
# CHECK WHETHER DENSENET AND PROPOSED MODEL ARE IDENTICAL
# ============================================================

if (

    "densenet121" in prediction_dataframes

    and

    "proposed_attention_efficientnet_b0"
    in prediction_dataframes

):

    densenet_predictions = prediction_dataframes[
        "densenet121"
    ][
        "predicted_class"
    ]


    proposed_predictions = prediction_dataframes[
        "proposed_attention_efficientnet_b0"
    ][
        "predicted_class"
    ]


    identical_predictions = (
        densenet_predictions.values
        ==
        proposed_predictions.values
    )


    total_identical = identical_predictions.sum()


    print(
        "\nDenseNet121 vs Proposed Model:"
    )


    print(
        f"Identical predictions: "
        f"{total_identical} / "
        f"{len(master_df)}"
    )


    print(
        f"Prediction agreement: "
        f"{total_identical / len(master_df) * 100:.2f}%"
    )


# ============================================================
# NUMBER OF MODELS CORRECT PER IMAGE
# ============================================================

correct_columns = [

    column

    for column in master_df.columns

    if column.endswith(
        "_correct"
    )

]


master_df[
    "number_of_models_correct"
] = master_df[
    correct_columns
].sum(
    axis=1
)


total_models = len(
    correct_columns
)


# ============================================================
# ALL MODELS CORRECT
# ============================================================

all_correct_df = master_df[

    master_df[
        "number_of_models_correct"
    ]

    ==

    total_models

].copy()


# ============================================================
# ALL MODELS WRONG
# ============================================================

all_wrong_df = master_df[

    master_df[
        "number_of_models_correct"
    ]

    ==

    0

].copy()


# ============================================================
# PROPOSED MODEL UNIQUELY CORRECT
# ============================================================

proposed_column = (
    "proposed_attention_efficientnet_b0_correct"
)


other_correct_columns = [

    column

    for column in correct_columns

    if column != proposed_column

]


proposed_unique_correct_df = pd.DataFrame()


if proposed_column in master_df.columns:

    proposed_unique_correct_df = master_df[

        (

            master_df[
                proposed_column
            ]

            ==

            True

        )

        &

        (

            master_df[
                other_correct_columns
            ].sum(
                axis=1
            )

            ==

            0

        )

    ].copy()


# ============================================================
# PROPOSED MODEL WRONG BUT ALL BASELINES CORRECT
# ============================================================

proposed_wrong_others_correct_df = pd.DataFrame()


if proposed_column in master_df.columns:

    proposed_wrong_others_correct_df = master_df[

        (

            master_df[
                proposed_column
            ]

            ==

            False

        )

        &

        (

            master_df[
                other_correct_columns
            ].sum(
                axis=1
            )

            ==

            len(
                other_correct_columns
            )

        )

    ].copy()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("IMAGE-LEVEL PERFORMANCE SUMMARY")
print("=" * 80)


print(
    f"\nTotal images: "
    f"{len(master_df)}"
)


print(
    f"Images correctly classified by all models: "
    f"{len(all_correct_df)}"
)


print(
    f"Images incorrectly classified by all models: "
    f"{len(all_wrong_df)}"
)


print(
    f"Images uniquely correct for proposed model: "
    f"{len(proposed_unique_correct_df)}"
)


print(
    f"Images where proposed model failed but "
    f"all baselines succeeded: "
    f"{len(proposed_wrong_others_correct_df)}"
)


# ============================================================
# PER-MODEL ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MODEL ERROR ANALYSIS")
print("=" * 80)


model_error_summary = []


for model_name in prediction_dataframes:

    correct_column = (
        f"{model_name}_correct"
    )


    correct_count = master_df[
        correct_column
    ].sum()


    incorrect_count = (
        len(master_df)
        -
        correct_count
    )


    model_error_summary.append({

        "model": model_name,

        "correct_predictions":
        int(correct_count),

        "incorrect_predictions":
        int(incorrect_count),

        "accuracy_percent":
        (
            correct_count
            /
            len(master_df)
        )
        *
        100

    })


error_summary_df = pd.DataFrame(
    model_error_summary
)


error_summary_df = error_summary_df.sort_values(
    "accuracy_percent",
    ascending=False
)


print("\n")

print(
    error_summary_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

master_file = (

    OUTPUT_DIR /

    "all_model_predictions_comparison.csv"

)


master_df.to_csv(

    master_file,

    index=False

)


all_correct_file = (

    OUTPUT_DIR /

    "all_models_correct.csv"

)


all_correct_df.to_csv(

    all_correct_file,

    index=False

)


all_wrong_file = (

    OUTPUT_DIR /

    "all_models_wrong.csv"

)


all_wrong_df.to_csv(

    all_wrong_file,

    index=False

)


proposed_unique_file = (

    OUTPUT_DIR /

    "proposed_model_uniquely_correct.csv"

)


proposed_unique_correct_df.to_csv(

    proposed_unique_file,

    index=False

)


proposed_wrong_file = (

    OUTPUT_DIR /

    "proposed_model_failed_all_baselines_correct.csv"

)


proposed_wrong_others_correct_df.to_csv(

    proposed_wrong_file,

    index=False

)


error_summary_file = (

    OUTPUT_DIR /

    "model_error_summary.csv"

)


error_summary_df.to_csv(

    error_summary_file,

    index=False

)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(master_file)

print(all_correct_file)

print(all_wrong_file)

print(proposed_unique_file)

print(proposed_wrong_file)

print(error_summary_file)


print("\n" + "=" * 80)
print("CROSS-MODEL ANALYSIS COMPLETED")
print("=" * 80)
