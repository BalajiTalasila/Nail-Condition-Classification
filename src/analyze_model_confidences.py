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
    "cross_model_confidence_analysis"
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
print("CROSS-MODEL CONFIDENCE ANALYSIS")
print("=" * 80)


prediction_dataframes = {}


# ============================================================
# LOAD PREDICTION FILES
# ============================================================

for model_name in MODEL_NAMES:

    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        print(
            f"\nWARNING: File not found for {model_name}"
        )

        continue


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


if not prediction_dataframes:

    raise RuntimeError(
        "No prediction files were loaded."
    )


# ============================================================
# BUILD CONFIDENCE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OVERALL CONFIDENCE ANALYSIS")
print("=" * 80)


confidence_summary = []


for model_name, dataframe in (
    prediction_dataframes.items()
):

    dataframe = dataframe.copy()


    dataframe[
        "is_correct"
    ] = (

        dataframe[
            "true_class"
        ]

        ==

        dataframe[
            "predicted_class"
        ]

    )


    overall_average_confidence = dataframe[
        "confidence"
    ].mean()


    correct_predictions = dataframe[
        dataframe[
            "is_correct"
        ]
    ]


    incorrect_predictions = dataframe[
        ~dataframe[
            "is_correct"
        ]
    ]


    average_correct_confidence = (
        correct_predictions[
            "confidence"
        ].mean()
    )


    if len(incorrect_predictions) > 0:

        average_incorrect_confidence = (
            incorrect_predictions[
                "confidence"
            ].mean()
        )

    else:

        average_incorrect_confidence = None


    confidence_summary.append({

        "model": model_name,

        "total_images":
        len(dataframe),

        "correct_predictions":
        len(correct_predictions),

        "incorrect_predictions":
        len(incorrect_predictions),

        "average_confidence":
        overall_average_confidence,

        "average_correct_confidence":
        average_correct_confidence,

        "average_incorrect_confidence":
        average_incorrect_confidence,

        "minimum_confidence":
        dataframe[
            "confidence"
        ].min(),

        "maximum_confidence":
        dataframe[
            "confidence"
        ].max()

    })


confidence_summary_df = pd.DataFrame(
    confidence_summary
)


for column in [

    "average_confidence",

    "average_correct_confidence",

    "average_incorrect_confidence",

    "minimum_confidence",

    "maximum_confidence"

]:

    confidence_summary_df[
        f"{column}_percent"
    ] = (

        confidence_summary_df[
            column
        ]

        *

        100

    )


confidence_summary_df = confidence_summary_df.sort_values(

    "average_confidence",

    ascending=False

)


print("\n")


print(

    confidence_summary_df.to_string(

        index=False

    )

)


# ============================================================
# DENSENET VS PROPOSED CONFIDENCE COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("DENSENET121 VS PROPOSED MODEL CONFIDENCE")
print("=" * 80)


comparison_df = pd.DataFrame()


if (

    "densenet121"
    in prediction_dataframes

    and

    "proposed_attention_efficientnet_b0"
    in prediction_dataframes

):

    densenet_df = prediction_dataframes[
        "densenet121"
    ].copy()


    proposed_df = prediction_dataframes[
        "proposed_attention_efficientnet_b0"
    ].copy()


    comparison_df = pd.DataFrame({

        "image_path":
        densenet_df[
            "image_path"
        ],

        "true_class":
        densenet_df[
            "true_class"
        ],

        "densenet_prediction":
        densenet_df[
            "predicted_class"
        ],

        "proposed_prediction":
        proposed_df[
            "predicted_class"
        ],

        "densenet_confidence":
        densenet_df[
            "confidence"
        ],

        "proposed_confidence":
        proposed_df[
            "confidence"
        ]

    })


    comparison_df[
        "confidence_difference"
    ] = (

        comparison_df[
            "proposed_confidence"
        ]

        -

        comparison_df[
            "densenet_confidence"
        ]

    )


    comparison_df[
        "same_prediction"
    ] = (

        comparison_df[
            "densenet_prediction"
        ]

        ==

        comparison_df[
            "proposed_prediction"
        ]

    )


    proposed_more_confident = comparison_df[

        comparison_df[
            "confidence_difference"
        ]

        >

        0

    ]


    densenet_more_confident = comparison_df[

        comparison_df[
            "confidence_difference"
        ]

        <

        0

    ]


    same_confidence = comparison_df[

        comparison_df[
            "confidence_difference"
        ]

        ==

        0

    ]


    print(

        f"\nTotal images: "
        f"{len(comparison_df)}"

    )


    print(

        f"Same predicted class: "
        f"{comparison_df['same_prediction'].sum()} / "
        f"{len(comparison_df)}"

    )


    print(

        f"Average DenseNet confidence: "
        f"{comparison_df['densenet_confidence'].mean() * 100:.2f}%"

    )


    print(

        f"Average Proposed confidence: "
        f"{comparison_df['proposed_confidence'].mean() * 100:.2f}%"

    )


    print(

        f"Average confidence difference "
        f"(Proposed - DenseNet): "
        f"{comparison_df['confidence_difference'].mean() * 100:.2f}%"

    )


    print(

        f"\nImages where Proposed model is more confident: "
        f"{len(proposed_more_confident)}"

    )


    print(

        f"Images where DenseNet is more confident: "
        f"{len(densenet_more_confident)}"

    )


    print(

        f"Images with identical confidence: "
        f"{len(same_confidence)}"

    )


    print(

        f"\nMaximum Proposed confidence advantage: "
        f"{comparison_df['confidence_difference'].max() * 100:.2f}%"

    )


    print(

        f"Maximum DenseNet confidence advantage: "
        f"{abs(comparison_df['confidence_difference'].min()) * 100:.2f}%"

    )


# ============================================================
# MISCLASSIFIED IMAGE CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MISCLASSIFIED IMAGE CONFIDENCE ANALYSIS")
print("=" * 80)


master_error_df = None


for model_name, dataframe in (
    prediction_dataframes.items()
):

    temp_df = dataframe[

        [

            "image_path",

            "true_class",

            "predicted_class",

            "confidence"

        ]

    ].copy()


    temp_df = temp_df.rename(

        columns={

            "predicted_class":
            f"{model_name}_prediction",

            "confidence":
            f"{model_name}_confidence"

        }

    )


    if master_error_df is None:

        master_error_df = temp_df

    else:

        master_error_df = master_error_df.merge(

            temp_df.drop(

                columns=[

                    "true_class"

                ]

            ),

            on="image_path",

            how="inner"

        )


# Determine whether every model prediction is correct

prediction_columns = [

    f"{model_name}_prediction"

    for model_name in prediction_dataframes

]


for model_name in prediction_dataframes:

    prediction_column = (

        f"{model_name}_prediction"

    )


    master_error_df[

        f"{model_name}_correct"

    ] = (

        master_error_df[
            prediction_column
        ]

        ==

        master_error_df[
            "true_class"
        ]

    )


correct_columns = [

    column

    for column in master_error_df.columns

    if column.endswith(
        "_correct"
    )

]


master_error_df[
    "number_of_models_correct"
] = master_error_df[
    correct_columns
].sum(
    axis=1
)


error_images_df = master_error_df[

    master_error_df[
        "number_of_models_correct"
    ]

    <

    len(
        prediction_dataframes
    )

].copy()


print(

    f"\nImages where at least one model made an error: "
    f"{len(error_images_df)}"

)


if len(error_images_df) > 0:

    print("\n")

    print(

        error_images_df.to_string(

            index=False

        )

    )


# ============================================================
# SAVE RESULTS
# ============================================================

confidence_summary_file = (

    OUTPUT_DIR /

    "model_confidence_summary.csv"

)


confidence_summary_df.to_csv(

    confidence_summary_file,

    index=False

)


if len(comparison_df) > 0:

    densenet_proposed_file = (

        OUTPUT_DIR /

        "densenet_vs_proposed_confidence_comparison.csv"

    )


    comparison_df.to_csv(

        densenet_proposed_file,

        index=False

    )


error_images_file = (

    OUTPUT_DIR /

    "error_image_confidence_analysis.csv"

)


error_images_df.to_csv(

    error_images_file,

    index=False

)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    confidence_summary_file
)


if len(comparison_df) > 0:

    print(
        densenet_proposed_file
    )


print(
    error_images_file
)


print("\n" + "=" * 80)
print("CONFIDENCE ANALYSIS COMPLETED")
print("=" * 80)

