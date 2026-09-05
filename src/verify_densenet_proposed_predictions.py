from pathlib import Path

import pandas as pd
import numpy as np


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


MODEL_NAMES = [

    "convnextv2_tiny",

    "densenet121",

    "efficientnet_b0",

    "proposed_attention_efficientnet_b0"

]


print("=" * 80)
print("PREDICTION AGREEMENT VERIFICATION")
print("=" * 80)


prediction_data = {}


for model_name in MODEL_NAMES:

    prediction_file = (

        DETAILED_RESULTS_DIR /

        f"{model_name}_image_predictions.csv"

    )


    if not prediction_file.exists():

        print(
            f"\nWARNING: File not found for "
            f"{model_name}"
        )

        continue


    dataframe = pd.read_csv(
        prediction_file
    )


    prediction_data[
        model_name
    ] = dataframe


    print(
        f"\nLoaded {model_name}: "
        f"{len(dataframe)} images"
    )


if (
    "densenet121"
    not in prediction_data

    or

    "proposed_attention_efficientnet_b0"
    not in prediction_data
):

    raise RuntimeError(
        "DenseNet121 or proposed model "
        "prediction data is missing."
    )


densenet_df = prediction_data[
    "densenet121"
]


proposed_df = prediction_data[
    "proposed_attention_efficientnet_b0"
]


# ============================================================
# VERIFY IMAGE ORDER
# ============================================================

print("\n" + "=" * 80)
print("IMAGE ORDER VERIFICATION")
print("=" * 80)


same_image_order = np.array_equal(

    densenet_df[
        "image_path"
    ].values,

    proposed_df[
        "image_path"
    ].values

)


print(
    f"\nSame image order: "
    f"{same_image_order}"
)


same_true_labels = np.array_equal(

    densenet_df[
        "true_class"
    ].values,

    proposed_df[
        "true_class"
    ].values

)


print(
    f"Same true labels: "
    f"{same_true_labels}"
)


# ============================================================
# COMPARE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("DENSENET VS PROPOSED PREDICTION COMPARISON")
print("=" * 80)


prediction_match = (

    densenet_df[
        "predicted_class"
    ].values

    ==

    proposed_df[
        "predicted_class"
    ].values

)


matching_count = int(
    prediction_match.sum()
)


different_count = int(
    len(prediction_match)
    -
    matching_count
)


print(
    f"\nTotal images: "
    f"{len(prediction_match)}"
)


print(
    f"Matching predictions: "
    f"{matching_count}"
)


print(
    f"Different predictions: "
    f"{different_count}"
)


print(
    f"Agreement percentage: "
    f"{matching_count / len(prediction_match) * 100:.2f}%"
)


# ============================================================
# CHECK WHETHER ERROR PATTERNS ARE IDENTICAL
# ============================================================

print("\n" + "=" * 80)
print("ERROR PATTERN COMPARISON")
print("=" * 80)


densenet_correct = (

    densenet_df[
        "true_class"
    ]

    ==

    densenet_df[
        "predicted_class"
    ]

)


proposed_correct = (

    proposed_df[
        "true_class"
    ]

    ==

    proposed_df[
        "predicted_class"
    ]

)


same_correctness = np.array_equal(

    densenet_correct.values,

    proposed_correct.values

)


print(
    f"\nIdentical correctness pattern: "
    f"{same_correctness}"
)


print(
    f"DenseNet errors: "
    f"{(~densenet_correct).sum()}"
)


print(
    f"Proposed model errors: "
    f"{(~proposed_correct).sum()}"
)


# ============================================================
# DISPLAY MISCLASSIFIED IMAGES
# ============================================================

errors_df = pd.DataFrame({

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
    ]

})


errors_df = errors_df[

    (

        errors_df[
            "true_class"
        ]

        !=

        errors_df[
            "densenet_prediction"
        ]

    )

    |

    (

        errors_df[
            "true_class"
        ]

        !=

        errors_df[
            "proposed_prediction"
        ]

    )

].copy()


print("\n" + "=" * 80)
print("MISCLASSIFIED IMAGES")
print("=" * 80)


if errors_df.empty:

    print(
        "\nNo misclassified images found."
    )

else:

    print("\n")

    print(
        errors_df.to_string(
            index=False
        )
    )


# ============================================================
# SAVE VERIFICATION RESULTS
# ============================================================

output_file = (

    OUTPUT_DIR /

    "densenet_vs_proposed_prediction_verification.csv"

)


verification_df = pd.DataFrame({

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

    "predictions_match":
    prediction_match,

    "densenet_correct":
    densenet_correct,

    "proposed_correct":
    proposed_correct

})


verification_df.to_csv(

    output_file,

    index=False

)


print("\n" + "=" * 80)
print("VERIFICATION FILE SAVED")
print("=" * 80)


print(
    output_file
)


print("\n" + "=" * 80)
print("VERIFICATION COMPLETED")
print("=" * 80)

