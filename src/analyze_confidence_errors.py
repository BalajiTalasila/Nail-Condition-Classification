from pathlib import Path

import numpy as np
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
    "confidence_error_analysis"
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


DISPLAY_NAMES = {
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
print("CONFIDENCE AND ERROR ANALYSIS")
print("=" * 80)


analysis_results = []


for model_name in MODEL_NAMES:

    print("\n" + "=" * 80)

    print(
        f"ANALYZING: "
        f"{DISPLAY_NAMES[model_name]}"
    )

    print("=" * 80)


    prediction_file = (
        DETAILED_RESULTS_DIR /
        f"{model_name}_image_predictions.csv"
    )


    if not prediction_file.exists():

        print(
            f"WARNING: Missing file:"
        )

        print(
            prediction_file
        )

        continue


    dataframe = pd.read_csv(
        prediction_file
    )


    dataframe[
        "correct"
    ] = (
        dataframe[
            "true_class"
        ]
        ==
        dataframe[
            "predicted_class"
        ]
    )


    correct_predictions = dataframe[
        dataframe[
            "correct"
        ]
    ]


    incorrect_predictions = dataframe[
        ~dataframe[
            "correct"
        ]
    ]


    correct_confidences = (
        correct_predictions[
            "confidence"
        ]
    )


    incorrect_confidences = (
        incorrect_predictions[
            "confidence"
        ]
    )


    total_images = len(
        dataframe
    )


    correct_count = len(
        correct_predictions
    )


    incorrect_count = len(
        incorrect_predictions
    )


    accuracy = (
        correct_count
        /
        total_images
    )


    correct_average_confidence = (
        correct_confidences.mean()
        if correct_count > 0
        else np.nan
    )


    incorrect_average_confidence = (
        incorrect_confidences.mean()
        if incorrect_count > 0
        else np.nan
    )


    correct_median_confidence = (
        correct_confidences.median()
        if correct_count > 0
        else np.nan
    )


    incorrect_median_confidence = (
        incorrect_confidences.median()
        if incorrect_count > 0
        else np.nan
    )


    correct_min_confidence = (
        correct_confidences.min()
        if correct_count > 0
        else np.nan
    )


    incorrect_min_confidence = (
        incorrect_confidences.min()
        if incorrect_count > 0
        else np.nan
    )


    correct_max_confidence = (
        correct_confidences.max()
        if correct_count > 0
        else np.nan
    )


    incorrect_max_confidence = (
        incorrect_confidences.max()
        if incorrect_count > 0
        else np.nan
    )


    confidence_separation = (
        correct_average_confidence
        -
        incorrect_average_confidence
    )


    high_confidence_error_count = len(
        incorrect_predictions[
            incorrect_predictions[
                "confidence"
            ]
            >=
            0.90
        ]
    )


    high_confidence_error_percentage = (
        (
            high_confidence_error_count
            /
            incorrect_count
            *
            100
        )
        if incorrect_count > 0
        else 0
    )


    print(
        f"\nTotal images: "
        f"{total_images}"
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
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    print(
        "\nCORRECT PREDICTIONS"
    )


    print(
        f"Average confidence: "
        f"{correct_average_confidence * 100:.2f}%"
    )


    print(
        f"Median confidence: "
        f"{correct_median_confidence * 100:.2f}%"
    )


    print(
        f"Minimum confidence: "
        f"{correct_min_confidence * 100:.2f}%"
    )


    print(
        f"Maximum confidence: "
        f"{correct_max_confidence * 100:.2f}%"
    )


    print(
        "\nINCORRECT PREDICTIONS"
    )


    print(
        f"Average confidence: "
        f"{incorrect_average_confidence * 100:.2f}%"
    )


    print(
        f"Median confidence: "
        f"{incorrect_median_confidence * 100:.2f}%"
    )


    print(
        f"Minimum confidence: "
        f"{incorrect_min_confidence * 100:.2f}%"
    )


    print(
        f"Maximum confidence: "
        f"{incorrect_max_confidence * 100:.2f}%"
    )


    print(
        "\nCONFIDENCE SEPARATION"
    )


    print(
        f"Correct confidence - "
        f"Incorrect confidence: "
        f"{confidence_separation * 100:.2f}%"
    )


    print(
        "\nHIGH-CONFIDENCE ERRORS"
    )


    print(
        f"Errors with confidence >= 90%: "
        f"{high_confidence_error_count}"
    )


    print(
        f"Percentage of errors >= 90% confidence: "
        f"{high_confidence_error_percentage:.2f}%"
    )


    if incorrect_count > 0:

        print(
            "\nINCORRECT PREDICTION DETAILS:"
        )


        error_details = (
            incorrect_predictions[
                [
                    "image_path",
                    "true_class",
                    "predicted_class",
                    "confidence"
                ]
            ]
            .sort_values(
                "confidence",
                ascending=False
            )
        )


        print(
            error_details.to_string(
                index=False
            )
        )


        error_file = (
            OUTPUT_DIR /
            f"{model_name}_incorrect_predictions.csv"
        )


        error_details.to_csv(
            error_file,
            index=False
        )


    analysis_results.append({

        "model":
        model_name,

        "model_display":
        DISPLAY_NAMES[
            model_name
        ],

        "total_images":
        total_images,

        "correct_predictions":
        correct_count,

        "incorrect_predictions":
        incorrect_count,

        "accuracy_percent":
        accuracy * 100,

        "correct_average_confidence_percent":
        correct_average_confidence * 100,

        "incorrect_average_confidence_percent":
        incorrect_average_confidence * 100,

        "confidence_separation_percent":
        confidence_separation * 100,

        "correct_median_confidence_percent":
        correct_median_confidence * 100,

        "incorrect_median_confidence_percent":
        incorrect_median_confidence * 100,

        "correct_min_confidence_percent":
        correct_min_confidence * 100,

        "incorrect_min_confidence_percent":
        incorrect_min_confidence * 100,

        "correct_max_confidence_percent":
        correct_max_confidence * 100,

        "incorrect_max_confidence_percent":
        incorrect_max_confidence * 100,

        "high_confidence_error_count":
        high_confidence_error_count,

        "high_confidence_error_percentage":
        high_confidence_error_percentage

    })


print("\n" + "=" * 80)
print("FINAL CONFIDENCE-ERROR COMPARISON")
print("=" * 80)


summary_dataframe = pd.DataFrame(
    analysis_results
)


summary_dataframe = (
    summary_dataframe
    .sort_values(
        "confidence_separation_percent",
        ascending=False
    )
)


print("\n")


print(
    summary_dataframe.to_string(
        index=False
    )
)


summary_file = (
    OUTPUT_DIR /
    "confidence_error_summary.csv"
)


summary_dataframe.to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    summary_file
)


print("\n" + "=" * 80)
print("CONFIDENCE AND ERROR ANALYSIS COMPLETED")
print("=" * 80)
