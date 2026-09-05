import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = (
    PROJECT_ROOT /
    "results" /
    "predictions"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "detailed_calibration_analysis"
)

FIGURES_DIR = (
    PROJECT_ROOT /
    "results" /
    "figures"
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

MODEL_CONFIGS = {
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
# FIND PREDICTION FILE
# =============================================================================

def find_prediction_file(model_name):

    possible_files = list(
        PREDICTIONS_DIR.rglob(
            f"*{model_name}*.csv"
        )
    )

    if not possible_files:

        raise FileNotFoundError(
            f"Prediction CSV not found for {model_name}"
        )

    return possible_files[0]


# =============================================================================
# FIND COLUMN
# =============================================================================

def find_column(df, keywords):

    for column in df.columns:

        column_lower = column.lower()

        for keyword in keywords:

            if keyword in column_lower:

                return column

    raise ValueError(
        f"Could not find column using keywords: {keywords}"
    )


# =============================================================================
# LOAD MODEL DATA
# =============================================================================

def load_model_predictions(model_name):

    prediction_file = find_prediction_file(
        model_name
    )

    df = pd.read_csv(
        prediction_file
    )

    # Verify required columns
    required_columns = [
        "true_class",
        "predicted_class",
        "confidence",
        "correct"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns for {model_name}: "
            f"{missing_columns}"
        )

    result_df = pd.DataFrame()

    result_df["confidence"] = (
        df["confidence"].astype(float)
    )

    # Recalculate correctness directly from
    # the true and predicted class labels.
    result_df["correct"] = (
        df["predicted_class"].astype(str) ==
        df["true_class"].astype(str)
    )

    return result_df


# =============================================================================
# CALIBRATION BIN ANALYSIS
# =============================================================================

def calculate_calibration_bins(
    df,
    num_bins=10
):

    bins = np.linspace(
        0,
        1,
        num_bins + 1
    )

    records = []

    for i in range(num_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == num_bins - 1:

            mask = (
                (df["confidence"] >= lower) &
                (df["confidence"] <= upper)
            )

        else:

            mask = (
                (df["confidence"] >= lower) &
                (df["confidence"] < upper)
            )

        bin_data = df[mask]

        if len(bin_data) == 0:

            records.append({
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_center": (
                    lower + upper
                ) / 2,
                "sample_count": 0,
                "average_confidence": np.nan,
                "accuracy": np.nan,
                "calibration_gap": np.nan
            })

        else:

            average_confidence = (
                bin_data["confidence"].mean()
            )

            accuracy = (
                bin_data["correct"].mean()
            )

            calibration_gap = (
                average_confidence -
                accuracy
            )

            records.append({
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_center": (
                    lower + upper
                ) / 2,
                "sample_count": len(
                    bin_data
                ),
                "average_confidence": (
                    average_confidence
                ),
                "accuracy": accuracy,
                "calibration_gap": (
                    calibration_gap
                )
            })

    return pd.DataFrame(
        records
    )


# =============================================================================
# OVERCONFIDENCE ANALYSIS
# =============================================================================

def calculate_confidence_behavior(df):

    correct_df = df[
        df["correct"]
    ]

    incorrect_df = df[
        ~df["correct"]
    ]

    accuracy = (
        df["correct"].mean()
    )

    average_confidence = (
        df["confidence"].mean()
    )

    calibration_difference = (
        average_confidence -
        accuracy
    )

    if calibration_difference > 0.02:

        calibration_behavior = (
            "Overconfident"
        )

    elif calibration_difference < -0.02:

        calibration_behavior = (
            "Underconfident"
        )

    else:

        calibration_behavior = (
            "Well calibrated"
        )

    return {

        "total_images": len(df),

        "accuracy_percent":
            accuracy * 100,

        "average_confidence_percent":
            average_confidence * 100,

        "confidence_accuracy_gap_percent":
            calibration_difference * 100,

        "calibration_behavior":
            calibration_behavior,

        "correct_average_confidence_percent":
            correct_df["confidence"].mean() * 100,

        "incorrect_average_confidence_percent":
            incorrect_df["confidence"].mean() * 100
            if len(incorrect_df) > 0
            else np.nan,

        "high_confidence_errors":

            len(
                incorrect_df[
                    incorrect_df["confidence"] >= 0.90
                ]
            ),

        "total_errors":

            len(
                incorrect_df
            )
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print("=" * 80)
print("DETAILED CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)


all_results = {}

summary_records = []


for model_name, config in MODEL_CONFIGS.items():

    print()
    print("=" * 80)

    print(
        f"ANALYZING: "
        f"{config['display']}"
    )

    print("=" * 80)


    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    df = load_model_predictions(
        model_name
    )


    # -------------------------------------------------------------------------
    # CALCULATE CONFIDENCE BEHAVIOR
    # -------------------------------------------------------------------------

    behavior = (
        calculate_confidence_behavior(
            df
        )
    )

    behavior["model"] = (
        model_name
    )

    behavior["model_display"] = (
        config["display"]
    )

    summary_records.append(
        behavior
    )


    # -------------------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------------------

    print()

    print(
        f"Total images: "
        f"{behavior['total_images']}"
    )

    print()

    print(
        f"Accuracy: "
        f"{behavior['accuracy_percent']:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{behavior['average_confidence_percent']:.2f}%"
    )

    print(
        f"Confidence - Accuracy gap: "
        f"{behavior['confidence_accuracy_gap_percent']:.2f}%"
    )

    print()

    print(
        f"Calibration behavior: "
        f"{behavior['calibration_behavior']}"
    )

    print()

    print(
        f"High-confidence errors "
        f"(>=90%): "
        f"{behavior['high_confidence_errors']}"
    )

    print(
        f"Total errors: "
        f"{behavior['total_errors']}"
    )


    # -------------------------------------------------------------------------
    # BIN ANALYSIS
    # -------------------------------------------------------------------------

    bin_df = (
        calculate_calibration_bins(
            df
        )
    )

    bin_df["model"] = (
        model_name
    )

    bin_df["model_display"] = (
        config["display"]
    )


    print()

    print(
        "CONFIDENCE BIN ANALYSIS"
    )

    print()

    print(
        bin_df[
            [
                "bin_lower",
                "bin_upper",
                "sample_count",
                "average_confidence",
                "accuracy",
                "calibration_gap"
            ]
        ].to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # SAVE INDIVIDUAL MODEL BIN ANALYSIS
    # -------------------------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"{model_name}_confidence_bins.csv"
    )

    bin_df.to_csv(
        output_file,
        index=False
    )


    all_results[
        model_name
    ] = {
        "predictions": df,
        "bins": bin_df,
        "display": config["display"]
    }


# =============================================================================
# CREATE SUMMARY DATAFRAME
# =============================================================================

summary_df = pd.DataFrame(
    summary_records
)


print()
print("=" * 80)
print("FINAL CALIBRATION SUMMARY")
print("=" * 80)

print()

print(
    summary_df[
        [
            "model_display",
            "accuracy_percent",
            "average_confidence_percent",
            "confidence_accuracy_gap_percent",
            "calibration_behavior",
            "high_confidence_errors",
            "total_errors"
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary_file = (
    OUTPUT_DIR /
    "detailed_calibration_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# =============================================================================
# FIGURE 1: RELIABILITY DIAGRAM
# =============================================================================

print()
print("=" * 80)
print(
    "GENERATING RELIABILITY DIAGRAM"
)
print("=" * 80)


plt.figure(
    figsize=(10, 8)
)


for model_name, result in all_results.items():

    bin_df = result["bins"]

    valid_bins = bin_df.dropna(
        subset=[
            "average_confidence",
            "accuracy"
        ]
    )

    plt.plot(
        valid_bins[
            "average_confidence"
        ],
        valid_bins[
            "accuracy"
        ],
        marker="o",
        linewidth=2,
        label=result[
            "display"
        ]
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    label="Perfect Calibration"
)


plt.xlabel(
    "Average Confidence",
    fontsize=12
)

plt.ylabel(
    "Observed Accuracy",
    fontsize=12
)

plt.title(
    "Reliability Diagram: Confidence vs Accuracy",
    fontsize=14,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


reliability_file = (
    FIGURES_DIR /
    "model_reliability_diagram.png"
)

plt.savefig(
    reliability_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    reliability_file
)


# =============================================================================
# FIGURE 2: CONFIDENCE GAP
# =============================================================================

print()
print(
    "GENERATING CONFIDENCE GAP COMPARISON"
)


plt.figure(
    figsize=(10, 7)
)


plot_df = summary_df.sort_values(
    "confidence_accuracy_gap_percent"
)


plt.bar(
    plot_df[
        "model_display"
    ],
    plot_df[
        "confidence_accuracy_gap_percent"
    ]
)


plt.axhline(
    y=0,
    linewidth=1.5
)


plt.ylabel(
    "Confidence - Accuracy Gap (%)",
    fontsize=12
)

plt.title(
    "Model Confidence Calibration Gap",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()


gap_file = (
    FIGURES_DIR /
    "confidence_accuracy_gap.png"
)

plt.savefig(
    gap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    gap_file
)


# =============================================================================
# FIGURE 3: CORRECT VS INCORRECT CONFIDENCE
# =============================================================================

print()
print(
    "GENERATING CONFIDENCE DISTRIBUTION COMPARISON"
)


plot_records = []

for model_name, result in all_results.items():

    df = result[
        "predictions"
    ]

    for _, row in df.iterrows():

        if row["correct"]:

            prediction_type = (
                "Correct"
            )

        else:

            prediction_type = (
                "Incorrect"
            )

        plot_records.append({
            "model":
                result["display"],

            "prediction_type":
                prediction_type,

            "confidence":
                row["confidence"] * 100
        })


plot_df = pd.DataFrame(
    plot_records
)


plt.figure(
    figsize=(12, 8)
)


models_order = [
    config["display"]
    for config in MODEL_CONFIGS.values()
]


positions = []

data_to_plot = []

labels = []

position = 1


for model in models_order:

    correct_values = (
        plot_df[
            (
                plot_df["model"] ==
                model
            ) &
            (
                plot_df[
                    "prediction_type"
                ] ==
                "Correct"
            )
        ][
            "confidence"
        ].values
    )

    incorrect_values = (
        plot_df[
            (
                plot_df["model"] ==
                model
            ) &
            (
                plot_df[
                    "prediction_type"
                ] ==
                "Incorrect"
            )
        ][
            "confidence"
        ].values
    )


    data_to_plot.append(
        correct_values
    )

    positions.append(
        position
    )

    labels.append(
        f"{model}\nCorrect"
    )


    position += 1


    data_to_plot.append(
        incorrect_values
    )

    positions.append(
        position
    )

    labels.append(
        f"{model}\nIncorrect"
    )


    position += 2


plt.boxplot(
    data_to_plot,
    positions=positions
)


plt.xticks(
    positions,
    labels,
    rotation=35,
    ha="right"
)


plt.ylabel(
    "Prediction Confidence (%)",
    fontsize=12
)


plt.title(
    "Confidence Distribution: Correct vs Incorrect Predictions",
    fontsize=14,
    fontweight="bold"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


distribution_file = (
    FIGURES_DIR /
    "correct_vs_incorrect_confidence.png"
)

plt.savefig(
    distribution_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    distribution_file
)


# =============================================================================
# FIGURE 4: HIGH-CONFIDENCE ERROR RATE
# =============================================================================

print()
print(
    "GENERATING HIGH-CONFIDENCE ERROR COMPARISON"
)


summary_df[
    "high_confidence_error_rate_percent"
] = (
    summary_df[
        "high_confidence_errors"
    ] /
    summary_df[
        "total_images"
    ]
) * 100


plt.figure(
    figsize=(10, 7)
)


plt.bar(
    summary_df[
        "model_display"
    ],
    summary_df[
        "high_confidence_error_rate_percent"
    ]
)


plt.ylabel(
    "High-Confidence Errors / Total Images (%)",
    fontsize=12
)


plt.title(
    "High-Confidence Error Rate Across Models",
    fontsize=14,
    fontweight="bold"
)


plt.xticks(
    rotation=20,
    ha="right"
)


plt.tight_layout()


high_error_file = (
    FIGURES_DIR /
    "high_confidence_error_rate.png"
)

plt.savefig(
    high_error_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    high_error_file
)


# =============================================================================
# SAVE UPDATED SUMMARY
# =============================================================================

summary_df.to_csv(
    summary_file,
    index=False
)


print()

print("=" * 80)

print(
    "DETAILED CALIBRATION ANALYSIS COMPLETED"
)

print("=" * 80)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Figures directory:"
)

print(
    FIGURES_DIR
)

