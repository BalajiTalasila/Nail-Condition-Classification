from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = (
    PROJECT_ROOT /
    "results" /
    "unseen_source_evaluation"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_calibration_analysis"
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

    file_mapping = {

        "convnextv2_tiny": (
            PROJECT_ROOT /
            "results" /
            "unseen_source_evaluation" /
            "convnextv2_tiny_unseen_source_predictions.csv"
        ),

        "densenet121": (
            PROJECT_ROOT /
            "results" /
            "unseen_source_evaluation" /
            "densenet121_unseen_source_predictions.csv"
        ),

        "efficientnet_b0": (
            PROJECT_ROOT /
            "results" /
            "unseen_source_evaluation" /
            "efficientnet_b0_unseen_source_predictions.csv"
        ),

        "proposed_attention_efficientnet_b0": (
            PROJECT_ROOT /
            "results" /
            "metrics" /
            "unseen_source_detailed_results" /
            "proposed_attention_efficientnet_b0_image_predictions.csv"
        ),
    }

    if model_name not in file_mapping:

        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    prediction_file = file_mapping[
        model_name
    ]

    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction CSV not found: {prediction_file}"
        )

    return prediction_file

# =============================================================================
# LOAD MODEL PREDICTIONS
# =============================================================================

def load_model_predictions(model_name):

    prediction_file = find_prediction_file(
        model_name
    )

    df = pd.read_csv(
        prediction_file
    )

    # Standardize correctness information.
    # Some prediction files already contain a
    # "correct" column, while the detailed proposed
    # model prediction file does not.

    if "correct" not in df.columns:

        if (
            "true_class" in df.columns
            and
            "predicted_class" in df.columns
        ):

            df["correct"] = (
                df["true_class"] ==
                df["predicted_class"]
            )

        else:

            raise ValueError(
                "Could not determine prediction "
                "correctness because the required "
                "columns are missing."
            )

    # Convert correctness values safely to boolean.

    if df["correct"].dtype == object:

        df["correct"] = (
            df["correct"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False
            })
        )

        if df["correct"].isna().any():

            raise ValueError(
                "The correct column contains "
                "unrecognized values."
            )

    required_columns = [
        "true_class",
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
            f"Missing columns in {prediction_file}: "
            f"{missing_columns}"
        )

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    df["correct"] = (
        df["correct"]
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False
        })
    )

    if df["correct"].isna().any():
        raise ValueError(
            f"Invalid values found in correct column "
            f"for {model_name}"
        )

    return df


# =============================================================================
# CALIBRATION BIN ANALYSIS
# =============================================================================

def calculate_confidence_bins(
    df,
    number_of_bins=10
):

    bin_edges = np.linspace(
        0,
        1,
        number_of_bins + 1
    )

    results = []

    for i in range(number_of_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == number_of_bins - 1:

            bin_data = df[
                (df["confidence"] >= lower) &
                (df["confidence"] <= upper)
            ]

        else:

            bin_data = df[
                (df["confidence"] >= lower) &
                (df["confidence"] < upper)
            ]

        sample_count = len(
            bin_data
        )

        if sample_count > 0:

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

        else:

            average_confidence = np.nan
            accuracy = np.nan
            calibration_gap = np.nan

        results.append({
            "bin_lower": lower,
            "bin_upper": upper,
            "sample_count": sample_count,
            "average_confidence": average_confidence,
            "accuracy": accuracy,
            "calibration_gap": calibration_gap
        })

    return pd.DataFrame(
        results
    )


# =============================================================================
# EXPECTED CALIBRATION ERROR
# =============================================================================

def calculate_ece(
    bin_df,
    total_samples
):

    ece = 0.0

    for _, row in bin_df.iterrows():

        if row["sample_count"] > 0:

            weight = (
                row["sample_count"] /
                total_samples
            )

            gap = abs(
                row["calibration_gap"]
            )

            ece += (
                weight *
                gap
            )

    return ece


# =============================================================================
# MAXIMUM CALIBRATION ERROR
# =============================================================================

def calculate_mce(
    bin_df
):

    valid_gaps = (
        bin_df["calibration_gap"]
        .dropna()
        .abs()
    )

    if len(valid_gaps) == 0:
        return np.nan

    return valid_gaps.max()


# =============================================================================
# BRIER SCORE
# =============================================================================

def calculate_brier_score(df):

    probability_columns = [
        column
        for column in df.columns
        if column.startswith(
            "probability_"
        )
    ]

    if not probability_columns:
        return np.nan

    classes = [
        column.replace(
            "probability_",
            ""
        )
        for column in probability_columns
    ]

    probabilities = (
        df[
            probability_columns
        ]
        .to_numpy()
    )

    brier_scores = []

    for index, row in df.iterrows():

        true_class = row[
            "true_class"
        ]

        target = np.array([
            1.0 if class_name == true_class else 0.0
            for class_name in classes
        ])

        predicted_probabilities = (
            probabilities[
                index
            ]
        )

        score = np.sum(
            (
                predicted_probabilities -
                target
            ) ** 2
        )

        brier_scores.append(
            score
        )

    return np.mean(
        brier_scores
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print(
    "=" * 80
)

print(
    "UNSEEN-SOURCE CONFIDENCE CALIBRATION ANALYSIS"
)

print(
    "=" * 80
)


all_results = {}

summary_rows = []


for model_name, config in MODEL_CONFIGS.items():

    print()

    print(
        "=" * 80
    )

    print(
        f"ANALYZING: {config['display']}"
    )

    print(
        "=" * 80
    )


    df = load_model_predictions(
        model_name
    )


    total_images = len(
        df
    )

    accuracy = (
        df["correct"].mean()
    )

    average_confidence = (
        df["confidence"].mean()
    )

    confidence_accuracy_gap = (
        average_confidence -
        accuracy
    )


    bin_df = calculate_confidence_bins(
        df
    )


    ece = calculate_ece(
        bin_df,
        total_images
    )


    mce = calculate_mce(
        bin_df
    )


    brier_score = calculate_brier_score(
        df
    )


    high_confidence_errors = len(
        df[
            (~df["correct"]) &
            (df["confidence"] >= 0.90)
        ]
    )


    total_errors = len(
        df[
            ~df["correct"]
        ]
    )


    if confidence_accuracy_gap > 0.02:

        calibration_behavior = (
            "Overconfident"
        )

    elif confidence_accuracy_gap < -0.02:

        calibration_behavior = (
            "Underconfident"
        )

    else:

        calibration_behavior = (
            "Well calibrated"
        )


    print()

    print(
        f"Total images: {total_images}"
    )

    print()

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Confidence - Accuracy gap: "
        f"{confidence_accuracy_gap * 100:.2f}%"
    )

    print()

    print(
        f"Calibration behavior: "
        f"{calibration_behavior}"
    )

    print()

    print(
        f"ECE: "
        f"{ece * 100:.2f}%"
    )

    print(
        f"MCE: "
        f"{mce * 100:.2f}%"
    )

    print(
        f"Brier Score: "
        f"{brier_score:.6f}"
    )

    print()

    print(
        f"High-confidence errors (>=90%): "
        f"{high_confidence_errors}"
    )

    print(
        f"Total errors: "
        f"{total_errors}"
    )

    print()

    print(
        "CONFIDENCE BIN ANALYSIS"
    )

    print()

    print(
        bin_df.to_string(
            index=False
        )
    )


    output_file = (
        OUTPUT_DIR /
        f"{model_name}_confidence_bins.csv"
    )

    bin_df.to_csv(
        output_file,
        index=False
    )


    summary_rows.append({
        "model": model_name,
        "model_display": config["display"],
        "total_images": total_images,
        "accuracy_percent": accuracy * 100,
        "average_confidence_percent": (
            average_confidence * 100
        ),
        "confidence_accuracy_gap_percent": (
            confidence_accuracy_gap * 100
        ),
        "calibration_behavior": (
            calibration_behavior
        ),
        "ece_percent": ece * 100,
        "mce_percent": mce * 100,
        "brier_score": brier_score,
        "high_confidence_errors": (
            high_confidence_errors
        ),
        "total_errors": total_errors
    })


    all_results[
        model_name
    ] = {
        "predictions": df,
        "bins": bin_df,
        "display": config["display"]
    }


# =============================================================================
# CREATE SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


print()

print(
    "=" * 80
)

print(
    "FINAL UNSEEN-SOURCE CALIBRATION SUMMARY"
)

print(
    "=" * 80
)

print()

print(
    summary_df.to_string(
        index=False
    )
)


summary_file = (
    OUTPUT_DIR /
    "unseen_source_calibration_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# =============================================================================
# RELIABILITY DIAGRAM
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "GENERATING RELIABILITY DIAGRAM"
)

print(
    "=" * 80
)


plt.figure(
    figsize=(10, 8)
)


for model_name, result in all_results.items():

    bin_df = result[
        "bins"
    ]

    valid_bins = bin_df[
        bin_df["sample_count"] > 0
    ]

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
    "Reliability Diagram: Unseen-Source Test Set",
    fontsize=14,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


reliability_file = (
    FIGURES_DIR /
    "unseen_source_reliability_diagram.png"
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
# CONFIDENCE-ACCURACY GAP FIGURE
# =============================================================================

print()

print(
    "GENERATING CONFIDENCE GAP COMPARISON"
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    summary_df[
        "model_display"
    ],
    summary_df[
        "confidence_accuracy_gap_percent"
    ]
)


plt.axhline(
    0,
    linewidth=1
)


plt.ylabel(
    "Confidence - Accuracy (%)",
    fontsize=12
)

plt.title(
    "Confidence-Accuracy Gap: Unseen-Source Test Set",
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
    "unseen_source_confidence_accuracy_gap.png"
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
# HIGH-CONFIDENCE ERROR FIGURE
# =============================================================================

print()

print(
    "GENERATING HIGH-CONFIDENCE ERROR COMPARISON"
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    summary_df[
        "model_display"
    ],
    summary_df[
        "high_confidence_errors"
    ]
)


plt.ylabel(
    "Number of Errors",
    fontsize=12
)

plt.title(
    "High-Confidence Errors (>=90%): Unseen-Source Test Set",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()


high_confidence_file = (
    FIGURES_DIR /
    "unseen_source_high_confidence_errors.png"
)

plt.savefig(
    high_confidence_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    high_confidence_file
)


# =============================================================================
# COMPLETION
# =============================================================================

print()

print(
    "=" * 80
)

print(
    "UNSEEN-SOURCE CALIBRATION ANALYSIS COMPLETED"
)

print(
    "=" * 80
)

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
