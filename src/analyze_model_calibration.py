import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


EFFICIENTNET_PATH = os.path.join(
    METRICS_DIR,
    "efficientnet_b0_test_predictions.csv"
)

PROPOSED_PATH = os.path.join(
    METRICS_DIR,
    "proposed_model_test_predictions.csv"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "model_calibration_analysis.txt"
)


# ============================================================
# FUNCTIONS
# ============================================================

def get_probability_columns(dataframe):

    return [
        column
        for column in dataframe.columns
        if column.startswith(
            "Probability_"
        )
    ]


def calculate_confidence_and_accuracy(dataframe):

    probability_columns = (
        get_probability_columns(
            dataframe
        )
    )

    probabilities = dataframe[
        probability_columns
    ].values

    confidence = np.max(
        probabilities,
        axis=1
    )

    correctness = dataframe[
        "Correct"
    ].astype(bool).values

    return confidence, correctness


def calculate_ece(
    confidence,
    correctness,
    num_bins=10
):

    bin_boundaries = np.linspace(
        0,
        1,
        num_bins + 1
    )

    ece = 0.0

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    total_samples = len(
        confidence
    )

    for i in range(num_bins):

        lower_bound = (
            bin_boundaries[i]
        )

        upper_bound = (
            bin_boundaries[i + 1]
        )

        if i == num_bins - 1:

            mask = (
                (confidence >= lower_bound)
                &
                (confidence <= upper_bound)
            )

        else:

            mask = (
                (confidence >= lower_bound)
                &
                (confidence < upper_bound)
            )

        if np.sum(mask) > 0:

            bin_confidence = (
                confidence[mask].mean()
            )

            bin_accuracy = (
                correctness[mask].mean()
            )

            bin_count = (
                np.sum(mask)
            )

            ece += (
                abs(
                    bin_accuracy
                    -
                    bin_confidence
                )
                *
                bin_count
                /
                total_samples
            )

            bin_confidences.append(
                bin_confidence
            )

            bin_accuracies.append(
                bin_accuracy
            )

            bin_counts.append(
                bin_count
            )

        else:

            bin_confidences.append(
                np.nan
            )

            bin_accuracies.append(
                np.nan
            )

            bin_counts.append(
                0
            )

    return (
        ece,
        bin_confidences,
        bin_accuracies,
        bin_counts
    )


def calculate_multiclass_brier_score(
    dataframe
):

    probability_columns = (
        get_probability_columns(
            dataframe
        )
    )

    probabilities = dataframe[
        probability_columns
    ].values

    class_names = [
        column.replace(
            "Probability_",
            ""
        )
        for column in probability_columns
    ]

    true_labels = dataframe[
        "True_Label"
    ].values

    total_brier_score = 0.0

    for class_index, class_name in enumerate(
        class_names
    ):

        binary_truth = (
            true_labels == class_name
        ).astype(int)

        class_probability = (
            probabilities[
                :,
                class_index
            ]
        )

        total_brier_score += (
            brier_score_loss(
                binary_truth,
                class_probability
            )
        )

    return (
        total_brier_score
        /
        len(class_names)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL CALIBRATION ANALYSIS")
    print("=" * 70)

    print(
        "\nLoading prediction data..."
    )

    efficientnet_df = pd.read_csv(
        EFFICIENTNET_PATH
    )

    proposed_df = pd.read_csv(
        PROPOSED_PATH
    )

    print(
        f"EfficientNet-B0 samples: "
        f"{len(efficientnet_df)}"
    )

    print(
        f"Proposed model samples: "
        f"{len(proposed_df)}"
    )

    # ========================================================
    # CALCULATE CONFIDENCE
    # ========================================================

    (
        efficientnet_confidence,
        efficientnet_correctness
    ) = (
        calculate_confidence_and_accuracy(
            efficientnet_df
        )
    )

    (
        proposed_confidence,
        proposed_correctness
    ) = (
        calculate_confidence_and_accuracy(
            proposed_df
        )
    )

    # ========================================================
    # ECE
    # ========================================================

    (
        efficientnet_ece,
        efficientnet_bin_confidence,
        efficientnet_bin_accuracy,
        efficientnet_bin_counts
    ) = (
        calculate_ece(
            efficientnet_confidence,
            efficientnet_correctness
        )
    )

    (
        proposed_ece,
        proposed_bin_confidence,
        proposed_bin_accuracy,
        proposed_bin_counts
    ) = (
        calculate_ece(
            proposed_confidence,
            proposed_correctness
        )
    )

    # ========================================================
    # BRIER SCORE
    # ========================================================

    efficientnet_brier = (
        calculate_multiclass_brier_score(
            efficientnet_df
        )
    )

    proposed_brier = (
        calculate_multiclass_brier_score(
            proposed_df
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("CALIBRATION METRICS")
    print("=" * 70)

    print(
        "\nEfficientNet-B0"
    )

    print(
        f"Expected Calibration Error "
        f"(ECE): {efficientnet_ece:.6f}"
    )

    print(
        f"Multiclass Brier Score: "
        f"{efficientnet_brier:.6f}"
    )

    print(
        "\nProposed Attention-EfficientNet-B0"
    )

    print(
        f"Expected Calibration Error "
        f"(ECE): {proposed_ece:.6f}"
    )

    print(
        f"Multiclass Brier Score: "
        f"{proposed_brier:.6f}"
    )

    # ========================================================
    # RELIABILITY DIAGRAM
    # ========================================================

    bin_centers = np.linspace(
        0.05,
        0.95,
        10
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.plot(
        efficientnet_bin_confidence,
        efficientnet_bin_accuracy,
        marker="o",
        label="EfficientNet-B0"
    )

    plt.plot(
        proposed_bin_confidence,
        proposed_bin_accuracy,
        marker="s",
        label="Proposed Model"
    )

    plt.xlabel(
        "Average Confidence"
    )

    plt.ylabel(
        "Actual Accuracy"
    )

    plt.title(
        "Model Reliability Diagram"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    reliability_path = os.path.join(
        FIGURES_DIR,
        "model_reliability_diagram.png"
    )

    plt.savefig(
        reliability_path,
        dpi=300
    )

    plt.close()

    # ========================================================
    # CONFIDENCE DISTRIBUTION
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        efficientnet_confidence * 100,
        bins=20,
        alpha=0.6,
        label="EfficientNet-B0"
    )

    plt.hist(
        proposed_confidence * 100,
        bins=20,
        alpha=0.6,
        label="Proposed Model"
    )

    plt.xlabel(
        "Prediction Confidence (%)"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Prediction Confidence Distribution"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    confidence_path = os.path.join(
        FIGURES_DIR,
        "prediction_confidence_distribution.png"
    )

    plt.savefig(
        confidence_path,
        dpi=300
    )

    plt.close()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "MODEL CALIBRATION ANALYSIS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "EFFICIENTNET-B0\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Expected Calibration Error "
            f"(ECE): "
            f"{efficientnet_ece:.6f}\n"
        )

        file.write(
            f"Multiclass Brier Score: "
            f"{efficientnet_brier:.6f}\n\n"
        )

        file.write(
            "PROPOSED ATTENTION-EFFICIENTNET-B0\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Expected Calibration Error "
            f"(ECE): "
            f"{proposed_ece:.6f}\n"
        )

        file.write(
            f"Multiclass Brier Score: "
            f"{proposed_brier:.6f}\n\n"
        )

        file.write(
            "INTERPRETATION\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "Lower ECE indicates better "
            "confidence calibration.\n"
        )

        file.write(
            "Lower Brier Score indicates "
            "better probabilistic predictions.\n"
        )

    print(
        "\n✓ Reliability diagram saved:"
    )

    print(
        reliability_path
    )

    print(
        "\n✓ Confidence distribution saved:"
    )

    print(
        confidence_path
    )

    print(
        "\n✓ Calibration analysis saved:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 70)
    print(
        "MODEL CALIBRATION ANALYSIS COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()
