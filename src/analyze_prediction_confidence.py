import os
import pandas as pd
import numpy as np


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

EFFICIENTNET_PATH = os.path.join(
    METRICS_DIR,
    "efficientnet_b0_test_predictions.csv"
)

PROPOSED_MODEL_PATH = os.path.join(
    METRICS_DIR,
    "proposed_model_test_predictions.csv"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "prediction_confidence_analysis.txt"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_confidence_columns(dataframe):

    return [
        column
        for column in dataframe.columns
        if column.startswith("Probability_")
    ]


def calculate_confidence(dataframe):

    probability_columns = get_confidence_columns(
        dataframe
    )

    probabilities = dataframe[
        probability_columns
    ].values

    return np.max(
        probabilities,
        axis=1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading prediction files...")

    efficientnet_df = pd.read_csv(
        EFFICIENTNET_PATH
    )

    proposed_df = pd.read_csv(
        PROPOSED_MODEL_PATH
    )

    print(
        f"EfficientNet-B0 predictions: "
        f"{len(efficientnet_df)}"
    )

    print(
        f"Proposed model predictions: "
        f"{len(proposed_df)}"
    )

    # --------------------------------------------------------
    # VALIDATE ALIGNMENT
    # --------------------------------------------------------

    if not (
        efficientnet_df["True_Label"]
        ==
        proposed_df["True_Label"]
    ).all():

        raise ValueError(
            "True labels are not aligned."
        )

    print("✓ Prediction files are aligned")

    # --------------------------------------------------------
    # CALCULATE CONFIDENCE
    # --------------------------------------------------------

    efficientnet_df["Confidence"] = (
        calculate_confidence(
            efficientnet_df
        )
    )

    proposed_df["Confidence"] = (
        calculate_confidence(
            proposed_df
        )
    )

    # --------------------------------------------------------
    # CONFIDENCE GROUPS
    # --------------------------------------------------------

    efficientnet_correct = efficientnet_df[
        efficientnet_df["Correct"] == True
    ]

    efficientnet_incorrect = efficientnet_df[
        efficientnet_df["Correct"] == False
    ]

    proposed_correct = proposed_df[
        proposed_df["Correct"] == True
    ]

    proposed_incorrect = proposed_df[
        proposed_df["Correct"] == False
    ]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    models = [

        (
            "EfficientNet-B0",
            efficientnet_df,
            efficientnet_correct,
            efficientnet_incorrect
        ),

        (
            "Proposed Attention-EfficientNet-B0",
            proposed_df,
            proposed_correct,
            proposed_incorrect
        )

    ]

    for (
        model_name,
        full_df,
        correct_df,
        incorrect_df
    ) in models:

        results.append({

            "Model": model_name,

            "Average Confidence (%)":
            full_df["Confidence"].mean() * 100,

            "Correct Prediction Confidence (%)":
            correct_df["Confidence"].mean() * 100,

            "Incorrect Prediction Confidence (%)":
            incorrect_df["Confidence"].mean() * 100,

            "Minimum Confidence (%)":
            full_df["Confidence"].min() * 100,

            "Maximum Confidence (%)":
            full_df["Confidence"].max() * 100

        })

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # DISAGREEMENT ANALYSIS
    # --------------------------------------------------------

    disagreement_mask = (
        efficientnet_df["Predicted_Label"]
        !=
        proposed_df["Predicted_Label"]
    )

    disagreement_count = (
        disagreement_mask.sum()
    )

    efficientnet_disagreement_confidence = (
        efficientnet_df.loc[
            disagreement_mask,
            "Confidence"
        ].mean() * 100
    )

    proposed_disagreement_confidence = (
        proposed_df.loc[
            disagreement_mask,
            "Confidence"
        ].mean() * 100
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL CONFIDENCE RESULTS")
    print("=" * 70)

    for _, row in results_df.iterrows():

        print(
            f"\nModel: {row['Model']}"
        )

        print(
            f"Average Confidence: "
            f"{row['Average Confidence (%)']:.2f}%"
        )

        print(
            f"Correct Prediction Confidence: "
            f"{row['Correct Prediction Confidence (%)']:.2f}%"
        )

        print(
            f"Incorrect Prediction Confidence: "
            f"{row['Incorrect Prediction Confidence (%)']:.2f}%"
        )

        print(
            f"Minimum Confidence: "
            f"{row['Minimum Confidence (%)']:.2f}%"
        )

        print(
            f"Maximum Confidence: "
            f"{row['Maximum Confidence (%)']:.2f}%"
        )

    print("\n" + "=" * 70)
    print("MODEL DISAGREEMENT CONFIDENCE")
    print("=" * 70)

    print(
        f"\nPrediction disagreements: "
        f"{disagreement_count}"
    )

    print(
        f"EfficientNet-B0 average confidence "
        f"during disagreements: "
        f"{efficientnet_disagreement_confidence:.2f}%"
    )

    print(
        f"Proposed model average confidence "
        f"during disagreements: "
        f"{proposed_disagreement_confidence:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "PREDICTION CONFIDENCE ANALYSIS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for _, row in results_df.iterrows():

            file.write(
                f"Model: {row['Model']}\n"
            )

            file.write(
                f"Average Confidence: "
                f"{row['Average Confidence (%)']:.2f}%\n"
            )

            file.write(
                f"Correct Prediction Confidence: "
                f"{row['Correct Prediction Confidence (%)']:.2f}%\n"
            )

            file.write(
                f"Incorrect Prediction Confidence: "
                f"{row['Incorrect Prediction Confidence (%)']:.2f}%\n"
            )

            file.write(
                f"Minimum Confidence: "
                f"{row['Minimum Confidence (%)']:.2f}%\n"
            )

            file.write(
                f"Maximum Confidence: "
                f"{row['Maximum Confidence (%)']:.2f}%\n\n"
            )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "MODEL DISAGREEMENT CONFIDENCE\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Prediction disagreements: "
            f"{disagreement_count}\n"
        )

        file.write(
            f"EfficientNet-B0 average confidence "
            f"during disagreements: "
            f"{efficientnet_disagreement_confidence:.2f}%\n"
        )

        file.write(
            f"Proposed model average confidence "
            f"during disagreements: "
            f"{proposed_disagreement_confidence:.2f}%\n"
        )

    print(
        "\n✓ Confidence analysis saved to:"
    )

    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print(
        "PREDICTION CONFIDENCE ANALYSIS COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()
