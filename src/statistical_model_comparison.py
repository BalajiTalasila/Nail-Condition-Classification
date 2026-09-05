import os
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "metrics"
)

INPUT_PATH = os.path.join(
    METRICS_DIR,
    "model_prediction_level_comparison.csv"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "statistical_model_comparison.txt"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("STATISTICAL MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading prediction-level comparison data...")

    df = pd.read_csv(INPUT_PATH)

    print(f"✓ Loaded {len(df)} prediction records")

    # --------------------------------------------------------
    # CREATE CONTINGENCY TABLE
    # --------------------------------------------------------

    efficient_correct = (
        df["EfficientNet_B0_Correct"]
        .astype(bool)
    )

    proposed_correct = (
        df["Proposed_Model_Correct"]
        .astype(bool)
    )

    both_correct = (
        efficient_correct &
        proposed_correct
    ).sum()

    efficient_only_correct = (
        efficient_correct &
        ~proposed_correct
    ).sum()

    proposed_only_correct = (
        ~efficient_correct &
        proposed_correct
    ).sum()

    both_incorrect = (
        ~efficient_correct &
        ~proposed_correct
    ).sum()

    print("\nContingency Table:")

    print(
        f"Both Correct: {both_correct}"
    )

    print(
        f"EfficientNet Only Correct: "
        f"{efficient_only_correct}"
    )

    print(
        f"Proposed Model Only Correct: "
        f"{proposed_only_correct}"
    )

    print(
        f"Both Incorrect: {both_incorrect}"
    )

    # --------------------------------------------------------
    # MCNEMAR'S TEST
    # --------------------------------------------------------

    contingency_table = [
        [
            both_correct,
            efficient_only_correct
        ],
        [
            proposed_only_correct,
            both_incorrect
        ]
    ]

    print("\nPerforming McNemar's Test...")

    result = mcnemar(
        contingency_table,
        exact=True
    )

    statistic = result.statistic
    p_value = result.pvalue

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    alpha = 0.05

    if p_value < alpha:

        interpretation = (
            "The performance difference between "
            "EfficientNet-B0 and the proposed model "
            "is statistically significant."
        )

        significance = "Statistically Significant"

    else:

        interpretation = (
            "The performance difference between "
            "EfficientNet-B0 and the proposed model "
            "is not statistically significant."
        )

        significance = "Not Statistically Significant"

    # --------------------------------------------------------
    # CALCULATE ACCURACY
    # --------------------------------------------------------

    total_images = len(df)

    efficient_accuracy = (
        (both_correct + efficient_only_correct)
        / total_images
    ) * 100

    proposed_accuracy = (
        (both_correct + proposed_only_correct)
        / total_images
    ) * 100

    accuracy_difference = (
        efficient_accuracy -
        proposed_accuracy
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON RESULTS")
    print("=" * 70)

    print(
        f"\nTotal Test Images: "
        f"{total_images}"
    )

    print(
        f"\nEfficientNet-B0 Accuracy: "
        f"{efficient_accuracy:.4f}%"
    )

    print(
        f"Proposed Model Accuracy: "
        f"{proposed_accuracy:.4f}%"
    )

    print(
        f"Accuracy Difference: "
        f"{accuracy_difference:.4f} percentage points"
    )

    print(
        f"\nMcNemar Test Statistic: "
        f"{statistic}"
    )

    print(
        f"McNemar Test P-Value: "
        f"{p_value:.6f}"
    )

    print(
        f"\nSignificance Level: "
        f"{alpha}"
    )

    print(
        f"Result: {significance}"
    )

    print(
        f"\nInterpretation:\n"
        f"{interpretation}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    summary_lines = [

        "=" * 70,
        "STATISTICAL MODEL COMPARISON",
        "=" * 70,

        f"\nTotal Test Images: {total_images}",

        "\nCONTINGENCY TABLE",
        "-" * 70,

        f"Both Models Correct: {both_correct}",
        f"Only EfficientNet-B0 Correct: {efficient_only_correct}",
        f"Only Proposed Model Correct: {proposed_only_correct}",
        f"Both Models Incorrect: {both_incorrect}",

        "\nMODEL ACCURACY",
        "-" * 70,

        f"EfficientNet-B0 Accuracy: "
        f"{efficient_accuracy:.4f}%",

        f"Proposed Model Accuracy: "
        f"{proposed_accuracy:.4f}%",

        f"Accuracy Difference: "
        f"{accuracy_difference:.4f} percentage points",

        "\nMCNEMAR'S TEST",
        "-" * 70,

        f"Test Statistic: {statistic}",
        f"P-Value: {p_value:.6f}",
        f"Significance Level: {alpha}",
        f"Result: {significance}",

        "\nINTERPRETATION",
        "-" * 70,

        interpretation,

        "\n" + "=" * 70
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(summary_lines)
        )

    print("\n✓ Statistical comparison saved to:")

    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
