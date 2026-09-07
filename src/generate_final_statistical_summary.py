from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "publication_statistical_analysis"
    / "publication_statistical_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_statistical_summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(INPUT_FILE)


summary_columns = [
    "baseline_model",
    "proposed_model",
    "baseline_accuracy_percent",
    "proposed_accuracy_percent",
    "accuracy_difference_percent",
    "mcnemar_p_value",
    "mcnemar_significance",
    "baseline_average_confidence",
    "proposed_average_confidence",
    "mean_confidence_difference",
    "wilcoxon_p_value",
    "wilcoxon_significance",
    "rank_biserial_correlation",
    "rank_biserial_interpretation",
    "cohens_dz",
    "cohens_dz_interpretation"
]


summary_df = df[summary_columns].copy()


summary_df = summary_df.rename(
    columns={
        "baseline_model": "Baseline Model",
        "proposed_model": "Proposed Model",
        "baseline_accuracy_percent": "Baseline Accuracy (%)",
        "proposed_accuracy_percent": "Proposed Accuracy (%)",
        "accuracy_difference_percent": "Accuracy Difference (%)",
        "mcnemar_p_value": "McNemar p-value",
        "mcnemar_significance": "Accuracy Significance",
        "baseline_average_confidence": "Baseline Average Confidence",
        "proposed_average_confidence": "Proposed Average Confidence",
        "mean_confidence_difference": "Confidence Difference",
        "wilcoxon_p_value": "Wilcoxon p-value",
        "wilcoxon_significance": "Confidence Significance",
        "rank_biserial_correlation": "Rank Biserial Correlation",
        "rank_biserial_interpretation": "Effect Size",
        "cohens_dz": "Cohens dz",
        "cohens_dz_interpretation": "Cohens dz Interpretation"
    }
)


output_file = (
    OUTPUT_DIR
    / "final_statistical_comparison.csv"
)

summary_df.to_csv(
    output_file,
    index=False
)


print("=" * 80)
print("FINAL STATISTICAL ANALYSIS")
print("=" * 80)

print("\nProposed Model Statistical Comparisons:\n")

for _, row in summary_df.iterrows():

    print(
        f"Baseline: {row['Baseline Model']}"
    )

    print(
        f"Accuracy Difference: "
        f"{row['Accuracy Difference (%)']:.2f}%"
    )

    print(
        f"McNemar Test: "
        f"{row['Accuracy Significance']}"
    )

    print(
        f"Confidence Difference: "
        f"{row['Confidence Difference']:.4f}"
    )

    print(
        f"Wilcoxon Test: "
        f"{row['Confidence Significance']}"
    )

    print(
        f"Effect Size: "
        f"{row['Effect Size']}"
    )

    print("-" * 60)


print(
    f"\nSaved: {output_file}"
)

print("\nFINAL STATISTICAL SUMMARY COMPLETED")