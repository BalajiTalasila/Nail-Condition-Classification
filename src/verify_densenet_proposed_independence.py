from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path.cwd()


RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics" /
    "unseen_source_detailed_results"
)


densenet_file = (
    RESULTS_DIR /
    "densenet121_image_predictions.csv"
)


proposed_file = (
    RESULTS_DIR /
    "proposed_attention_efficientnet_b0_image_predictions.csv"
)


densenet_df = pd.read_csv(
    densenet_file
)


proposed_df = pd.read_csv(
    proposed_file
)


print("=" * 80)
print("DENSENET VS PROPOSED MODEL VERIFICATION")
print("=" * 80)


print("\nTotal DenseNet images:")
print(len(densenet_df))


print("\nTotal Proposed model images:")
print(len(proposed_df))


same_image_order = (
    densenet_df["image_path"].equals(
        proposed_df["image_path"]
    )
)


print("\nSame image order:")
print(same_image_order)


same_predictions = (
    densenet_df["predicted_class"]
    ==
    proposed_df["predicted_class"]
)


print("\nSame predictions:")
print(
    same_predictions.sum(),
    "/",
    len(densenet_df)
)


confidence_difference = (
    proposed_df["confidence"]
    -
    densenet_df["confidence"]
)


print("\nConfidence values identical:")
print(
    np.allclose(
        densenet_df["confidence"],
        proposed_df["confidence"]
    )
)


print("\nAverage absolute confidence difference:")
print(
    abs(
        confidence_difference
    ).mean()
)


probability_columns = [
    column
    for column in densenet_df.columns
    if column.startswith(
        "probability_"
    )
]


densenet_probabilities = (
    densenet_df[
        probability_columns
    ].to_numpy()
)


proposed_probabilities = (
    proposed_df[
        probability_columns
    ].to_numpy()
)


probabilities_identical = (
    np.allclose(
        densenet_probabilities,
        proposed_probabilities
    )
)


print("\nProbability distributions identical:")
print(
    probabilities_identical
)


absolute_probability_difference = (
    np.abs(
        densenet_probabilities
        -
        proposed_probabilities
    )
)


print("\nAverage probability difference:")
print(
    absolute_probability_difference.mean()
)


print("\nMaximum probability difference:")
print(
    absolute_probability_difference.max()
)


print("\n" + "=" * 80)
print("FIRST 10 IMAGE COMPARISONS")
print("=" * 80)


comparison = pd.DataFrame({

    "true_class":
    densenet_df["true_class"],

    "densenet_prediction":
    densenet_df["predicted_class"],

    "densenet_confidence":
    densenet_df["confidence"],

    "proposed_prediction":
    proposed_df["predicted_class"],

    "proposed_confidence":
    proposed_df["confidence"],

    "confidence_difference":
    confidence_difference

})


print(
    comparison.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("VERIFICATION COMPLETED")
print("=" * 80)

