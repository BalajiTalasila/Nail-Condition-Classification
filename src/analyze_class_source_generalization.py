from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
OVERLAP_FILE = PROJECT_ROOT / "results" / "metrics" / "test_source_overlap_analysis.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "metrics" / "model_class_performance_by_source_overlap.csv"

print("=" * 85)
print("CLASS-WISE MODEL PERFORMANCE: OVERLAPPING VS UNSEEN SOURCES")
print("=" * 85)

# --------------------------------------------------
# LOAD SOURCE OVERLAP DATA
# --------------------------------------------------

overlap_df = pd.read_csv(OVERLAP_FILE)

print("\nLoaded source overlap data")
print(f"Total test images: {len(overlap_df)}")

# Normalize paths
overlap_df["file_path_normalized"] = (
    overlap_df["file_path"]
    .astype(str)
    .str.replace("/", "\\", regex=False)
    .str.lower()
)

# Convert overlap column safely
overlap_df["source_overlap_with_train"] = (
    overlap_df["source_overlap_with_train"]
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

if overlap_df["source_overlap_with_train"].isna().any():
    raise ValueError(
        "Some values in source_overlap_with_train could not be converted."
    )

print("\nSource groups:")
print(
    overlap_df["source_overlap_with_train"]
    .value_counts()
    .rename({
        True: "Overlapping source",
        False: "Unseen source"
    })
)

all_results = []

prediction_files = sorted(PREDICTIONS_DIR.glob("*_predictions.csv"))

print("\nPrediction files:")
for file in prediction_files:
    print(f"- {file.name}")

# --------------------------------------------------
# ANALYZE EACH MODEL
# --------------------------------------------------

for prediction_file in prediction_files:

    model_name = prediction_file.name.replace("_predictions.csv", "")

    print("\n" + "=" * 85)
    print(f"MODEL: {model_name}")
    print("=" * 85)

    pred_df = pd.read_csv(prediction_file)

    pred_df["file_path_normalized"] = (
        pred_df["filepath"]
        .astype(str)
        .str.replace("/", "\\", regex=False)
        .str.lower()
    )

    merged = pred_df.merge(
        overlap_df[
            [
                "file_path_normalized",
                "source_overlap_with_train"
            ]
        ],
        on="file_path_normalized",
        how="left"
    )

    unmatched = merged["source_overlap_with_train"].isna().sum()

    print(f"\nPrediction rows: {len(merged)}")
    print(f"Unmatched rows: {unmatched}")

    if unmatched > 0:
        print("\nWARNING: Some prediction paths could not be matched.")
        continue

    merged["correct_bool"] = (
        merged["correct"]
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

    if merged["correct_bool"].isna().any():
        raise ValueError(
            f"Could not convert correctness column for {model_name}"
        )

    # --------------------------------------------------
    # CLASS + SOURCE GROUP ANALYSIS
    # --------------------------------------------------

    grouped = (
        merged
        .groupby(
            [
                "true_class",
                "source_overlap_with_train"
            ]
        )
        .agg(
            images=("correct_bool", "size"),
            correct_predictions=("correct_bool", "sum")
        )
        .reset_index()
    )

    grouped["accuracy_percent"] = (
        grouped["correct_predictions"]
        / grouped["images"]
        * 100
    )

    grouped["model"] = model_name

    grouped["source_group"] = grouped[
        "source_overlap_with_train"
    ].map({
        True: "Overlapping",
        False: "Unseen"
    })

    grouped = grouped[
        [
            "model",
            "true_class",
            "source_group",
            "images",
            "correct_predictions",
            "accuracy_percent"
        ]
    ]

    all_results.append(grouped)

    print("\nClass-wise results:")

    for class_name in sorted(
        merged["true_class"].unique()
    ):

        print(f"\n{class_name}")

        class_results = grouped[
            grouped["true_class"] == class_name
        ]

        for _, row in class_results.iterrows():

            print(
                f"  {row['source_group']:12s} | "
                f"Images: {int(row['images']):3d} | "
                f"Accuracy: {row['accuracy_percent']:6.2f}%"
            )


# --------------------------------------------------
# COMBINE RESULTS
# --------------------------------------------------

final_df = pd.concat(
    all_results,
    ignore_index=True
)

final_df = final_df.sort_values(
    [
        "model",
        "true_class",
        "source_group"
    ]
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 85)
print("GENERATED FILE")
print("=" * 85)

print(OUTPUT_FILE)

# --------------------------------------------------
# CREATE PIVOT SUMMARY
# --------------------------------------------------

pivot = final_df.pivot_table(
    index=[
        "model",
        "true_class"
    ],
    columns="source_group",
    values="accuracy_percent"
).reset_index()

pivot.columns.name = None

if (
    "Overlapping" in pivot.columns
    and "Unseen" in pivot.columns
):

    pivot["accuracy_gap_percentage_points"] = (
        pivot["Overlapping"]
        - pivot["Unseen"]
    )

pivot_file = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "model_class_source_overlap_comparison.csv"
)

pivot.to_csv(
    pivot_file,
    index=False
)

print("\nGenerated comparison file:")
print(pivot_file)

print("\n" + "=" * 85)
print("ANALYSIS COMPLETED")
print("=" * 85)

