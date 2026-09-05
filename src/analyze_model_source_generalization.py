from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

OVERLAP_FILE = METRICS_DIR / "test_source_overlap_analysis.csv"
OUTPUT_FILE = METRICS_DIR / "model_performance_by_source_overlap.csv"

print("=" * 78)
print("MODEL PERFORMANCE: OVERLAPPING VS UNSEEN TEST SOURCES")
print("=" * 78)

# ------------------------------------------------------------
# Load source overlap analysis
# ------------------------------------------------------------

if not OVERLAP_FILE.exists():
    raise FileNotFoundError(
        f"Required file not found:\n{OVERLAP_FILE}"
    )

overlap_df = pd.read_csv(OVERLAP_FILE)

print("\nLoaded source overlap analysis")
print(f"Total test images: {len(overlap_df)}")
print(f"Columns: {list(overlap_df.columns)}")

# ------------------------------------------------------------
# Identify required columns
# ------------------------------------------------------------

filepath_col = "file_path"
overlap_col = "source_overlap_with_train"

if filepath_col not in overlap_df.columns:
    raise ValueError(
        f"Expected filepath column '{filepath_col}' not found."
    )

if overlap_col not in overlap_df.columns:
    raise ValueError(
        f"Expected overlap column '{overlap_col}' not found."
    )

print("\nUsing columns:")
print(f"Filepath column: {filepath_col}")
print(f"Overlap column: {overlap_col}")

# ------------------------------------------------------------
# Normalize paths
# ------------------------------------------------------------

def normalize_path(path_value):
    return (
        str(path_value)
        .replace("/", "\\")
        .strip()
        .lower()
    )

overlap_df["normalized_filepath"] = overlap_df[
    filepath_col
].apply(normalize_path)

# ------------------------------------------------------------
# Convert overlap values to boolean
# ------------------------------------------------------------

def to_bool(value):
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in [
        "true",
        "1",
        "yes",
        "y"
    ]

overlap_df["has_source_overlap"] = overlap_df[
    overlap_col
].apply(to_bool)

print("\nSOURCE GROUP SUMMARY")
print("-" * 78)

print(
    f"Overlapping-source test images: "
    f"{overlap_df['has_source_overlap'].sum()}"
)

print(
    f"Unseen-source test images: "
    f"{(~overlap_df['has_source_overlap']).sum()}"
)

# Create lookup dictionary
overlap_map = dict(
    zip(
        overlap_df["normalized_filepath"],
        overlap_df["has_source_overlap"]
    )
)

# ------------------------------------------------------------
# Find prediction files
# ------------------------------------------------------------

prediction_files = sorted(
    PREDICTIONS_DIR.glob("*_predictions.csv")
)

if not prediction_files:
    raise FileNotFoundError(
        f"No prediction CSV files found in:\n{PREDICTIONS_DIR}"
    )

print("\nPrediction files found:")

for file in prediction_files:
    print(f"- {file.name}")

# ------------------------------------------------------------
# Analyze each model
# ------------------------------------------------------------

results = []

for prediction_file in prediction_files:

    print("\n" + "=" * 78)
    print(f"ANALYZING: {prediction_file.name}")
    print("=" * 78)

    df = pd.read_csv(prediction_file)

    required_columns = [
        "filepath",
        "correct"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in "
                f"{prediction_file.name}"
            )

    # Normalize prediction paths
    df["normalized_filepath"] = df[
        "filepath"
    ].apply(normalize_path)

    # Map source-overlap status
    df["has_source_overlap"] = df[
        "normalized_filepath"
    ].map(overlap_map)

    unmatched = df[
        "has_source_overlap"
    ].isna().sum()

    if unmatched > 0:

        print(
            f"\nWARNING: {unmatched} prediction paths "
            f"did not match the overlap analysis."
        )

        unmatched_paths = df.loc[
            df["has_source_overlap"].isna(),
            "filepath"
        ]

        print("Example unmatched paths:")

        for path in unmatched_paths.head(5):
            print(path)

    # Convert correct column robustly
    df["correct"] = df[
        "correct"
    ].apply(to_bool)

    # Remove unmatched rows rather than incorrectly
    # classifying them as unseen
    matched_df = df.dropna(
        subset=["has_source_overlap"]
    ).copy()

    matched_df["has_source_overlap"] = matched_df[
        "has_source_overlap"
    ].astype(bool)

    overlapping_df = matched_df[
        matched_df["has_source_overlap"]
    ]

    unseen_df = matched_df[
        ~matched_df["has_source_overlap"]
    ]

    # --------------------------------------------------------
    # Calculate accuracy
    # --------------------------------------------------------

    total_images = len(df)

    matched_images = len(matched_df)

    overall_accuracy = (
        df["correct"].mean() * 100
    )

    overlap_accuracy = (
        overlapping_df["correct"].mean() * 100
        if len(overlapping_df) > 0
        else float("nan")
    )

    unseen_accuracy = (
        unseen_df["correct"].mean() * 100
        if len(unseen_df) > 0
        else float("nan")
    )

    accuracy_gap = (
        overlap_accuracy - unseen_accuracy
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(f"\nTotal prediction rows: {total_images}")
    print(f"Matched with overlap audit: {matched_images}")
    print(f"Unmatched paths: {unmatched}")

    print(
        f"\nOverlapping-source images: "
        f"{len(overlapping_df)}"
    )

    print(
        f"Unseen-source images: "
        f"{len(unseen_df)}"
    )

    print(
        f"\nOverall test accuracy: "
        f"{overall_accuracy:.2f}%"
    )

    print(
        f"Overlapping-source accuracy: "
        f"{overlap_accuracy:.2f}%"
    )

    print(
        f"Unseen-source accuracy: "
        f"{unseen_accuracy:.2f}%"
    )

    print(
        f"Performance gap: "
        f"{accuracy_gap:.2f} percentage points"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    model_name = prediction_file.stem.replace(
        "_predictions",
        ""
    )

    results.append({
        "model": model_name,
        "total_test_images": total_images,
        "matched_images": matched_images,
        "unmatched_prediction_paths": unmatched,
        "overlapping_source_images": len(overlapping_df),
        "unseen_source_images": len(unseen_df),
        "overall_accuracy_percent": overall_accuracy,
        "overlapping_source_accuracy_percent": overlap_accuracy,
        "unseen_source_accuracy_percent": unseen_accuracy,
        "accuracy_gap_percentage_points": accuracy_gap
    })

# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="unseen_source_accuracy_percent",
    ascending=False
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 78)
print("FINAL MODEL COMPARISON")
print("=" * 78)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nGenerated file:")
print(OUTPUT_FILE)

print("\n" + "=" * 78)
print("INTERPRETATION")
print("=" * 78)

print(
    "Overall accuracy represents performance on the complete "
    "576-image test set."
)

print(
    "Overlapping-source accuracy represents test images whose "
    "source image also appears in the training dataset."
)

print(
    "Unseen-source accuracy represents test images originating "
    "from sources not present in training."
)

print(
    "A large positive performance gap indicates that source "
    "overlap may be inflating the conventional test accuracy."
)

print("\nANALYSIS COMPLETED")
print("=" * 78)
