from pathlib import Path
import hashlib
from itertools import combinations

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_DIR / "results" / "metrics"

SPLIT_METADATA_FILE = (
    METRICS_DIR / "dataset_split_metadata.csv"
)

OUTPUT_DISTRIBUTION = (
    METRICS_DIR / "dataset_split_distribution.csv"
)

OUTPUT_DUPLICATES = (
    METRICS_DIR / "cross_split_duplicate_check.csv"
)

OUTPUT_REPORT = (
    METRICS_DIR / "dataset_split_integrity_report.txt"
)


# ============================================================
# FILE HASH FUNCTION
# ============================================================

def calculate_sha256(file_path):

    hash_object = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hash_object.update(chunk)

    return hash_object.hexdigest()


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_dataset_split():

    print("\n" + "=" * 70)
    print("DATASET SPLIT INTEGRITY AUDIT")
    print("=" * 70)

    if not SPLIT_METADATA_FILE.exists():

        print(
            "\nERROR: dataset_split_metadata.csv was not found."
        )

        return

    print("\nLoading split metadata...")

    df = pd.read_csv(
        SPLIT_METADATA_FILE
    )

    print(
        f"✓ Total metadata records: {len(df)}"
    )

    print(
        f"✓ Number of classes: {df['class'].nunique()}"
    )

    print(
        f"✓ Split labels: {sorted(df['split'].unique())}"
    )


    # ========================================================
    # BASIC SPLIT STATISTICS
    # ========================================================

    print("\n" + "-" * 70)
    print("SPLIT SIZE ANALYSIS")
    print("-" * 70)

    split_counts = (
        df["split"]
        .value_counts()
        .sort_index()
    )

    total_images = len(df)

    for split_name, count in split_counts.items():

        percentage = (
            count / total_images * 100
        )

        print(
            f"{split_name}: "
            f"{count} images "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION BY SPLIT")
    print("-" * 70)

    distribution = pd.crosstab(

        df["class"],

        df["split"]

    )

    for expected_split in [

        "train",
        "val",
        "test"

    ]:

        if expected_split not in distribution.columns:

            distribution[
                expected_split
            ] = 0

    distribution = distribution[

        [
            "train",
            "val",
            "test"
        ]

    ]

    distribution["total"] = (

        distribution.sum(axis=1)

    )

    distribution["imbalance_ratio"] = (

        distribution["total"]

        / distribution["total"].min()

    )

    print()

    print(

        distribution.to_string()

    )

    distribution.to_csv(

        OUTPUT_DISTRIBUTION

    )

    class_totals = (

        distribution["total"]

    )

    largest_class = (

        class_totals.idxmax()

    )

    smallest_class = (

        class_totals.idxmin()

    )

    imbalance_ratio = (

        class_totals.max()

        / class_totals.min()

    )


    # ========================================================
    # SHA-256 INTEGRITY CHECK
    # ========================================================

    print("\n" + "-" * 70)
    print("CROSS-SPLIT EXACT DUPLICATE CHECK")
    print("-" * 70)

    if "sha256" not in df.columns:

        print(
            "\nWARNING: sha256 column not found."
        )

        print(
            "Recalculating hashes from file paths..."
        )

        df["sha256"] = df[
            "filepath"
        ].apply(

            lambda path:

            calculate_sha256(

                Path(path)

            )

        )

    duplicate_rows = []

    hash_groups = (

        df.groupby("sha256")

    )

    for sha256, group in hash_groups:

        unique_splits = (

            sorted(

                group["split"].unique()

            )

        )

        if len(unique_splits) > 1:

            for _, row in group.iterrows():

                duplicate_rows.append({

                    "sha256": sha256,

                    "filepath": row[
                        "filepath"
                    ],

                    "filename": row[
                        "filename"
                    ],

                    "class": row[
                        "class"
                    ],

                    "split": row[
                        "split"
                    ],

                    "splits_present": (
                        ",".join(
                            unique_splits
                        )
                    )

                })

    duplicates_df = pd.DataFrame(

        duplicate_rows

    )

    if duplicates_df.empty:

        duplicates_df = pd.DataFrame(

            columns=[

                "sha256",
                "filepath",
                "filename",
                "class",
                "split",
                "splits_present"

            ]

        )

        print(
            "\n✓ No exact SHA-256 duplicates found across splits."
        )

    else:

        print(

            f"\n⚠ Cross-split duplicate records found: "
            f"{len(duplicates_df)}"

        )

        print(

            duplicates_df.to_string(
                index=False
            )

        )

    duplicates_df.to_csv(

        OUTPUT_DUPLICATES,

        index=False

    )


    # ========================================================
    # FILENAME OVERLAP CHECK
    # ========================================================

    print("\n" + "-" * 70)
    print("CROSS-SPLIT FILENAME OVERLAP CHECK")
    print("-" * 70)

    filename_overlap_records = []

    filename_groups = (

        df.groupby("filename")

    )

    for filename, group in filename_groups:

        unique_splits = (

            sorted(

                group["split"].unique()

            )

        )

        if len(unique_splits) > 1:

            filename_overlap_records.append({

                "filename": filename,

                "splits_present": (
                    ",".join(
                        unique_splits
                    )
                ),

                "number_of_records": (
                    len(group)
                )

            })

    filename_overlap_df = pd.DataFrame(

        filename_overlap_records

    )

    filename_overlap_count = (

        len(filename_overlap_df)

    )

    if filename_overlap_count == 0:

        print(
            "\n✓ No filenames overlap across splits."
        )

    else:

        print(

            f"\n⚠ Filename overlaps across splits: "
            f"{filename_overlap_count}"

        )

        print(

            filename_overlap_df.head(
                20
            ).to_string(
                index=False
            )

        )


    # ========================================================
    # ORIGINAL DATASET FOLDER ANALYSIS
    # ========================================================

    print("\n" + "-" * 70)
    print("ORIGINAL SOURCE FOLDER ANALYSIS")
    print("-" * 70)

    source_split_counts = {

        "train": 0,

        "validation": 0,

        "test": 0,

        "unknown": 0

    }

    for filepath in df["filepath"]:

        path_parts = [

            part.lower()

            for part in Path(
                filepath
            ).parts

        ]

        if "train" in path_parts:

            source_split_counts[
                "train"
            ] += 1

        elif "validation" in path_parts:

            source_split_counts[
                "validation"
            ] += 1

        elif "val" in path_parts:

            source_split_counts[
                "validation"
            ] += 1

        elif "test" in path_parts:

            source_split_counts[
                "test"
            ] += 1

        else:

            source_split_counts[
                "unknown"
            ] += 1

    for source_name, count in (

        source_split_counts.items()

    ):

        print(

            f"{source_name}: "
            f"{count} source images"

        )


    # ========================================================
    # CROSS-TAB OF ORIGINAL VS NEW SPLIT
    # ========================================================

    print("\n" + "-" * 70)
    print("ORIGINAL FOLDER VS NEW RANDOM SPLIT")
    print("-" * 70)

    source_labels = []

    for filepath in df["filepath"]:

        path_parts = [

            part.lower()

            for part in Path(
                filepath
            ).parts

        ]

        if "train" in path_parts:

            source_labels.append(
                "original_train"
            )

        elif "validation" in path_parts:

            source_labels.append(
                "original_validation"
            )

        elif "val" in path_parts:

            source_labels.append(
                "original_validation"
            )

        elif "test" in path_parts:

            source_labels.append(
                "original_test"
            )

        else:

            source_labels.append(
                "unknown"
            )

    df["original_source_split"] = (

        source_labels

    )

    source_vs_new = pd.crosstab(

        df[
            "original_source_split"
        ],

        df["split"]

    )

    print()

    print(

        source_vs_new.to_string()

    )


    # ========================================================
    # FINAL INTEGRITY STATUS
    # ========================================================

    exact_leakage = (

        not duplicates_df.empty

    )

    filename_overlap = (

        filename_overlap_count > 0

    )

    original_split_detected = (

        source_split_counts["train"] > 0

        and

        source_split_counts["validation"] > 0

        and

        source_split_counts["test"] > 0

    )

    if exact_leakage:

        integrity_status = (

            "FAILED: Exact duplicate leakage detected "
            "across new splits."

        )

    elif original_split_detected:

        integrity_status = (

            "WARNING: The source dataset appears to contain "
            "a pre-existing train/validation/test structure. "
            "The current processed split was created by "
            "re-randomizing images from those original folders. "
            "This requires further leakage and near-duplicate "
            "investigation before final experimental use."

        )

    elif filename_overlap:

        integrity_status = (

            "WARNING: Filename overlap exists across splits. "
            "Manual investigation is recommended."

        )

    else:

        integrity_status = (

            "PASSED: No exact cross-split duplicate leakage "
            "was detected."

        )


    # ========================================================
    # GENERATE REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "DATASET SPLIT INTEGRITY REPORT"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        f"Total images: {total_images}"
    )

    report_lines.append(
        f"Number of classes: "
        f"{df['class'].nunique()}"
    )

    report_lines.append("")

    report_lines.append(
        "SPLIT COUNTS"
    )

    report_lines.append(
        "-" * 70
    )

    for split_name, count in split_counts.items():

        report_lines.append(

            f"{split_name}: {count}"

        )

    report_lines.append("")

    report_lines.append(
        "CLASS DISTRIBUTION"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(

        distribution.to_string()

    )

    report_lines.append("")

    report_lines.append(
        "DATASET IMBALANCE"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(

        f"Largest class: "
        f"{largest_class} "
        f"({class_totals.max()} images)"

    )

    report_lines.append(

        f"Smallest class: "
        f"{smallest_class} "
        f"({class_totals.min()} images)"

    )

    report_lines.append(

        f"Imbalance ratio: "
        f"{imbalance_ratio:.4f}"

    )

    report_lines.append("")

    report_lines.append(
        "EXACT DUPLICATE CHECK"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(

        f"Cross-split duplicate records: "
        f"{len(duplicates_df)}"

    )

    report_lines.append("")

    report_lines.append(
        "FILENAME OVERLAP CHECK"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(

        f"Filename overlaps: "
        f"{filename_overlap_count}"

    )

    report_lines.append("")

    report_lines.append(
        "ORIGINAL SOURCE FOLDER ANALYSIS"
    )

    report_lines.append(
        "-" * 70
    )

    for source_name, count in (

        source_split_counts.items()

    ):

        report_lines.append(

            f"{source_name}: {count}"

        )

    report_lines.append("")

    report_lines.append(
        "FINAL STATUS"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        integrity_status
    )

    report_lines.append("")

    report_lines.append(
        "=" * 70
    )

    OUTPUT_REPORT.write_text(

        "\n".join(
            report_lines
        ),

        encoding="utf-8"

    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL DATASET INTEGRITY STATUS")
    print("=" * 70)

    print(
        f"\n{integrity_status}"
    )

    print("\nGenerated files:")

    print(
        f"✓ {OUTPUT_DISTRIBUTION}"
    )

    print(
        f"✓ {OUTPUT_DUPLICATES}"
    )

    print(
        f"✓ {OUTPUT_REPORT}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    audit_dataset_split()
