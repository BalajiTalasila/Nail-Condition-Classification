# ============================================================
# DETAILED SOURCE-LEVEL DATA LEAKAGE ANALYSIS
# ============================================================

from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "train"
)

VAL_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "val"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"

}


# ============================================================
# EXTRACT SOURCE NAME
# ============================================================

def get_source_name(file_path):

    filename = file_path.stem

    # Remove Roboflow hash suffix
    if ".rf." in filename:

        source_name = filename.split(
            ".rf."
        )[0]

    else:

        source_name = filename

    return source_name


# ============================================================
# SCAN DATASET
# ============================================================

def scan_dataset(dataset_dir, dataset_type):

    sources = defaultdict(list)

    for file_path in dataset_dir.rglob("*"):

        if (

            file_path.is_file()

            and file_path.suffix.lower()
            in IMAGE_EXTENSIONS

        ):

            source_name = get_source_name(
                file_path
            )

            class_name = (
                file_path.parent.name
            )

            sources[
                source_name
            ].append({

                "dataset": dataset_type,

                "class": class_name,

                "file_name": file_path.name,

                "file_path": str(
                    file_path
                )

            })

    return sources


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "DETAILED SOURCE-LEVEL DATA LEAKAGE ANALYSIS"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # SCAN TRAINING DATA
    # --------------------------------------------------------

    print(
        "\nScanning training dataset..."
    )

    train_sources = scan_dataset(

        TRAIN_DIR,

        "train"

    )


    # --------------------------------------------------------
    # SCAN VALIDATION DATA
    # --------------------------------------------------------

    print(
        "Scanning validation dataset..."
    )

    val_sources = scan_dataset(

        VAL_DIR,

        "validation"

    )


    # --------------------------------------------------------
    # FIND OVERLAPPING SOURCES
    # --------------------------------------------------------

    overlapping_sources = (

        set(
            train_sources.keys()
        )

        &

        set(
            val_sources.keys()
        )

    )


    print(

        f"\nOverlapping sources found: "
        f"{len(overlapping_sources)}"
    )


    # --------------------------------------------------------
    # CREATE DETAILED RECORDS
    # --------------------------------------------------------

    records = []


    for source_name in sorted(
        overlapping_sources
    ):

        all_images = (

            train_sources[source_name]

            +

            val_sources[source_name]

        )


        for image_info in all_images:

            records.append({

                "source_name": source_name,

                "dataset": image_info[
                    "dataset"
                ],

                "class": image_info[
                    "class"
                ],

                "file_name": image_info[
                    "file_name"
                ],

                "file_path": image_info[
                    "file_path"
                ]

            })


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    details_df = pd.DataFrame(
        records
    )


    # --------------------------------------------------------
    # SAVE DETAILED CSV
    # --------------------------------------------------------

    details_path = (

        RESULTS_DIR

        / "source_level_overlap_details.csv"

    )


    details_df.to_csv(

        details_path,

        index=False

    )


    # --------------------------------------------------------
    # COUNT AFFECTED IMAGES
    # --------------------------------------------------------

    train_affected = len(

        details_df[
            details_df["dataset"]
            == "train"
        ]

    )


    validation_affected = len(

        details_df[
            details_df["dataset"]
            == "validation"
        ]

    )


    # --------------------------------------------------------
    # CLASS-WISE SUMMARY
    # --------------------------------------------------------

    class_summary = (

        details_df

        .groupby(

            [

                "class",

                "dataset"

            ]

        )

        .size()

        .reset_index(

            name="affected_images"

        )

    )


    summary_path = (

        RESULTS_DIR

        / "source_level_overlap_summary.csv"

    )


    class_summary.to_csv(

        summary_path,

        index=False

    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "LEAKAGE ANALYSIS RESULTS"
    )

    print(
        "=" * 70
    )


    print(

        f"\nOverlapping original sources: "
        f"{len(overlapping_sources)}"
    )


    print(

        f"Affected training images: "
        f"{train_affected}"
    )


    print(

        f"Affected validation images: "
        f"{validation_affected}"
    )


    print()

    print(
        "CLASS-WISE AFFECTED IMAGES"
    )

    print(
        "-" * 70
    )


    print(

        class_summary.to_string(

            index=False

        )

    )


    # --------------------------------------------------------
    # VALIDATION PERCENTAGE AFFECTED
    # --------------------------------------------------------

    total_validation_images = sum(

        1

        for file_path in VAL_DIR.rglob("*")

        if (

            file_path.is_file()

            and file_path.suffix.lower()
            in IMAGE_EXTENSIONS

        )

    )


    affected_percentage = (

        validation_affected

        / total_validation_images

        * 100

    )


    print()

    print(

        f"Total validation images: "
        f"{total_validation_images}"
    )


    print(

        f"Validation images affected by "
        f"source overlap: "
        f"{validation_affected}"
    )


    print(

        f"Validation dataset potentially "
        f"affected: "
        f"{affected_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # FILE LOCATIONS
    # --------------------------------------------------------

    print()

    print(
        "SAVED FILES"
    )

    print(
        "-" * 70
    )


    print(

        f"\n1. Detailed overlap information:\n"
        f"{details_path}"

    )


    print(

        f"\n2. Class-wise overlap summary:\n"
        f"{summary_path}"

    )


    print()

    print(
        "=" * 70
    )

    print(
        "ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()