import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path.cwd()

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

SPLITS = {

    "train":
    DATA_DIR / "train",

    "validation":
    DATA_DIR / "val",

    "test":
    DATA_DIR / "test"
}


IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# HASH FUNCTION
# ============================================================

def calculate_sha256(
    file_path
):

    hasher = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                8192
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# COLLECT IMAGE HASHES
# ============================================================

def collect_image_hashes(
    split_name,
    split_path
):

    records = []

    print(
        f"\nScanning {split_name}..."
    )

    if not split_path.exists():

        print(
            f"ERROR: Directory not found: {split_path}"
        )

        return records

    image_files = [

        path

        for path in split_path.rglob("*")

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    print(
        f"Images found: {len(image_files)}"
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        file_hash = calculate_sha256(
            image_path
        )

        records.append({

            "split":
            split_name,

            "class":
            image_path.parent.name,

            "file_name":
            image_path.name,

            "file_path":
            str(
                image_path.resolve()
            ),

            "sha256":
            file_hash
        })

        if (
            index % 500 == 0
            or index == len(image_files)
        ):

            print(
                f"Processed {index}/{len(image_files)}"
            )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXACT DATASET DUPLICATE AND LEAKAGE AUDIT"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # COLLECT HASHES
    # --------------------------------------------------------

    all_records = []

    for split_name, split_path in SPLITS.items():

        records = collect_image_hashes(

            split_name,

            split_path
        )

        all_records.extend(
            records
        )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        all_records
    )


    results_dir = (

        PROJECT_DIR
        / "results"
        / "metrics"
    )

    results_dir.mkdir(

        parents=True,

        exist_ok=True
    )


    all_hashes_path = (

        results_dir
        / "dataset_image_hashes.csv"
    )

    dataframe.to_csv(

        all_hashes_path,

        index=False
    )


    # --------------------------------------------------------
    # FIND DUPLICATES
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYZING DUPLICATES"
    )

    print(
        "=" * 70
    )


    duplicate_groups = (

        dataframe.groupby(
            "sha256"
        )

        .filter(
            lambda group:
            len(group) > 1
        )

        .sort_values(
            "sha256"
        )
    )


    duplicate_path = (

        results_dir
        / "exact_duplicate_images.csv"
    )

    duplicate_groups.to_csv(

        duplicate_path,

        index=False
    )


    # --------------------------------------------------------
    # SPLIT LEAKAGE ANALYSIS
    # --------------------------------------------------------

    train_hashes = set(

        dataframe[
            dataframe["split"]
            == "train"
        ]["sha256"]
    )


    validation_hashes = set(

        dataframe[
            dataframe["split"]
            == "validation"
        ]["sha256"]
    )


    test_hashes = set(

        dataframe[
            dataframe["split"]
            == "test"
        ]["sha256"]
    )


    train_validation = (

        train_hashes
        &
        validation_hashes
    )


    train_test = (

        train_hashes
        &
        test_hashes
    )


    validation_test = (

        validation_hashes
        &
        test_hashes
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\nDATASET SUMMARY"
    )

    print(
        "-" * 70
    )


    for split_name in SPLITS:

        count = len(

            dataframe[
                dataframe["split"]
                == split_name
            ]

        )

        print(

            f"{split_name.capitalize()}: "

            f"{count} images"

        )


    print(
        "\nEXACT CROSS-SPLIT DUPLICATES"
    )

    print(
        "-" * 70
    )


    print(

        f"Train ↔ Validation: "

        f"{len(train_validation)}"
    )


    print(

        f"Train ↔ Test: "

        f"{len(train_test)}"
    )


    print(

        f"Validation ↔ Test: "

        f"{len(validation_test)}"
    )


    # --------------------------------------------------------
    # LEAKAGE RECORDS
    # --------------------------------------------------------

    leakage_hashes = (

        train_validation
        |
        train_test
        |
        validation_test
    )


    leakage_records = (

        dataframe[
            dataframe["sha256"].isin(
                leakage_hashes
            )
        ]

        .sort_values(
            [
                "sha256",
                "split"
            ]
        )
    )


    leakage_path = (

        results_dir
        / "cross_split_exact_duplicates.csv"
    )


    leakage_records.to_csv(

        leakage_path,

        index=False
    )


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    if len(leakage_hashes) == 0:

        print(
            "RESULT: NO EXACT CROSS-SPLIT DUPLICATES FOUND"
        )

        print(
            "Exact file-level data leakage: NOT DETECTED"
        )

    else:

        print(
            "WARNING: EXACT CROSS-SPLIT DUPLICATES DETECTED"
        )

        print(
            f"Number of duplicated image hashes: "

            f"{len(leakage_hashes)}"
        )


    print(
        "=" * 70
    )


    print(
        "\nGenerated files:"
    )


    print(
        all_hashes_path
    )


    print(
        duplicate_path
    )


    print(
        leakage_path
    )


    print(
        "\nAUDIT COMPLETED"
    )


if __name__ == "__main__":

    main()
