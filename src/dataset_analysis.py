import os
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "extracted"
RESULTS_DIR = PROJECT_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FIND DATASET ROOT
# ============================================================

def find_class_directories(data_dir):
    """
    Searches recursively for directories that contain images.
    """

    class_dirs = []

    for directory in data_dir.rglob("*"):

        if directory.is_dir():

            image_files = [
                file
                for file in directory.iterdir()
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if image_files:
                class_dirs.append(directory)

    return class_dirs


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path):
    """
    Calculates SHA-256 hash to identify exact duplicate files.
    """

    hash_object = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hash_object.update(chunk)

    return hash_object.hexdigest()


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_dataset():

    print("\n" + "=" * 70)
    print("NAIL DISEASE DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nProject directory: {PROJECT_DIR}")
    print(f"Dataset directory: {DATA_DIR}")

    if not DATA_DIR.exists():

        print("\nERROR: Extracted dataset directory not found.")

        return

    print("\nSearching for class directories...")

    class_directories = find_class_directories(DATA_DIR)

    if not class_directories:

        print("\nERROR: No directories containing images were found.")

        return

    print(f"\nFound {len(class_directories)} directories containing images:\n")

    for directory in class_directories:

        print(directory)

    records = []

    corrupted_images = []

    file_hashes = {}

    print("\nAnalyzing images...\n")

    for class_directory in class_directories:

        class_name = class_directory.name

        image_files = [
            file
            for file in class_directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"Processing class: {class_name}")

        for image_path in tqdm(image_files):

            try:

                with Image.open(image_path) as image:

                    image.verify()

                with Image.open(image_path) as image:

                    width, height = image.size
                    image_format = image.format

                file_hash = calculate_file_hash(image_path)

                record = {
                    "filepath": str(image_path),
                    "filename": image_path.name,
                    "class": class_name,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "file_size_kb": round(
                        image_path.stat().st_size / 1024,
                        2
                    ),
                    "sha256": file_hash
                }

                records.append(record)

                if file_hash not in file_hashes:

                    file_hashes[file_hash] = []

                file_hashes[file_hash].append(str(image_path))

            except Exception as error:

                corrupted_images.append({
                    "filepath": str(image_path),
                    "error": str(error)
                })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(records)

    if df.empty:

        print("\nERROR: No valid images were analyzed.")

        return

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    total_images = len(df)

    print(f"\nTotal valid images: {total_images}")

    print("\nClass Distribution:")

    class_counts = df["class"].value_counts()

    for class_name, count in class_counts.items():

        percentage = (count / total_images) * 100

        print(
            f"{class_name}: "
            f"{count} images "
            f"({percentage:.2f}%)"
        )

    print("\nImage Formats:")

    print(df["format"].value_counts())

    print("\nImage Dimension Statistics:")

    print(
        df[["width", "height"]]
        .describe()
    )

    # ========================================================
    # EXACT DUPLICATE ANALYSIS
    # ========================================================

    duplicate_groups = {
        file_hash: paths
        for file_hash, paths in file_hashes.items()
        if len(paths) > 1
    }

    total_duplicate_images = sum(
        len(paths) - 1
        for paths in duplicate_groups.values()
    )

    print("\n" + "=" * 70)
    print("EXACT DUPLICATE ANALYSIS")
    print("=" * 70)

    print(
        f"\nNumber of duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Total duplicate images: "
        f"{total_duplicate_images}"
    )

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    dataset_csv = RESULTS_DIR / "dataset_metadata.csv"

    df.to_csv(
        dataset_csv,
        index=False
    )

    class_distribution_csv = (
        RESULTS_DIR /
        "class_distribution.csv"
    )

    class_distribution = (
        df["class"]
        .value_counts()
        .reset_index()
    )

    class_distribution.columns = [
        "class",
        "image_count"
    ]

    class_distribution[
        "percentage"
    ] = (
        class_distribution[
            "image_count"
        ]
        / total_images
        * 100
    )

    class_distribution.to_csv(
        class_distribution_csv,
        index=False
    )

    # ========================================================
    # CORRUPTED IMAGES REPORT
    # ========================================================

    corrupted_csv = (
        RESULTS_DIR /
        "corrupted_images.csv"
    )

    pd.DataFrame(
        corrupted_images
    ).to_csv(
        corrupted_csv,
        index=False
    )

    # ========================================================
    # DUPLICATES REPORT
    # ========================================================

    duplicate_records = []

    for file_hash, paths in duplicate_groups.items():

        for path in paths:

            duplicate_records.append({
                "sha256": file_hash,
                "filepath": path
            })

    duplicates_csv = (
        RESULTS_DIR /
        "exact_duplicates.csv"
    )

    pd.DataFrame(
        duplicate_records
    ).to_csv(
        duplicates_csv,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nValid images: {total_images}")

    print(
        f"Corrupted images: "
        f"{len(corrupted_images)}"
    )

    print(
        f"Duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Duplicate images: "
        f"{total_duplicate_images}"
    )

    print("\nReports saved to:")

    print(RESULTS_DIR)

    print("\nGenerated files:")

    print("- dataset_metadata.csv")
    print("- class_distribution.csv")
    print("- corrupted_images.csv")
    print("- exact_duplicates.csv")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    analyze_dataset()