from pathlib import Path
import shutil
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "dataset_metadata.csv"
)

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)


# ============================================================
# SET RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# CLEAN PREVIOUS SPLIT
# ============================================================

def clean_directory(directory):

    if directory.exists():

        for item in directory.iterdir():

            if item.is_dir():

                shutil.rmtree(item)

            else:

                item.unlink()

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CREATE SPLIT DIRECTORIES
# ============================================================

def create_directories(classes):

    for split in ["train", "val", "test"]:

        split_directory = (
            PROCESSED_DIR
            / split
        )

        split_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        for class_name in classes:

            class_directory = (
                split_directory
                / class_name
            )

            class_directory.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(dataframe, split_name):

    print(
        f"\nCopying {split_name} images..."
    )

    for _, row in dataframe.iterrows():

        source_path = Path(
            row["filepath"]
        )

        class_name = row["class"]

        destination_directory = (
            PROCESSED_DIR
            / split_name
            / class_name
        )

        destination_path = (
            destination_directory
            / source_path.name
        )

        shutil.copy2(
            source_path,
            destination_path
        )


# ============================================================
# MAIN
# ============================================================

def create_dataset_split():

    print("\n" + "=" * 70)
    print("CREATING STRATIFIED DATASET SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK METADATA
    # --------------------------------------------------------

    if not METADATA_FILE.exists():

        print(
            "\nERROR: dataset_metadata.csv "
            "was not found."
        )

        print(
            "Please run dataset_analysis.py first."
        )

        return

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"\nTotal images: {len(df)}"
    )

    print(
        f"Classes: "
        f"{df['class'].nunique()}"
    )

    # --------------------------------------------------------
    # CLEAN OLD SPLITS
    # --------------------------------------------------------

    print(
        "\nCleaning previous split folders..."
    )

    for split in ["train", "val", "test"]:

        clean_directory(
            PROCESSED_DIR
            / split
        )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        test_size=(
            VALIDATION_RATIO
            + TEST_RATIO
        ),

        stratify=df["class"],

        random_state=RANDOM_SEED
    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=(
            TEST_RATIO
            / (
                VALIDATION_RATIO
                + TEST_RATIO
            )
        ),

        stratify=temp_df["class"],

        random_state=RANDOM_SEED
    )

    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    classes = sorted(
        df["class"].unique()
    )

    create_directories(
        classes
    )

    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    copy_images(
        train_df,
        "train"
    )

    copy_images(
        val_df,
        "val"
    )

    copy_images(
        test_df,
        "test"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SPLIT SUMMARY")
    print("=" * 70)

    splits = {

        "Train": train_df,

        "Validation": val_df,

        "Test": test_df
    }

    for split_name, split_df in splits.items():

        print(
            f"\n{split_name}: "
            f"{len(split_df)} images "
            f"({len(split_df) / len(df) * 100:.2f}%)"
        )

        class_counts = (
            split_df["class"]
            .value_counts()
            .sort_index()
        )

        for class_name, count in class_counts.items():

            print(
                f"  {class_name}: "
                f"{count}"
            )

    # --------------------------------------------------------
    # SAVE SPLIT METADATA
    # --------------------------------------------------------

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_metadata = pd.concat(
        [
            train_df,
            val_df,
            test_df
        ]
    )

    output_file = (
        PROJECT_DIR
        / "results"
        / "metrics"
        / "dataset_split_metadata.csv"
    )

    split_metadata.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nSplit metadata saved to:\n"
        f"{output_file}"
    )

    print(
        "\nProcessed dataset location:\n"
        f"{PROCESSED_DIR}"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    create_dataset_split()