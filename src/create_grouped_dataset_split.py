# ============================================================
# CREATE LEAKAGE-SAFE GROUPED DATASET SPLIT
# ============================================================
#
# This script prevents source-level data leakage by ensuring
# that all augmented versions of the same original image are
# placed entirely in either training or validation.
#
# ============================================================

from pathlib import Path
from collections import defaultdict
import random
import shutil
import re

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.80


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent.parent


ORIGINAL_DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "extracted"
    / "data"
)


ORIGINAL_TRAIN_DIR = (
    ORIGINAL_DATA_DIR
    / "train"
)


ORIGINAL_VAL_DIR = (
    ORIGINAL_DATA_DIR
    / "validation"
)


NEW_DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "grouped_split"
)


NEW_TRAIN_DIR = (
    NEW_DATA_DIR
    / "train"
)


NEW_VAL_DIR = (
    NEW_DATA_DIR
    / "val"
)


RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
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
# GET ORIGINAL SOURCE NAME
# ============================================================

def get_source_name(
    file_path
):

    filename = file_path.name


    # Roboflow format:
    #
    # original_name.rf.hash.jpg
    #
    # Example:
    #
    # image_png.rf.123abc.jpg
    #
    # Source:
    #
    # image_png


    source_name = re.split(

        r"\.rf\.",

        filename,

        maxsplit=1
    )[0]


    return source_name


# ============================================================
# GET ALL IMAGE FILES
# ============================================================

def get_image_files(
    directory
):

    image_files = []


    for file_path in directory.iterdir():

        if (

            file_path.is_file()

            and

            file_path.suffix.lower()

            in IMAGE_EXTENSIONS

        ):

            image_files.append(
                file_path
            )


    return image_files


# ============================================================
# COLLECT DATASET
# ============================================================

def collect_class_images(
    class_name
):

    images = []


    train_class_dir = (

        ORIGINAL_TRAIN_DIR
        / class_name

    )


    val_class_dir = (

        ORIGINAL_VAL_DIR
        / class_name

    )


    # --------------------------------------------------------
    # COLLECT ORIGINAL TRAINING IMAGES
    # --------------------------------------------------------

    if train_class_dir.exists():

        for image_path in get_image_files(
            train_class_dir
        ):

            images.append({

                "path": image_path,

                "class": class_name,

                "original_split": "train",

                "source": get_source_name(
                    image_path
                )

            })


    # --------------------------------------------------------
    # COLLECT ORIGINAL VALIDATION IMAGES
    # --------------------------------------------------------

    if val_class_dir.exists():

        for image_path in get_image_files(
            val_class_dir
        ):

            images.append({

                "path": image_path,

                "class": class_name,

                "original_split": "validation",

                "source": get_source_name(
                    image_path
                )

            })


    return images


# ============================================================
# CREATE GROUPED SPLIT
# ============================================================

def split_by_source(
    images
):

    # --------------------------------------------------------
    # GROUP IMAGES BY ORIGINAL SOURCE
    # --------------------------------------------------------

    source_groups = defaultdict(
        list
    )


    for image_info in images:

        source_groups[
            image_info["source"]
        ].append(
            image_info
        )


    # --------------------------------------------------------
    # CREATE SOURCE LIST
    # --------------------------------------------------------

    sources = list(

        source_groups.keys()

    )


    # Reproducible random split

    random.shuffle(
        sources
    )


    # --------------------------------------------------------
    # DETERMINE TARGET NUMBER OF IMAGES
    # --------------------------------------------------------

    total_images = len(
        images
    )


    target_train_images = int(

        total_images
        * TRAIN_RATIO

    )


    train_sources = []

    val_sources = []


    train_image_count = 0


    # --------------------------------------------------------
    # ASSIGN ENTIRE SOURCE GROUPS
    # --------------------------------------------------------

    for source in sources:


        group_size = len(

            source_groups[source]

        )


        if train_image_count < target_train_images:


            train_sources.append(
                source
            )


            train_image_count += (
                group_size
            )


        else:


            val_sources.append(
                source
            )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    overlap = set(
        train_sources
    ).intersection(
        set(val_sources)
    )


    if overlap:

        raise RuntimeError(

            "Source leakage detected "
            "during splitting."

        )


    # --------------------------------------------------------
    # CREATE IMAGE LISTS
    # --------------------------------------------------------

    train_images = []

    val_images = []


    for source in train_sources:


        train_images.extend(

            source_groups[source]

        )


    for source in val_sources:


        val_images.extend(

            source_groups[source]

        )


    return (

        train_images,

        val_images,

        source_groups

    )


# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(
    images,
    destination_root
):

    copied_count = 0


    for image_info in images:


        class_name = (

            image_info["class"]

        )


        destination_dir = (

            destination_root
            / class_name

        )


        destination_dir.mkdir(

            parents=True,

            exist_ok=True

        )


        source_path = (

            image_info["path"]

        )


        destination_path = (

            destination_dir
            / source_path.name

        )


        shutil.copy2(

            source_path,

            destination_path

        )


        copied_count += 1


    return copied_count


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "LEAKAGE-SAFE GROUPED DATASET SPLIT"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # SET RANDOM SEED
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )


    # --------------------------------------------------------
    # CHECK ORIGINAL DATA
    # --------------------------------------------------------

    if not ORIGINAL_TRAIN_DIR.exists():

        print(
            "\nERROR: Original training "
            "directory not found."
        )

        print(
            ORIGINAL_TRAIN_DIR
        )

        return


    if not ORIGINAL_VAL_DIR.exists():

        print(
            "\nERROR: Original validation "
            "directory not found."
        )

        print(
            ORIGINAL_VAL_DIR
        )

        return


    # --------------------------------------------------------
    # GET CLASS NAMES
    # --------------------------------------------------------

    class_names = sorted(

        [

            item.name

            for item in ORIGINAL_TRAIN_DIR.iterdir()

            if item.is_dir()

        ]

    )


    print()

    print(
        "Classes:"
    )


    for class_name in class_names:

        print(
            f"- {class_name}"
        )


    # --------------------------------------------------------
    # CHECK OUTPUT DIRECTORY
    # --------------------------------------------------------

    if NEW_DATA_DIR.exists():

        print()

        print(
            "WARNING: Existing grouped_split "
            "directory found."
        )

        print(
            NEW_DATA_DIR
        )

        print()

        print(
            "Removing old grouped split..."
        )


        shutil.rmtree(
            NEW_DATA_DIR
        )


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    NEW_TRAIN_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    NEW_VAL_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # COLLECT ALL RESULTS
    # --------------------------------------------------------

    all_records = []

    summary_records = []


    total_train = 0

    total_val = 0

    total_sources = 0


    # --------------------------------------------------------
    # PROCESS EACH CLASS
    # --------------------------------------------------------

    for class_name in class_names:


        print()

        print(
            "-" * 70
        )

        print(
            f"Processing: {class_name}"
        )

        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # COLLECT IMAGES
        # ----------------------------------------------------

        images = (

            collect_class_images(
                class_name
            )

        )


        print(

            f"Total images collected: "
            f"{len(images)}"

        )


        # ----------------------------------------------------
        # GROUP AND SPLIT
        # ----------------------------------------------------

        (

            train_images,

            val_images,

            source_groups

        ) = split_by_source(

            images

        )


        unique_sources = len(

            source_groups

        )


        print(

            f"Unique original sources: "
            f"{unique_sources}"

        )


        print(

            f"Training images: "
            f"{len(train_images)}"

        )


        print(

            f"Validation images: "
            f"{len(val_images)}"

        )


        # ----------------------------------------------------
        # COPY TRAINING IMAGES
        # ----------------------------------------------------

        train_copied = (

            copy_images(

                train_images,

                NEW_TRAIN_DIR

            )

        )


        # ----------------------------------------------------
        # COPY VALIDATION IMAGES
        # ----------------------------------------------------

        val_copied = (

            copy_images(

                val_images,

                NEW_VAL_DIR

            )

        )


        # ----------------------------------------------------
        # RECORD SUMMARY
        # ----------------------------------------------------

        summary_records.append({

            "class": class_name,

            "total_images": len(
                images
            ),

            "unique_sources": unique_sources,

            "train_images": train_copied,

            "validation_images": val_copied,

            "train_percentage": round(

                train_copied
                / len(images)
                * 100,

                2

            ),

            "validation_percentage": round(

                val_copied
                / len(images)
                * 100,

                2

            )

        })


        # ----------------------------------------------------
        # RECORD INDIVIDUAL FILES
        # ----------------------------------------------------

        for image_info in train_images:


            all_records.append({

                "filename": image_info[
                    "path"
                ].name,

                "class": class_name,

                "source": image_info[
                    "source"
                ],

                "original_split": image_info[
                    "original_split"
                ],

                "new_split": "train"

            })


        for image_info in val_images:


            all_records.append({

                "filename": image_info[
                    "path"
                ].name,

                "class": class_name,

                "source": image_info[
                    "source"
                ],

                "original_split": image_info[
                    "original_split"
                ],

                "new_split": "validation"

            })


        total_train += (

            train_copied

        )


        total_val += (

            val_copied

        )


        total_sources += (

            unique_sources

        )


    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    summary_df = pd.DataFrame(

        summary_records

    )


    details_df = pd.DataFrame(

        all_records

    )


    # ========================================================
    # VERIFY SOURCE OVERLAP
    # ========================================================

    train_sources = set(

        details_df[

            details_df[
                "new_split"
            ]
            == "train"

        ][
            "source"
        ]

    )


    val_sources = set(

        details_df[

            details_df[
                "new_split"
            ]
            == "validation"

        ][
            "source"
        ]

    )


    overlapping_sources = (

        train_sources.intersection(
            val_sources
        )

    )


    # ========================================================
    # SAVE REPORTS
    # ========================================================

    summary_path = (

        RESULTS_DIR

        / "grouped_split_summary.csv"

    )


    details_path = (

        RESULTS_DIR

        / "grouped_split_details.csv"

    )


    summary_df.to_csv(

        summary_path,

        index=False

    )


    details_df.to_csv(

        details_path,

        index=False

    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "GROUPED SPLIT RESULTS"
    )

    print(
        "=" * 70
    )


    print()

    print(
        summary_df.to_string(
            index=False
        )
    )


    print()

    print(
        f"Total training images: "
        f"{total_train}"
    )


    print(
        f"Total validation images: "
        f"{total_val}"
    )


    print(
        f"Total images: "
        f"{total_train + total_val}"
    )


    print(
        f"Total unique sources: "
        f"{total_sources}"
    )


    print()

    print(
        "SOURCE LEAKAGE VERIFICATION"
    )

    print(
        "-" * 70
    )


    print(

        f"Training sources: "
        f"{len(train_sources)}"

    )


    print(

        f"Validation sources: "
        f"{len(val_sources)}"

    )


    print(

        f"Overlapping sources: "
        f"{len(overlapping_sources)}"

    )


    print()


    if len(overlapping_sources) == 0:


        print(
            "SUCCESS!"
        )

        print(
            "No original image source appears "
            "in both training and validation."
        )


    else:


        print(
            "ERROR!"
        )

        print(
            "Source-level data leakage "
            "still exists."
        )


        print()

        print(

            sorted(
                overlapping_sources
            )

        )


    print()

    print(
        "-" * 70
    )

    print(
        "SAVED REPORTS"
    )

    print(
        "-" * 70
    )


    print()

    print(
        f"1. Split Summary:\n"
        f"{summary_path}"
    )


    print()

    print(
        f"2. Split Details:\n"
        f"{details_path}"
    )


    print()

    print(
        f"New Training Dataset:\n"
        f"{NEW_TRAIN_DIR}"
    )


    print()

    print(
        f"New Validation Dataset:\n"
        f"{NEW_VAL_DIR}"
    )


    print()

    print(
        "=" * 70
    )

    print(
        "GROUPED SPLIT COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()