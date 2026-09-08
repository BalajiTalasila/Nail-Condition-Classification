# ============================================================
# DUPLICATE IMAGE AND DATA LEAKAGE CHECK
# NAIL CONDITION CLASSIFICATION PROJECT
# ============================================================

from pathlib import Path
import hashlib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)


TRAIN_DIR = PROJECT_DIR / "data" / "grouped_split" / "train"
VAL_DIR = PROJECT_DIR / "data" / "grouped_split" / "val"


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",

}


# ============================================================
# CALCULATE FILE HASH
# ============================================================

def calculate_file_hash(
    file_path,
):

    hash_object = hashlib.md5()


    with open(
        file_path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                8192
            )


            if not chunk:

                break


            hash_object.update(
                chunk
            )


    return hash_object.hexdigest()


# ============================================================
# GET ALL IMAGES
# ============================================================

def get_image_files(
    directory,
):

    image_files = []


    for file_path in directory.rglob(
        "*"
    ):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            image_files.append(
                file_path
            )


    return image_files


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "DUPLICATE IMAGE AND DATA LEAKAGE CHECK"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    if not TRAIN_DIR.exists():

        print()

        print(
            "ERROR: Training directory not found."
        )

        print(
            TRAIN_DIR
        )

        return


    if not VAL_DIR.exists():

        print()

        print(
            "ERROR: Validation directory not found."
        )

        print(
            VAL_DIR
        )

        return


    # --------------------------------------------------------
    # GET IMAGE FILES
    # --------------------------------------------------------

    print()

    print(
        "Scanning training images..."
    )


    train_images = get_image_files(
        TRAIN_DIR
    )


    print(
        f"Training images found: "
        f"{len(train_images)}"
    )


    print()

    print(
        "Scanning validation images..."
    )


    val_images = get_image_files(
        VAL_DIR
    )


    print(
        f"Validation images found: "
        f"{len(val_images)}"
    )


    # --------------------------------------------------------
    # CALCULATE TRAIN HASHES
    # --------------------------------------------------------

    print()

    print(
        "Calculating training image hashes..."
    )


    train_hashes = {}


    for image_path in train_images:

        image_hash = calculate_file_hash(
            image_path
        )


        if image_hash not in train_hashes:

            train_hashes[
                image_hash
            ] = []


        train_hashes[
            image_hash
        ].append(
            image_path
        )


    # --------------------------------------------------------
    # CHECK VALIDATION IMAGES
    # --------------------------------------------------------

    print(
        "Checking validation images..."
    )


    duplicate_pairs = []


    for val_image in val_images:

        val_hash = calculate_file_hash(
            val_image
        )


        if val_hash in train_hashes:

            for train_image in train_hashes[
                val_hash
            ]:

                duplicate_pairs.append(

                    (
                        train_image,
                        val_image,
                    )

                )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DATA LEAKAGE CHECK RESULTS"
    )

    print(
        "=" * 70
    )


    print()

    print(
        f"Training images checked: "
        f"{len(train_images)}"
    )


    print(
        f"Validation images checked: "
        f"{len(val_images)}"
    )


    print()

    print(
        f"Exact duplicates found: "
        f"{len(duplicate_pairs)}"
    )


    # --------------------------------------------------------
    # PRINT DUPLICATES
    # --------------------------------------------------------

    if len(
        duplicate_pairs
    ) > 0:

        print()

        print(
            "WARNING: POSSIBLE DATA LEAKAGE DETECTED!"
        )

        print()

        print(
            "-" * 70
        )


        for index, (
            train_image,
            val_image,
        ) in enumerate(

            duplicate_pairs,
            start=1,

        ):

            print()

            print(
                f"Duplicate {index}"
            )


            print(
                f"Training Image:"
            )


            print(
                train_image
            )


            print(
                f"Validation Image:"
            )


            print(
                val_image
            )


            print(
                "-" * 70
            )


    else:

        print()

        print(
            "GOOD NEWS!"
        )


        print()

        print(
            "No exact duplicate images were found "
            "between the training and validation datasets."
        )


    print()

    print(
        "=" * 70
    )

    print(
        "DUPLICATE CHECK COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()