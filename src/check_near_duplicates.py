# ============================================================
# NEAR-DUPLICATE IMAGE AND DATA LEAKAGE CHECK
# USING PERCEPTUAL HASHING
# ============================================================

from pathlib import Path

from PIL import Image
import imagehash


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
    / "grouped_split"
)


TRAIN_DIR = (
    DATA_DIR
    / "train"
)


VAL_DIR = (
    DATA_DIR
    / "val"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",

}


# Lower value = images must be more similar
#
# Recommended interpretation:
#
# 0      = almost identical
# 1 - 5  = extremely similar
# 6 - 10 = potentially similar
# > 10   = generally different
#
# We will initially use a strict threshold.

SIMILARITY_THRESHOLD = 5


# ============================================================
# GET IMAGE FILES
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
# CALCULATE PERCEPTUAL HASH
# ============================================================

def calculate_image_hash(
    image_path,
):

    try:

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )


            return imagehash.phash(
                image
            )


    except Exception as error:

        print()

        print(
            f"WARNING: Could not process:"
        )

        print(
            image_path
        )

        print(
            f"Error: {error}"
        )

        return None


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()

    print(
        "=" * 70
    )

    print(
        "NEAR-DUPLICATE IMAGE DATA LEAKAGE CHECK"
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
    # LOAD IMAGE FILES
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
    # CALCULATE TRAINING HASHES
    # --------------------------------------------------------

    print()

    print(
        "Calculating perceptual hashes "
        "for training images..."
    )


    train_hashes = []


    for index, image_path in enumerate(
        train_images,
        start=1,
    ):

        image_hash = calculate_image_hash(
            image_path
        )


        if image_hash is not None:

            train_hashes.append(

                (
                    image_path,
                    image_hash,
                )

            )


        if index % 500 == 0:

            print(
                f"Processed {index} "
                f"of {len(train_images)} training images..."
            )


    # --------------------------------------------------------
    # CHECK VALIDATION IMAGES
    # --------------------------------------------------------

    print()

    print(
        "Calculating validation image hashes "
        "and comparing images..."
    )


    near_duplicates = []


    for index, val_image_path in enumerate(
        val_images,
        start=1,
    ):

        val_hash = calculate_image_hash(
            val_image_path
        )


        if val_hash is None:

            continue


        best_match = None

        lowest_distance = float(
            "inf"
        )


        for train_image_path, train_hash in train_hashes:

            distance = (

                val_hash
                -
                train_hash

            )


            if distance < lowest_distance:

                lowest_distance = distance

                best_match = train_image_path


        # ----------------------------------------------------
        # STORE POTENTIAL MATCH
        # ----------------------------------------------------

        if (

            lowest_distance
            <= SIMILARITY_THRESHOLD

        ):

            near_duplicates.append(

                {

                    "validation_image": val_image_path,

                    "training_image": best_match,

                    "hash_distance": lowest_distance,

                }

            )


        print(

            f"[{index}/{len(val_images)}] "
            f"Closest distance: "
            f"{lowest_distance}"

        )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "NEAR-DUPLICATE CHECK RESULTS"
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
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )


    print(
        f"Potential near-duplicates found: "
        f"{len(near_duplicates)}"
    )


    # --------------------------------------------------------
    # DISPLAY MATCHES
    # --------------------------------------------------------

    if len(
        near_duplicates
    ) > 0:

        print()

        print(
            "POTENTIAL DATA LEAKAGE DETECTED"
        )


        print()

        print(
            "-" * 70
        )


        for index, match in enumerate(
            near_duplicates,
            start=1,
        ):

            print()

            print(
                f"Potential Match {index}"
            )


            print()

            print(
                "Validation image:"
            )

            print(
                match[
                    "validation_image"
                ]
            )


            print()

            print(
                "Closest training image:"
            )

            print(
                match[
                    "training_image"
                ]
            )


            print()

            print(
                "Perceptual hash distance:"
            )

            print(
                match[
                    "hash_distance"
                ]
            )


            print()

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
            "No highly similar images were "
            "detected between training and "
            "validation datasets."
        )


    print()

    print(
        "=" * 70
    )

    print(
        "NEAR-DUPLICATE CHECK COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()