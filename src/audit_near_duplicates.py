from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
import imagehash


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
# CALCULATE PERCEPTUAL HASH
# ============================================================

def calculate_phash(
    image_path
):

    try:

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            return str(
                imagehash.phash(
                    image
                )
            )

    except Exception as error:

        print(
            f"ERROR reading {image_path}: {error}"
        )

        return None


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hash_distance(
    hash_one,
    hash_two
):

    integer_one = int(
        hash_one,
        16
    )

    integer_two = int(
        hash_two,
        16
    )

    xor_value = (
        integer_one
        ^
        integer_two
    )

    return xor_value.bit_count()


# ============================================================
# COLLECT IMAGE HASHES
# ============================================================

def collect_hashes():

    records = []

    print(
        "\nCOLLECTING PERCEPTUAL HASHES"
    )

    print(
        "-" * 70
    )


    for split_name, split_path in SPLITS.items():

        print(
            f"\nScanning {split_name}..."
        )


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

            perceptual_hash = calculate_phash(

                image_path
            )


            if perceptual_hash is None:

                continue


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

                "phash":
                perceptual_hash
            })


            if (

                index % 500 == 0

                or index == len(image_files)
            ):

                print(

                    f"Processed "

                    f"{index}/{len(image_files)}"
                )


    return pd.DataFrame(
        records
    )


# ============================================================
# ANALYZE CROSS-SPLIT PAIRS
# ============================================================

def compare_splits(
    dataframe,
    split_one,
    split_two,
    threshold
):

    data_one = dataframe[

        dataframe["split"]
        == split_one

    ].reset_index(
        drop=True
    )


    data_two = dataframe[

        dataframe["split"]
        == split_two

    ].reset_index(
        drop=True
    )


    matches = []


    total_comparisons = (

        len(data_one)
        *
        len(data_two)
    )


    print(
        f"\nComparing "
        f"{split_one} ↔ {split_two}"
    )


    print(
        f"Potential comparisons: "
        f"{total_comparisons:,}"
    )


    comparison_count = 0


    for _, row_one in data_one.iterrows():

        for _, row_two in data_two.iterrows():

            distance = hash_distance(

                row_one["phash"],

                row_two["phash"]
            )


            comparison_count += 1


            if distance <= threshold:

                matches.append({

                    "split_1":
                    split_one,

                    "class_1":
                    row_one["class"],

                    "file_1":
                    row_one["file_name"],

                    "path_1":
                    row_one["file_path"],

                    "split_2":
                    split_two,

                    "class_2":
                    row_two["class"],

                    "file_2":
                    row_two["file_name"],

                    "path_2":
                    row_two["file_path"],

                    "phash_distance":
                    distance
                })


        if comparison_count % 500000 == 0:

            print(

                f"Compared "

                f"{comparison_count:,}/"

                f"{total_comparisons:,}"
            )


    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "NEAR-DUPLICATE DATASET LEAKAGE AUDIT"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # HASH ALL IMAGES
    # --------------------------------------------------------

    dataframe = collect_hashes()


    # --------------------------------------------------------
    # SAVE HASHES
    # --------------------------------------------------------

    metrics_dir = (

        PROJECT_DIR
        / "results"
        / "metrics"
    )


    metrics_dir.mkdir(

        parents=True,

        exist_ok=True
    )


    hash_path = (

        metrics_dir
        / "dataset_perceptual_hashes.csv"
    )


    dataframe.to_csv(

        hash_path,

        index=False
    )


    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    HASH_DISTANCE_THRESHOLD = 5


    print(
        "\n" + "=" * 70
    )

    print(
        f"PERCEPTUAL HASH THRESHOLD: "
        f"{HASH_DISTANCE_THRESHOLD}"
    )

    print(
        "Lower distance = visually more similar"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CROSS-SPLIT COMPARISONS
    # --------------------------------------------------------

    all_matches = []


    all_matches.extend(

        compare_splits(

            dataframe,

            "train",

            "validation",

            HASH_DISTANCE_THRESHOLD
        )
    )


    all_matches.extend(

        compare_splits(

            dataframe,

            "train",

            "test",

            HASH_DISTANCE_THRESHOLD
        )
    )


    all_matches.extend(

        compare_splits(

            dataframe,

            "validation",

            "test",

            HASH_DISTANCE_THRESHOLD
        )
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    matches_dataframe = pd.DataFrame(

        all_matches
    )


    matches_path = (

        metrics_dir

        / "cross_split_near_duplicates.csv"
    )


    matches_dataframe.to_csv(

        matches_path,

        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "NEAR-DUPLICATE AUDIT RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"\nTotal images analyzed: "
        f"{len(dataframe)}"
    )


    print(
        f"Near-duplicate threshold: "
        f"{HASH_DISTANCE_THRESHOLD}"
    )


    print(
        f"Potential cross-split matches: "
        f"{len(matches_dataframe)}"
    )


    if len(matches_dataframe) == 0:

        print(
            "\nRESULT: NO STRONG NEAR-DUPLICATES FOUND"
        )

        print(
            "No cross-split image pairs had "
            f"pHash distance <= "
            f"{HASH_DISTANCE_THRESHOLD}"
        )

    else:

        print(
            "\nWARNING: POTENTIAL NEAR-DUPLICATES DETECTED"
        )


        print(
            "\nMatches by split:"
        )


        print(

            matches_dataframe.groupby(

                [
                    "split_1",

                    "split_2"
                ]

            ).size()
        )


        print(
            "\nClosest matches:"
        )


        print(

            matches_dataframe.sort_values(

                "phash_distance"

            ).head(

                20

            ).to_string(

                index=False
            )
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "Generated files:"
    )


    print(
        hash_path
    )


    print(
        matches_path
    )


    print(
        "\nAUDIT COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
