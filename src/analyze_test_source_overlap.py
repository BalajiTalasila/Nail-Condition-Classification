import os
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


TRAIN_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "train"
)


TEST_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "test"
)


RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "metrics"
)


os.makedirs(
    RESULTS_DIR,
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
# EXTRACT ORIGINAL SOURCE NAME
# ============================================================

def get_source_name(filename):

    name_without_extension = os.path.splitext(
        filename
    )[0]


    source_name = re.sub(
        r"\.rf\.[a-fA-F0-9]+$",
        "",
        name_without_extension
    )


    return source_name


# ============================================================
# COLLECT DATASET INFORMATION
# ============================================================

def collect_images(
    dataset_dir,
    split_name
):

    records = []


    for root, _, files in os.walk(
        dataset_dir
    ):

        for file_name in files:

            extension = os.path.splitext(
                file_name
            )[1].lower()


            if extension not in IMAGE_EXTENSIONS:
                continue


            file_path = os.path.join(
                root,
                file_name
            )


            class_name = os.path.basename(
                os.path.dirname(
                    file_path
                )
            )


            source_name = get_source_name(
                file_name
            )


            records.append({

                "split": split_name,

                "class":
                    class_name,

                "file_name":
                    file_name,

                "source_name":
                    source_name,

                "file_path":
                    file_path

            })


    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TEST DATASET SOURCE OVERLAP ANALYSIS"
    )

    print("=" * 70)


    print(
        "\nScanning training dataset..."
    )


    train_records = collect_images(

        TRAIN_DIR,

        "train"

    )


    print(
        f"Training images: {len(train_records)}"
    )


    print(
        "\nScanning test dataset..."
    )


    test_records = collect_images(

        TEST_DIR,

        "test"

    )


    print(
        f"Test images: {len(test_records)}"
    )


    # --------------------------------------------------------
    # DATAFRAMES
    # --------------------------------------------------------

    train_df = pd.DataFrame(
        train_records
    )


    test_df = pd.DataFrame(
        test_records
    )


    # --------------------------------------------------------
    # SOURCE SETS
    # --------------------------------------------------------

    train_sources = set(

        train_df[
            "source_name"
        ]

    )


    test_sources = set(

        test_df[
            "source_name"
        ]

    )


    overlapping_sources = (

        train_sources.intersection(

            test_sources

        )

    )


    # --------------------------------------------------------
    # MARK TEST IMAGES
    # --------------------------------------------------------

    test_df[
        "source_overlap_with_train"
    ] = test_df[
        "source_name"
    ].isin(
        overlapping_sources
    )


    overlapping_test_images = test_df[
        test_df[
            "source_overlap_with_train"
        ]
    ]


    unseen_test_images = test_df[
        ~test_df[
            "source_overlap_with_train"
        ]
    ]


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_test_images = len(
        test_df
    )


    overlap_count = len(
        overlapping_test_images
    )


    unseen_count = len(
        unseen_test_images
    )


    print("\n")

    print("=" * 70)

    print(
        "TEST IMAGE SOURCE OVERLAP SUMMARY"
    )

    print("=" * 70)


    print(

        f"\nTotal test images: "

        f"{total_test_images}"

    )


    print(

        f"Test images with a source "

        f"also present in training: "

        f"{overlap_count}"

    )


    print(

        f"Test images from unseen sources: "

        f"{unseen_count}"

    )


    print(

        f"\nPercentage of test images "

        f"with train-source overlap: "

        f"{(overlap_count / total_test_images) * 100:.2f}%"

    )


    print(

        f"Percentage of truly unseen-source "

        f"test images: "

        f"{(unseen_count / total_test_images) * 100:.2f}%"

    )


    # --------------------------------------------------------
    # CLASS-WISE ANALYSIS
    # --------------------------------------------------------

    class_summary = (

        test_df.groupby(
            "class"
        )

        .agg(

            total_test_images=(

                "file_name",

                "count"

            ),

            overlapping_images=(

                "source_overlap_with_train",

                "sum"

            )

        )

        .reset_index()

    )


    class_summary[
        "unseen_images"
    ] = (

        class_summary[
            "total_test_images"
        ]

        -

        class_summary[
            "overlapping_images"
        ]

    )


    class_summary[
        "overlap_percentage"
    ] = (

        class_summary[
            "overlapping_images"
        ]

        /

        class_summary[
            "total_test_images"
        ]

        *

        100

    )


    print("\n")

    print("=" * 70)

    print(
        "CLASS-WISE TEST SOURCE OVERLAP"
    )

    print("=" * 70)

    print(

        class_summary.to_string(

            index=False

        )

    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_path = os.path.join(

        RESULTS_DIR,

        "test_source_overlap_analysis.csv"

    )


    test_df.to_csv(

        output_path,

        index=False

    )


    class_output_path = os.path.join(

        RESULTS_DIR,

        "test_source_overlap_class_summary.csv"

    )


    class_summary.to_csv(

        class_output_path,

        index=False

    )


    print("\n")

    print("=" * 70)

    print(
        "GENERATED FILES"
    )

    print("=" * 70)


    print(
        output_path
    )


    print(
        class_output_path
    )


    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":

    main()
