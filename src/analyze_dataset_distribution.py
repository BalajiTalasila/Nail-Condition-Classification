# ============================================================
# DATASET DISTRIBUTION ANALYSIS
# SIX-CLASS NAIL CONDITION CLASSIFICATION
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

TRAIN_DIR = (
    DATA_DIR
    / "train"
)

VAL_DIR = (
    DATA_DIR
    / "val"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"

}


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(folder):

    count = 0

    for file_path in folder.rglob("*"):

        if (
            file_path.is_file()
            and
            file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            count += 1

    return count


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset():

    print()

    print(
        "=" * 70
    )

    print(
        "DATASET DISTRIBUTION ANALYSIS"
    )

    print(
        "=" * 70
    )

    print()


    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    if not TRAIN_DIR.exists():

        print(
            "ERROR: Training directory not found."
        )

        print(
            TRAIN_DIR
        )

        return


    if not VAL_DIR.exists():

        print(
            "ERROR: Validation directory not found."
        )

        print(
            VAL_DIR
        )

        return


    # --------------------------------------------------------
    # GET CLASS NAMES
    # --------------------------------------------------------

    train_classes = sorted(

        [

            folder.name

            for folder in TRAIN_DIR.iterdir()

            if folder.is_dir()

        ]

    )


    val_classes = sorted(

        [

            folder.name

            for folder in VAL_DIR.iterdir()

            if folder.is_dir()

        ]

    )


    # --------------------------------------------------------
    # VERIFY CLASS CONSISTENCY
    # --------------------------------------------------------

    if train_classes != val_classes:

        print()

        print(
            "WARNING:"
        )

        print(
            "Training and validation "
            "class folders do not match."
        )

        print()

        print(
            f"Training classes: {train_classes}"
        )

        print(
            f"Validation classes: {val_classes}"
        )

        return


    class_names = train_classes


    # --------------------------------------------------------
    # COUNT IMAGES PER CLASS
    # --------------------------------------------------------

    dataset_data = []


    print(
        "Counting images..."
    )

    print()


    for class_name in class_names:


        train_count = count_images(

            TRAIN_DIR
            / class_name

        )


        val_count = count_images(

            VAL_DIR
            / class_name

        )


        total_count = (

            train_count
            + val_count

        )


        train_percentage = (

            train_count
            / total_count
            * 100

        )


        val_percentage = (

            val_count
            / total_count
            * 100

        )


        dataset_data.append(

            {

                "class": class_name,

                "train_images": train_count,

                "validation_images": val_count,

                "total_images": total_count,

                "train_percentage": (
                    round(
                        train_percentage,
                        2
                    )
                ),

                "validation_percentage": (
                    round(
                        val_percentage,
                        2
                    )
                )

            }

        )


        print(

            f"{class_name}: "
            f"Train={train_count}, "
            f"Validation={val_count}, "
            f"Total={total_count}"

        )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    dataset_df = pd.DataFrame(

        dataset_data

    )


    total_train = dataset_df[
        "train_images"
    ].sum()


    total_val = dataset_df[
        "validation_images"
    ].sum()


    total_images = dataset_df[
        "total_images"
    ].sum()


    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 70
    )

    print()


    print(

        f"Total Training Images: "
        f"{total_train}"

    )


    print(

        f"Total Validation Images: "
        f"{total_val}"

    )


    print(

        f"Total Images: "
        f"{total_images}"

    )


    print(

        f"\nTraining Percentage: "
        f"{(total_train / total_images * 100):.2f}%"

    )


    print(

        f"Validation Percentage: "
        f"{(total_val / total_images * 100):.2f}%"

    )


    # --------------------------------------------------------
    # CREATE RESULTS DIRECTORY
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = (

        RESULTS_DIR

        / "dataset_distribution.csv"

    )


    dataset_df.to_csv(

        csv_path,

        index=False

    )


    # --------------------------------------------------------
    # CREATE BAR CHART
    # --------------------------------------------------------

    plt.figure(

        figsize=(14, 7)

    )


    x_positions = range(

        len(
            dataset_df
        )

    )


    plt.bar(

        x_positions,

        dataset_df[
            "train_images"
        ],

        label="Training Images"

    )


    plt.bar(

        x_positions,

        dataset_df[
            "validation_images"
        ],

        bottom=dataset_df[
            "train_images"
        ],

        label="Validation Images"

    )


    plt.xticks(

        x_positions,

        dataset_df[
            "class"
        ],

        rotation=25,

        ha="right"

    )


    plt.ylabel(

        "Number of Images"

    )


    plt.xlabel(

        "Nail Condition Class"

    )


    plt.title(

        "Dataset Distribution Across Classes"

    )


    plt.legend()


    plt.tight_layout()


    chart_path = (

        RESULTS_DIR

        / "dataset_distribution.png"

    )


    plt.savefig(

        chart_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # --------------------------------------------------------
    # CREATE TRAIN VS VALIDATION CHART
    # --------------------------------------------------------

    plt.figure(

        figsize=(14, 7)

    )


    bar_width = 0.35


    x_positions = list(

        range(
            len(dataset_df)
        )

    )


    train_positions = [

        x - bar_width / 2

        for x in x_positions

    ]


    validation_positions = [

        x + bar_width / 2

        for x in x_positions

    ]


    plt.bar(

        train_positions,

        dataset_df[
            "train_images"
        ],

        width=bar_width,

        label="Training"

    )


    plt.bar(

        validation_positions,

        dataset_df[
            "validation_images"
        ],

        width=bar_width,

        label="Validation"

    )


    plt.xticks(

        x_positions,

        dataset_df[
            "class"
        ],

        rotation=25,

        ha="right"

    )


    plt.ylabel(

        "Number of Images"

    )


    plt.xlabel(

        "Nail Condition Class"

    )


    plt.title(

        "Training vs Validation Dataset Distribution"

    )


    plt.legend()


    plt.tight_layout()


    comparison_chart_path = (

        RESULTS_DIR

        / "train_vs_validation_distribution.png"

    )


    plt.savefig(

        comparison_chart_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DATASET ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Generated files:"
    )

    print()

    print(
        f"1. Dataset Distribution CSV:\n"
        f"{csv_path}"
    )

    print()

    print(
        f"2. Dataset Distribution Chart:\n"
        f"{chart_path}"
    )

    print()

    print(
        f"3. Training vs Validation Chart:\n"
        f"{comparison_chart_path}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    analyze_dataset()