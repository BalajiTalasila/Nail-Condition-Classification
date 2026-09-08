# ============================================================
# SOURCE IMAGE OVERLAP ANALYSIS
# ============================================================

from pathlib import Path
from collections import defaultdict


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "grouped_split" / "train"

VAL_DIR = PROJECT_DIR / "data" / "grouped_split" / "val"


# ============================================================
# GET SOURCE NAME
# ============================================================

def get_source_name(file_path):

    filename = file_path.stem

    # Roboflow-generated filenames commonly contain .rf.
    if ".rf." in filename:

        source_name = filename.split(".rf.")[0]

    else:

        source_name = filename

    return source_name


# ============================================================
# GET IMAGE SOURCES
# ============================================================

def get_sources(dataset_dir):

    sources = defaultdict(list)

    image_extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"

    }


    for file_path in dataset_dir.rglob("*"):

        if (

            file_path.is_file()

            and file_path.suffix.lower()
            in image_extensions

        ):

            source_name = get_source_name(
                file_path
            )

            sources[
                source_name
            ].append(
                str(file_path)
            )


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
        "SOURCE IMAGE OVERLAP ANALYSIS"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # TRAINING SOURCES
    # --------------------------------------------------------

    print(
        "\nScanning training dataset..."
    )


    train_sources = get_sources(
        TRAIN_DIR
    )


    print(

        f"Unique training sources: "
        f"{len(train_sources)}"
    )


    # --------------------------------------------------------
    # VALIDATION SOURCES
    # --------------------------------------------------------

    print(
        "\nScanning validation dataset..."
    )


    val_sources = get_sources(
        VAL_DIR
    )


    print(

        f"Unique validation sources: "
        f"{len(val_sources)}"
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


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "SOURCE OVERLAP RESULTS"
    )

    print(
        "=" * 70
    )


    print(

        f"\nTraining sources: "
        f"{len(train_sources)}"
    )


    print(

        f"Validation sources: "
        f"{len(val_sources)}"
    )


    print(

        f"\nOverlapping sources: "
        f"{len(overlapping_sources)}"
    )


    # --------------------------------------------------------
    # DISPLAY OVERLAPS
    # --------------------------------------------------------

    if overlapping_sources:

        print()

        print(
            "WARNING: POSSIBLE SOURCE-LEVEL "
            "DATA LEAKAGE DETECTED"
        )

        print()

        print(
            "Sources appearing in both "
            "training and validation:"
        )


        for index, source in enumerate(

            sorted(
                overlapping_sources
            ),

            start=1

        ):

            print()

            print(

                f"{index}. "
                f"{source}"
            )

            print(

                f"   Training images: "
                f"{len(train_sources[source])}"
            )

            print(

                f"   Validation images: "
                f"{len(val_sources[source])}"
            )


    else:

        print()

        print(
            "GOOD NEWS!"
        )

        print()

        print(
            "No original image sources appear "
            "to overlap between training and "
            "validation datasets."
        )


    print()

    print(
        "=" * 70
    )

    print(
        "SOURCE OVERLAP ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()