from pathlib import Path
import re
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

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SPLITS = [
    "train",
    "validation",
    "test"
]


VALID_EXTENSIONS = {
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

    """
    Remove Roboflow augmentation/hash suffix.

    Example:

    4494_jpg.rf.36a86d22d015d849dd1cca246c540d00.jpg

    becomes:

    4494_jpg
    """

    name = Path(filename).stem

    source_name = re.split(
        r"\.rf\.",
        name,
        maxsplit=1
    )[0]

    return source_name


# ============================================================
# COLLECT SOURCE NAMES
# ============================================================

records = []


print("=" * 70)
print("SOURCE-FILENAME CROSS-SPLIT LEAKAGE AUDIT")
print("=" * 70)


for split in SPLITS:

    split_dir = DATA_DIR / split

    print(
        f"\nScanning {split}..."
    )


    image_files = [

        path

        for path in split_dir.rglob("*")

        if path.is_file()

        and path.suffix.lower()
        in VALID_EXTENSIONS
    ]


    print(
        f"Images found: {len(image_files)}"
    )


    for path in image_files:

        source_name = get_source_name(
            path.name
        )


        records.append(

            {
                "split": split,

                "class":
                path.parent.name,

                "file_name":
                path.name,

                "source_name":
                source_name,

                "file_path":
                str(
                    path.resolve()
                )
            }
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


print("\n" + "=" * 70)
print("DATASET SOURCE SUMMARY")
print("=" * 70)


for split in SPLITS:

    split_df = df[
        df["split"] == split
    ]


    print(

        f"\n{split.upper()}"

    )


    print(
        f"Images: {len(split_df)}"
    )


    print(
        "Unique source names: "
        f"{split_df['source_name'].nunique()}"
    )


# ============================================================
# FIND CROSS-SPLIT SOURCE OVERLAPS
# ============================================================

split_sources = {}


for split in SPLITS:

    split_sources[split] = set(

        df.loc[
            df["split"] == split,
            "source_name"
        ]

    )


comparisons = [

    ("train", "validation"),

    ("train", "test"),

    ("validation", "test")
]


overlap_records = []


print("\n" + "=" * 70)
print("CROSS-SPLIT SOURCE OVERLAP")
print("=" * 70)


for split_a, split_b in comparisons:


    common_sources = (

        split_sources[split_a]

        &

        split_sources[split_b]
    )


    print(

        f"\n{split_a.upper()} ↔ "
        f"{split_b.upper()}"

    )


    print(

        "Shared source names: "
        f"{len(common_sources)}"

    )


    for source_name in sorted(
        common_sources
    ):


        source_rows = df[

            (df["source_name"] == source_name)

            &

            (

                df["split"].isin(

                    [
                        split_a,
                        split_b
                    ]

                )

            )

        ]


        splits_found = sorted(

            source_rows[
                "split"
            ].unique()

        )


        classes_found = sorted(

            source_rows[
                "class"
            ].unique()

        )


        overlap_records.append(

            {

                "comparison":
                f"{split_a}_vs_{split_b}",

                "source_name":
                source_name,

                "splits":
                ", ".join(
                    splits_found
                ),

                "classes":
                ", ".join(
                    classes_found
                ),

                "number_of_images":
                len(source_rows)
            }

        )


# ============================================================
# SAVE SOURCE INVENTORY
# ============================================================

inventory_path = (

    RESULTS_DIR

    / "dataset_source_filename_inventory.csv"
)


df.to_csv(

    inventory_path,

    index=False
)


# ============================================================
# SAVE OVERLAPS
# ============================================================

overlap_df = pd.DataFrame(
    overlap_records
)


overlap_path = (

    RESULTS_DIR

    / "cross_split_source_name_overlaps.csv"
)


overlap_df.to_csv(

    overlap_path,

    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)


print(

    "\nTrain ↔ Validation overlaps:",

    len(

        split_sources["train"]

        &

        split_sources["validation"]
    )
)


print(

    "Train ↔ Test overlaps:",

    len(

        split_sources["train"]

        &

        split_sources["test"]
    )
)


print(

    "Validation ↔ Test overlaps:",

    len(

        split_sources["validation"]

        &

        split_sources["test"]
    )
)


total_overlaps = (

    len(
        split_sources["train"]

        &

        split_sources["validation"]
    )

    +

    len(
        split_sources["train"]

        &

        split_sources["test"]
    )

    +

    len(
        split_sources["validation"]

        &

        split_sources["test"]
    )
)


print(

    "\nTotal pairwise source overlaps:",

    total_overlaps
)


if total_overlaps == 0:

    print(

        "\nRESULT: NO SOURCE-FILENAME "
        "OVERLAP DETECTED"

    )

else:

    print(

        "\nWARNING: SOURCE-FILENAME "
        "OVERLAP DETECTED"

    )


print("\nGenerated files:")

print(
    inventory_path
)

print(
    overlap_path
)


print("\nAUDIT COMPLETED")

print("=" * 70)
