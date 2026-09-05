from pathlib import Path
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cross_split_source_family_details.csv"


def extract_source_name(filename):
    """
    Remove the Roboflow-style .rf.<hash> suffix
    to recover the original source filename.
    """

    marker = ".rf."

    if marker in filename:
        return filename.split(marker)[0]

    return Path(filename).stem


def scan_split(split_name, split_dir):

    records = []

    print(f"\nScanning {split_name}...")

    files = list(split_dir.rglob("*"))

    image_files = [
        f for f in files
        if f.is_file()
        and f.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]
    ]

    print(f"Images found: {len(image_files)}")

    for file_path in image_files:

        try:
            class_name = file_path.parent.name

            source_name = extract_source_name(
                file_path.name
            )

            records.append({
                "split": split_name,
                "class": class_name,
                "file_name": file_path.name,
                "source_name": source_name,
                "file_path": str(file_path)
            })

        except Exception as e:

            print(
                f"Error processing {file_path}: {e}"
            )

    return records


print("=" * 85)
print("CROSS-SPLIT SOURCE FAMILY INSPECTION")
print("=" * 85)

train_records = scan_split(
    "train",
    TRAIN_DIR
)

test_records = scan_split(
    "test",
    TEST_DIR
)

train_df = pd.DataFrame(train_records)
test_df = pd.DataFrame(test_records)

print("\n" + "=" * 85)
print("IDENTIFYING SHARED SOURCE FAMILIES")
print("=" * 85)

train_sources = set(
    train_df["source_name"]
)

test_sources = set(
    test_df["source_name"]
)

shared_sources = sorted(
    train_sources.intersection(
        test_sources
    )
)

print(
    f"\nShared train-test source families: "
    f"{len(shared_sources)}"
)

rows = []

for source_name in shared_sources:

    train_variants = train_df[
        train_df["source_name"] == source_name
    ]

    test_variants = test_df[
        test_df["source_name"] == source_name
    ]

    train_classes = sorted(
        train_variants["class"]
        .unique()
        .tolist()
    )

    test_classes = sorted(
        test_variants["class"]
        .unique()
        .tolist()
    )

    rows.append({

        "source_name": source_name,

        "train_variant_count":
            len(train_variants),

        "test_variant_count":
            len(test_variants),

        "total_variant_count":
            len(train_variants)
            + len(test_variants),

        "train_classes":
            " | ".join(train_classes),

        "test_classes":
            " | ".join(test_classes),

        "same_class_across_splits":
            set(train_classes)
            == set(test_classes),

        "example_train_file":
            train_variants
            .iloc[0]["file_name"],

        "example_train_path":
            train_variants
            .iloc[0]["file_path"],

        "example_test_file":
            test_variants
            .iloc[0]["file_name"],

        "example_test_path":
            test_variants
            .iloc[0]["file_path"]

    })


result_df = pd.DataFrame(rows)

result_df = result_df.sort_values(
    [
        "total_variant_count",
        "source_name"
    ],
    ascending=[
        False,
        True
    ]
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 85)
print("SOURCE FAMILY SUMMARY")
print("=" * 85)

print(
    f"\nTotal shared source families: "
    f"{len(result_df)}"
)

print(
    f"Total train variants belonging "
    f"to shared families: "
    f"{result_df['train_variant_count'].sum()}"
)

print(
    f"Total test variants belonging "
    f"to shared families: "
    f"{result_df['test_variant_count'].sum()}"
)


print("\nTop 20 source families by total variants:\n")

display_columns = [

    "source_name",
    "train_variant_count",
    "test_variant_count",
    "total_variant_count",
    "train_classes",
    "test_classes",
    "same_class_across_splits"

]

print(
    result_df[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 85)
print("GENERATED FILE")
print("=" * 85)

print(OUTPUT_FILE)

print("\nANALYSIS COMPLETED")
print("=" * 85)

