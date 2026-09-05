from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OVERLAP_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "test_source_overlap_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "unseen_source_test"
)

print("=" * 80)
print("CREATING UNSEEN-SOURCE TEST DATASET")
print("=" * 80)

print("\nLoading source overlap analysis...")

df = pd.read_csv(OVERLAP_FILE)

print(f"Total rows in analysis: {len(df)}")
print("\nColumns:")
print(list(df.columns))

unseen_df = df[
    df["source_overlap_with_train"] == False
].copy()

print("\n" + "=" * 80)
print("UNSEEN-SOURCE DATASET SUMMARY")
print("=" * 80)

print(f"\nTotal unseen-source images: {len(unseen_df)}")

print("\nClass distribution:")

class_counts = unseen_df["class"].value_counts().sort_index()

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("COPYING UNSEEN-SOURCE IMAGES")
print("=" * 80)

copied = 0
missing = 0

for _, row in unseen_df.iterrows():

    source_path = Path(row["file_path"])
    class_name = row["class"]

    destination_class_dir = OUTPUT_DIR / class_name

    destination_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_path = (
        destination_class_dir
        / source_path.name
    )

    if not source_path.exists():

        print(f"\nWARNING: FILE NOT FOUND\n{source_path}")

        missing += 1
        continue

    shutil.copy2(
        source_path,
        destination_path
    )

    copied += 1

inventory_file = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "unseen_source_test_inventory.csv"
)

unseen_df.to_csv(
    inventory_file,
    index=False
)

print("\n" + "=" * 80)
print("COPYING COMPLETED")
print("=" * 80)

print(f"\nSuccessfully copied: {copied}")
print(f"Missing files: {missing}")

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nInventory saved:")
print(inventory_file)

print("\n" + "=" * 80)
print("PROCESS COMPLETED")
print("=" * 80)
