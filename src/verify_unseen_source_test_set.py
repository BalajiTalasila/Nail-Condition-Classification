from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "evaluation" / "unseen_source_test"
INVENTORY_FILE = PROJECT_ROOT / "results" / "metrics" / "unseen_source_test_inventory.csv"

print("=" * 80)
print("UNSEEN-SOURCE TEST DATASET VERIFICATION")
print("=" * 80)

print("\nScanning evaluation dataset...")

dataset_records = []

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for image_path in class_dir.rglob("*"):
        if image_path.is_file():
            dataset_records.append({
                "class": class_dir.name,
                "file_name": image_path.name,
                "file_path": str(image_path.resolve())
            })

dataset_df = pd.DataFrame(dataset_records)

print(f"\nTotal images found: {len(dataset_df)}")

print("\nClass distribution:")
print(dataset_df["class"].value_counts().sort_index())

print("\n" + "=" * 80)
print("VERIFYING AGAINST INVENTORY")
print("=" * 80)

inventory_df = pd.read_csv(INVENTORY_FILE)

print(f"\nInventory rows: {len(inventory_df)}")

if len(dataset_df) == len(inventory_df):
    print("PASS: Dataset image count matches inventory.")
else:
    print("WARNING: Dataset image count does not match inventory.")

expected_total = 71

print(f"\nExpected unseen-source images: {expected_total}")
print(f"Actual unseen-source images: {len(dataset_df)}")

if len(dataset_df) == expected_total:
    print("PASS: Correct total number of unseen-source images.")
else:
    print("FAIL: Unexpected number of images.")

print("\n" + "=" * 80)
print("CHECKING FOR SOURCE OVERLAP")
print("=" * 80)

SOURCE_ANALYSIS = PROJECT_ROOT / "results" / "metrics" / "test_source_overlap_analysis.csv"

source_df = pd.read_csv(SOURCE_ANALYSIS)

unseen_source_names = set(
    source_df.loc[
        source_df["source_overlap_with_train"] == False,
        "source_name"
    ].astype(str)
)

print(f"\nUnseen source names in original audit: {len(unseen_source_names)}")

print("\nVerification completed.")
print("=" * 80)
