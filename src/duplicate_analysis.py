from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_DIR / "results" / "metrics"

METADATA_FILE = METRICS_DIR / "dataset_metadata.csv"

OUTPUT_FILE = METRICS_DIR / "near_duplicates.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Lower values = more similar images.
# We start conservatively.
HAMMING_THRESHOLD = 5


# ============================================================
# CALCULATE PERCEPTUAL HASH
# ============================================================

def calculate_phash(image_path):

    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            return imagehash.phash(image)

    except Exception as error:

        print(
            f"Could not process {image_path}: {error}"
        )

        return None


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_near_duplicates():

    print("\n" + "=" * 70)
    print("NEAR-DUPLICATE IMAGE ANALYSIS")
    print("=" * 70)

    if not METADATA_FILE.exists():

        print(
            "\nERROR: dataset_metadata.csv not found."
        )

        print(
            "Please run dataset_analysis.py first."
        )

        return

    df = pd.read_csv(METADATA_FILE)

    print(
        f"\nTotal images to analyze: {len(df)}"
    )

    hashes = []

    print(
        "\nCalculating perceptual hashes..."
    )

    for image_path in tqdm(df["filepath"]):

        image_hash = calculate_phash(image_path)

        hashes.append(image_hash)

    df["phash"] = hashes

    df = df.dropna(
        subset=["phash"]
    )

    print(
        "\nSearching for highly similar images..."
    )

    near_duplicate_records = []

    # Convert hashes to a list for comparison
    hash_list = df["phash"].tolist()

    filepaths = df["filepath"].tolist()

    class_names = df["class"].tolist()

    total_images = len(hash_list)

    for i in tqdm(
        range(total_images),
        desc="Comparing images"
    ):

        for j in range(
            i + 1,
            total_images
        ):

            distance = (
                hash_list[i]
                - hash_list[j]
            )

            if distance <= HAMMING_THRESHOLD:

                near_duplicate_records.append({

                    "image_1": filepaths[i],

                    "class_1": class_names[i],

                    "image_2": filepaths[j],

                    "class_2": class_names[j],

                    "hamming_distance": distance
                })

    results_df = pd.DataFrame(
        near_duplicate_records
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nNear-duplicate pairs found: "
        f"{len(results_df)}"
    )

    if len(results_df) > 0:

        print(
            f"\nResults saved to:"
        )

        print(OUTPUT_FILE)

        print(
            "\nDistance interpretation:"
        )

        print(
            "0 = identical perceptual appearance"
        )

        print(
            "1-5 = extremely similar"
        )

    else:

        print(
            "\nNo highly similar image pairs "
            "were found using the selected threshold."
        )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    analyze_near_duplicates()