# ============================================================
# SOURCE-LEVEL PERFORMANCE ANALYSIS
# EFFICIENTNET-B0 - LEAKAGE-SAFE GROUPED DATASET
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = (
    PROJECT_DIR
    / "results"
    / "metrics"
)

PREDICTIONS_PATH = (
    METRICS_DIR
    / "efficientnet_b0_grouped_predictions.csv"
)


# ============================================================
# GET ORIGINAL SOURCE NAME
# ============================================================

def get_source_name(image_name):

    filename = Path(image_name).stem

    # Roboflow-generated augmented filenames commonly contain .rf.
    if ".rf." in filename:

        return filename.split(".rf.")[0]

    return filename


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "SOURCE-LEVEL PERFORMANCE ANALYSIS"
    )

    print(
        "EFFICIENTNET-B0 - LEAKAGE-SAFE GROUPED DATASET"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nLoading prediction data..."
    )

    predictions_df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        f"Total image-level predictions: "
        f"{len(predictions_df)}"
    )


    # --------------------------------------------------------
    # CREATE SOURCE COLUMN
    # --------------------------------------------------------

    print(
        "\nIdentifying original image sources..."
    )

    predictions_df["source_name"] = (
        predictions_df["image_name"].apply(
            get_source_name
        )
    )


    # --------------------------------------------------------
    # BASIC SOURCE STATISTICS
    # --------------------------------------------------------

    unique_sources = (
        predictions_df["source_name"].nunique()
    )

    print(
        f"Unique original validation sources: "
        f"{unique_sources}"
    )

    print(
        f"Average images per source: "
        f"{len(predictions_df) / unique_sources:.2f}"
    )


    # --------------------------------------------------------
    # VERIFY SOURCE CLASS CONSISTENCY
    # --------------------------------------------------------

    source_class_counts = (
        predictions_df
        .groupby("source_name")["true_class"]
        .nunique()
    )

    inconsistent_sources = (
        source_class_counts[
            source_class_counts > 1
        ]
    )

    print(
        f"\nSources with inconsistent labels: "
        f"{len(inconsistent_sources)}"
    )

    if len(inconsistent_sources) == 0:

        print(
            "Source label consistency: PASSED"
        )

    else:

        print(
            "WARNING: Some sources have "
            "multiple class labels."
        )


    # --------------------------------------------------------
    # IMAGE-LEVEL PERFORMANCE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "IMAGE-LEVEL PERFORMANCE"
    )

    print(
        "-" * 70
    )

    total_images = len(
        predictions_df
    )

    correct_images = (
        predictions_df[
            "correct_prediction"
        ].sum()
    )

    image_accuracy = (
        correct_images
        / total_images
        * 100
    )

    print(
        f"Total images: {total_images}"
    )

    print(
        f"Correct predictions: "
        f"{correct_images}"
    )

    print(
        f"Image-level accuracy: "
        f"{image_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SOURCE-LEVEL AGGREGATION
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "SOURCE-LEVEL PERFORMANCE"
    )

    print(
        "-" * 70
    )


    source_results = []

    grouped_sources = (
        predictions_df.groupby(
            "source_name"
        )
    )


    for source_name, group in grouped_sources:

        true_classes = (
            group["true_class"].unique()
        )

        true_class = (
            true_classes[0]
        )

        total_variants = (
            len(group)
        )

        correct_variants = (
            group[
                "correct_prediction"
            ].sum()
        )

        incorrect_variants = (
            total_variants
            - correct_variants
        )

        source_accuracy = (
            correct_variants
            / total_variants
            * 100
        )


        # Majority vote prediction
        predicted_counts = (
            group[
                "predicted_class"
            ]
            .value_counts()
        )

        majority_prediction = (
            predicted_counts.index[0]
        )

        majority_vote_correct = (
            majority_prediction
            == true_class
        )


        # Average prediction confidence
        average_confidence = (
            group[
                "confidence_percent"
            ].mean()
        )


        # Source-level status
        if correct_variants == total_variants:

            status = "ALL_CORRECT"

        elif correct_variants == 0:

            status = "ALL_INCORRECT"

        else:

            status = "MIXED"


        source_results.append({

            "source_name":
                source_name,

            "true_class":
                true_class,

            "total_variants":
                total_variants,

            "correct_variants":
                correct_variants,

            "incorrect_variants":
                incorrect_variants,

            "source_accuracy_percent":
                source_accuracy,

            "majority_prediction":
                majority_prediction,

            "majority_vote_correct":
                majority_vote_correct,

            "average_confidence_percent":
                average_confidence,

            "source_status":
                status

        })


    source_results_df = pd.DataFrame(
        source_results
    )


    # --------------------------------------------------------
    # MAJORITY-VOTE SOURCE ACCURACY
    # --------------------------------------------------------

    total_sources = len(
        source_results_df
    )

    correct_sources = (
        source_results_df[
            "majority_vote_correct"
        ].sum()
    )

    source_accuracy = (
        correct_sources
        / total_sources
        * 100
    )


    print(
        f"Total unique sources: "
        f"{total_sources}"
    )

    print(
        f"Correct majority-vote sources: "
        f"{correct_sources}"
    )

    print(
        f"Source-level majority-vote accuracy: "
        f"{source_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SOURCE STATUS DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "SOURCE PREDICTION STATUS"
    )

    print(
        "-" * 70
    )


    status_counts = (
        source_results_df[
            "source_status"
        ]
        .value_counts()
    )


    for status, count in status_counts.items():

        percentage = (
            count
            / total_sources
            * 100
        )

        print(
            f"{status}: "
            f"{count} sources "
            f"({percentage:.2f}%)"
        )


    # --------------------------------------------------------
    # CLASS-WISE SOURCE PERFORMANCE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "CLASS-WISE SOURCE-LEVEL PERFORMANCE"
    )

    print(
        "-" * 70
    )


    class_source_results = []

    for class_name in sorted(

        source_results_df[
            "true_class"
        ].unique()

    ):

        class_sources = (
            source_results_df[
                source_results_df[
                    "true_class"
                ]
                == class_name
            ]
        )

        total_class_sources = (
            len(class_sources)
        )

        correct_class_sources = (
            class_sources[
                "majority_vote_correct"
            ].sum()
        )

        class_source_accuracy = (
            correct_class_sources
            / total_class_sources
            * 100
        )


        class_source_results.append({

            "class":
                class_name,

            "total_sources":
                total_class_sources,

            "correct_sources":
                correct_class_sources,

            "incorrect_sources":
                total_class_sources
                - correct_class_sources,

            "source_accuracy_percent":
                class_source_accuracy

        })


    class_source_df = pd.DataFrame(
        class_source_results
    )

    print(

        class_source_df.to_string(

            index=False

        )

    )


    # --------------------------------------------------------
    # DIFFICULT SOURCES
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "MOST DIFFICULT SOURCES"
    )

    print(
        "-" * 70
    )


    difficult_sources = (
        source_results_df
        .sort_values(

            by=[

                "source_accuracy_percent",
                "total_variants"

            ],

            ascending=[

                True,
                False

            ]

        )
    )


    print(

        difficult_sources[
            [

                "source_name",
                "true_class",
                "total_variants",
                "correct_variants",
                "incorrect_variants",
                "source_accuracy_percent",
                "majority_prediction",
                "average_confidence_percent",
                "source_status"

            ]

        ]
        .head(20)
        .to_string(

            index=False

        )

    )


    # --------------------------------------------------------
    # COMPLETELY INCORRECT SOURCES
    # --------------------------------------------------------

    completely_incorrect = (

        source_results_df[

            source_results_df[
                "source_status"
            ]

            == "ALL_INCORRECT"

        ]

    )


    print(
        "\nCompletely incorrect sources: "
        f"{len(completely_incorrect)}"
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    source_results_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_source_level_results.csv"

    )


    class_source_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_class_source_performance.csv"

    )


    difficult_sources_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_difficult_sources.csv"

    )


    source_results_df.to_csv(

        source_results_path,

        index=False

    )


    class_source_df.to_csv(

        class_source_path,

        index=False

    )


    difficult_sources.to_csv(

        difficult_sources_path,

        index=False

    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCE-LEVEL ANALYSIS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImage-level accuracy: "
        f"{image_accuracy:.2f}%"
    )

    print(
        f"Source-level majority-vote accuracy: "
        f"{source_accuracy:.2f}%"
    )

    print(
        f"Total unique validation sources: "
        f"{total_sources}"
    )

    print(
        f"Completely incorrect sources: "
        f"{len(completely_incorrect)}"
    )


    print(
        "\nSaved files:"
    )

    print(
        f"\n1. Source-Level Results:\n"
        f"{source_results_path}"
    )

    print(
        f"\n2. Class Source Performance:\n"
        f"{class_source_path}"
    )

    print(
        f"\n3. Difficult Sources:\n"
        f"{difficult_sources_path}"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCE-LEVEL ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()