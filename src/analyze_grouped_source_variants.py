# ============================================================
# SOURCE VARIANT ANALYSIS
# EFFICIENTNET-B0 - LEAKAGE-SAFE GROUPED DATASET
# ============================================================

from pathlib import Path
from collections import Counter

import pandas as pd


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

OUTPUT_PATH = (
    METRICS_DIR
    / "efficientnet_b0_grouped_source_variant_analysis.csv"
)


# ============================================================
# GET SOURCE NAME
# ============================================================

def get_source_name(image_name):

    filename = Path(image_name).stem

    if ".rf." in filename:

        return filename.split(".rf.")[0]

    return filename


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "SOURCE VARIANT PREDICTION ANALYSIS"
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

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        f"Total predictions: {len(df)}"
    )


    # --------------------------------------------------------
    # IDENTIFY SOURCES
    # --------------------------------------------------------

    df["source_name"] = df[
        "image_name"
    ].apply(
        get_source_name
    )


    print(
        f"Unique sources: "
        f"{df['source_name'].nunique()}"
    )


    # --------------------------------------------------------
    # SOURCE SUMMARY
    # --------------------------------------------------------

    source_rows = []

    for source_name, group in df.groupby(
        "source_name"
    ):

        true_classes = group[
            "true_class"
        ].unique()

        if len(true_classes) != 1:

            print(
                f"\nWARNING: Inconsistent label "
                f"for source: {source_name}"
            )

            continue


        true_class = true_classes[0]

        predictions = group[
            "predicted_class"
        ].tolist()

        prediction_counts = Counter(
            predictions
        )

        majority_prediction = (
            prediction_counts.most_common(1)[0][0]
        )

        correct_count = int(
            group[
                "correct_prediction"
            ].sum()
        )

        total_count = len(group)

        incorrect_count = (
            total_count - correct_count
        )

        source_accuracy = (
            correct_count
            / total_count
            * 100
        )

        majority_vote_correct = (
            majority_prediction
            == true_class
        )


        confidence_mean = group[
            "confidence_percent"
        ].mean()

        confidence_min = group[
            "confidence_percent"
        ].min()

        confidence_max = group[
            "confidence_percent"
        ].max()


        unique_predictions = (
            group[
                "predicted_class"
            ].nunique()
        )


        prediction_distribution = (
            group[
                "predicted_class"
            ]
            .value_counts()
            .to_dict()
        )


        source_rows.append({

            "source_name":
                source_name,

            "true_class":
                true_class,

            "total_variants":
                total_count,

            "correct_variants":
                correct_count,

            "incorrect_variants":
                incorrect_count,

            "source_accuracy_percent":
                source_accuracy,

            "majority_prediction":
                majority_prediction,

            "majority_vote_correct":
                majority_vote_correct,

            "unique_predictions":
                unique_predictions,

            "prediction_distribution":
                str(prediction_distribution),

            "average_confidence_percent":
                confidence_mean,

            "minimum_confidence_percent":
                confidence_min,

            "maximum_confidence_percent":
                confidence_max

        })


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        source_rows
    )


    results_df = results_df.sort_values(
        by=[
            "source_accuracy_percent",
            "total_variants"
        ],
        ascending=[
            True,
            False
        ]
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "SOURCE VARIANT SUMMARY"
    )

    print(
        "-" * 70
    )


    total_sources = len(
        results_df
    )

    inconsistent_sources = results_df[
        results_df[
            "unique_predictions"
        ] > 1
    ]

    perfectly_consistent = results_df[
        results_df[
            "unique_predictions"
        ] == 1
    ]


    print(
        f"\nTotal sources: "
        f"{total_sources}"
    )

    print(
        f"Sources with prediction "
        f"variation: "
        f"{len(inconsistent_sources)}"
    )

    print(
        f"Sources with identical "
        f"predictions across all variants: "
        f"{len(perfectly_consistent)}"
    )


    # --------------------------------------------------------
    # COMPLETELY INCORRECT SOURCES
    # --------------------------------------------------------

    completely_wrong = results_df[

        results_df[
            "source_accuracy_percent"
        ] == 0

    ]


    print(
        "\n" + "-" * 70
    )

    print(
        "COMPLETELY INCORRECT SOURCES"
    )

    print(
        "-" * 70
    )


    if len(completely_wrong) > 0:

        print(
            completely_wrong[
                [

                    "source_name",

                    "true_class",

                    "total_variants",

                    "majority_prediction",

                    "average_confidence_percent",

                    "prediction_distribution"

                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo completely incorrect "
            "sources found."
        )


    # --------------------------------------------------------
    # SOURCES WITH MIXED PREDICTIONS
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "MOST INCONSISTENT SOURCES"
    )

    print(
        "-" * 70
    )


    mixed = results_df[

        results_df[
            "unique_predictions"
        ] > 1

    ].head(
        20
    )


    if len(mixed) > 0:

        print(
            mixed[
                [

                    "source_name",

                    "true_class",

                    "total_variants",

                    "correct_variants",

                    "incorrect_variants",

                    "unique_predictions",

                    "prediction_distribution",

                    "majority_prediction"

                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nAll source variants "
            "received identical predictions."
        )


    # --------------------------------------------------------
    # CLASS-WISE VARIANT STABILITY
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "CLASS-WISE SOURCE VARIANT STABILITY"
    )

    print(
        "-" * 70
    )


    class_summary = (

        results_df

        .groupby(
            "true_class"
        )

        .agg(

            total_sources=(
                "source_name",
                "count"
            ),

            average_source_accuracy=(
                "source_accuracy_percent",
                "mean"
            ),

            average_variants_per_source=(
                "total_variants",
                "mean"
            ),

            sources_with_prediction_variation=(
                "unique_predictions",
                lambda x: (
                    x > 1
                ).sum()
            ),

            majority_vote_correct_sources=(
                "majority_vote_correct",
                "sum"
            )

        )

        .reset_index()

    )


    class_summary[
        "source_stability_percent"
    ] = (

        (
            class_summary[
                "total_sources"
            ]

            -

            class_summary[
                "sources_with_prediction_variation"
            ]

        )

        /

        class_summary[
            "total_sources"
        ]

        * 100

    )


    class_summary[
        "majority_vote_accuracy_percent"
    ] = (

        class_summary[
            "majority_vote_correct_sources"
        ]

        /

        class_summary[
            "total_sources"
        ]

        * 100

    )


    print()

    print(

        class_summary.to_string(

            index=False

        )

    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df.to_csv(

        OUTPUT_PATH,

        index=False

    )


    class_summary_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_source_variant_class_summary.csv"

    )


    class_summary.to_csv(

        class_summary_path,

        index=False

    )


    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCE VARIANT ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        "\nSaved files:"
    )


    print(
        "\n1. Source Variant Analysis:"
    )

    print(
        OUTPUT_PATH
    )


    print(
        "\n2. Class Variant Summary:"
    )

    print(
        class_summary_path
    )


if __name__ == "__main__":

    main()