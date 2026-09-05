import pandas as pd

from pathlib import Path


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)


# ============================================================
# INPUT AND OUTPUT FILES
# ============================================================

INPUT_FILE = (
    METRICS_DIR
    / "final_model_comparison.csv"
)


OUTPUT_FILE = (
    METRICS_DIR
    / "model_results_summary.txt"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL RESULTS SUMMARY GENERATION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"Input file not found:\n"
            f"{INPUT_FILE}"

        )


    dataframe = pd.read_csv(
        INPUT_FILE
    )


    print(
        "\n✓ Final model comparison data loaded"
    )


    # --------------------------------------------------------
    # RANK MODELS
    # --------------------------------------------------------

    accuracy_ranking = (
        dataframe
        .sort_values(
            by="Accuracy",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    speed_ranking = (
        dataframe
        .sort_values(
            by="Average_Inference_Time_ms",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )


    complexity_ranking = (
        dataframe
        .sort_values(
            by="Total_Parameters_Millions",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )


    efficiency_ranking = (
        dataframe
        .sort_values(
            by="Parameter_Efficiency",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # IDENTIFY BEST MODELS
    # --------------------------------------------------------

    best_accuracy = accuracy_ranking.iloc[0]

    fastest_model = speed_ranking.iloc[0]

    smallest_model = complexity_ranking.iloc[0]

    most_efficient_model = efficiency_ranking.iloc[0]


    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    report_lines = []


    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "MODEL PERFORMANCE AND COMPLEXITY ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )


    # --------------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------------

    report_lines.append(
        "\nOVERALL MODEL PERFORMANCE"
    )

    report_lines.append(
        "-" * 70
    )


    for index, row in accuracy_ranking.iterrows():

        report_lines.append(

            f"{index + 1}. "
            f"{row['Model']} | "
            f"Accuracy: "
            f"{row['Accuracy_Percentage']:.4f}% | "
            f"Macro F1: "
            f"{row['Macro_F1_Percentage']:.4f}% | "
            f"MCC: "
            f"{row['MCC']:.4f}"

        )


    # --------------------------------------------------------
    # ACCURACY RANKING
    # --------------------------------------------------------

    report_lines.append(
        "\nACCURACY RANKING"
    )

    report_lines.append(
        "-" * 70
    )


    for index, row in accuracy_ranking.iterrows():

        report_lines.append(

            f"{index + 1}. "
            f"{row['Model']} - "
            f"{row['Accuracy_Percentage']:.4f}%"

        )


    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    report_lines.append(
        "\nINFERENCE SPEED RANKING"
    )

    report_lines.append(
        "-" * 70
    )


    for index, row in speed_ranking.iterrows():

        report_lines.append(

            f"{index + 1}. "
            f"{row['Model']} - "
            f"{row['Average_Inference_Time_ms']:.4f} "
            f"ms/image"

        )


    # --------------------------------------------------------
    # MODEL COMPLEXITY
    # --------------------------------------------------------

    report_lines.append(
        "\nMODEL COMPLEXITY RANKING"
    )

    report_lines.append(
        "-" * 70
    )


    for index, row in complexity_ranking.iterrows():

        report_lines.append(

            f"{index + 1}. "
            f"{row['Model']} - "
            f"{row['Total_Parameters_Millions']:.4f} "
            f"million parameters"

        )


    # --------------------------------------------------------
    # PARAMETER EFFICIENCY
    # --------------------------------------------------------

    report_lines.append(
        "\nPARAMETER EFFICIENCY RANKING"
    )

    report_lines.append(
        "-" * 70
    )


    for index, row in efficiency_ranking.iterrows():

        report_lines.append(

            f"{index + 1}. "
            f"{row['Model']} - "
            f"{row['Parameter_Efficiency']:.4f} "
            f"accuracy percentage per million parameters"

        )


    # --------------------------------------------------------
    # MODEL HIGHLIGHTS
    # --------------------------------------------------------

    report_lines.append(
        "\nMODEL HIGHLIGHTS"
    )

    report_lines.append(
        "-" * 70
    )


    report_lines.append(

        f"Best Accuracy: "
        f"{best_accuracy['Model']} "
        f"({best_accuracy['Accuracy_Percentage']:.4f}%)"

    )


    report_lines.append(

        f"Fastest Model: "
        f"{fastest_model['Model']} "
        f"({fastest_model['Average_Inference_Time_ms']:.4f} ms/image)"

    )


    report_lines.append(

        f"Smallest Model: "
        f"{smallest_model['Model']} "
        f"({smallest_model['Total_Parameters_Millions']:.4f} million parameters)"

    )


    report_lines.append(

        f"Highest Parameter Efficiency: "
        f"{most_efficient_model['Model']} "
        f"({most_efficient_model['Parameter_Efficiency']:.4f})"

    )


    # --------------------------------------------------------
    # PROPOSED MODEL ANALYSIS
    # --------------------------------------------------------

    proposed_model = dataframe[
        dataframe[
            "Model"
        ]
        ==
        "Proposed Attention-EfficientNet-B0"
    ]


    if not proposed_model.empty:

        proposed_model = proposed_model.iloc[0]


        report_lines.append(
            "\nPROPOSED MODEL PERFORMANCE"
        )

        report_lines.append(
            "-" * 70
        )


        report_lines.append(

            f"Accuracy: "
            f"{proposed_model['Accuracy_Percentage']:.4f}%"

        )


        report_lines.append(

            f"Macro F1-Score: "
            f"{proposed_model['Macro_F1_Percentage']:.4f}%"

        )


        report_lines.append(

            f"MCC: "
            f"{proposed_model['MCC']:.4f}"

        )


        report_lines.append(

            f"Inference Time: "
            f"{proposed_model['Average_Inference_Time_ms']:.4f} "
            f"ms/image"

        )


        report_lines.append(

            f"Parameters: "
            f"{proposed_model['Total_Parameters_Millions']:.4f} "
            f"million"

        )


        report_lines.append(

            f"Model Size: "
            f"{proposed_model['Model_Size_MB']:.4f} MB"

        )


        report_lines.append(

            f"Parameter Efficiency: "
            f"{proposed_model['Parameter_Efficiency']:.4f}"

        )


    # --------------------------------------------------------
    # COMPARISON WITH BASE EFFICIENTNET
    # --------------------------------------------------------

    base_model = dataframe[
        dataframe[
            "Model"
        ]
        ==
        "EfficientNet-B0"
    ]


    if (
        not proposed_model.empty
        if isinstance(
            proposed_model,
            pd.DataFrame
        )
        else True
    ):

        pass


    if not base_model.empty:

        base_model = base_model.iloc[0]


        proposed_row = dataframe[
            dataframe[
                "Model"
            ]
            ==
            "Proposed Attention-EfficientNet-B0"
        ]


        if not proposed_row.empty:

            proposed_row = proposed_row.iloc[0]


            accuracy_difference = (

                proposed_row[
                    "Accuracy_Percentage"
                ]

                -

                base_model[
                    "Accuracy_Percentage"
                ]

            )


            inference_difference = (

                proposed_row[
                    "Average_Inference_Time_ms"
                ]

                -

                base_model[
                    "Average_Inference_Time_ms"
                ]

            )


            parameter_difference = (

                proposed_row[
                    "Total_Parameters_Millions"
                ]

                -

                base_model[
                    "Total_Parameters_Millions"
                ]

            )


            report_lines.append(
                "\nPROPOSED MODEL VS BASE EFFICIENTNET-B0"
            )

            report_lines.append(
                "-" * 70
            )


            report_lines.append(

                f"Accuracy Difference: "
                f"{accuracy_difference:.4f} percentage points"

            )


            report_lines.append(

                f"Inference Time Difference: "
                f"{inference_difference:.4f} ms/image"

            )


            report_lines.append(

                f"Parameter Difference: "
                f"{parameter_difference:.4f} million parameters"

            )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report_lines.append(
        "\n" + "=" * 70
    )

    report_lines.append(
        "END OF MODEL RESULTS SUMMARY"
    )

    report_lines.append(
        "=" * 70
    )


    report_text = "\n".join(
        report_lines
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )


    # --------------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------------

    print(
        "\n" + report_text
    )


    print(
        f"\n✓ Summary saved to:\n{OUTPUT_FILE}"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "SUMMARY GENERATION COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
