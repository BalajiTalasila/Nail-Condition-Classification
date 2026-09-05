import os
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

COMPREHENSIVE_PATH = os.path.join(
    METRICS_DIR,
    "comprehensive_model_comparison.csv"
)

CLASSWISE_PATH = os.path.join(
    METRICS_DIR,
    "classwise_model_comparison.csv"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "final_experimental_results_summary.txt"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("GENERATING FINAL EXPERIMENTAL RESULTS SUMMARY")
print("=" * 70)

comparison_df = pd.read_csv(
    COMPREHENSIVE_PATH
)

classwise_df = pd.read_csv(
    CLASSWISE_PATH
)


# ============================================================
# IDENTIFY BEST MODELS
# ============================================================

best_accuracy = comparison_df.loc[
    comparison_df[
        "Accuracy_Percentage"
    ].idxmax()
]

best_f1 = comparison_df.loc[
    comparison_df[
        "F1_Macro"
    ].idxmax()
]

fastest_model = comparison_df.loc[
    comparison_df[
        "Inference_Time_ms"
    ].idxmin()
]

smallest_model = comparison_df.loc[
    comparison_df[
        "Parameters_Millions"
    ].idxmin()
]


# ============================================================
# PROPOSED MODEL INFORMATION
# ============================================================

efficientnet = comparison_df[
    comparison_df["Model"] ==
    "EfficientNet-B0"
].iloc[0]

proposed = comparison_df[
    comparison_df["Model"] ==
    "Proposed Attention-EfficientNet-B0"
].iloc[0]


accuracy_difference = (
    efficientnet["Accuracy_Percentage"]
    -
    proposed["Accuracy_Percentage"]
)


f1_difference = (
    efficientnet["F1_Macro"]
    -
    proposed["F1_Macro"]
)


inference_difference = (
    proposed["Inference_Time_ms"]
    -
    efficientnet["Inference_Time_ms"]
)


parameter_difference = (
    proposed["Parameters_Millions"]
    -
    efficientnet["Parameters_Millions"]
)


# ============================================================
# CALIBRATION ANALYSIS
# ============================================================

calibrated_models = comparison_df.dropna(
    subset=["ECE"]
)

best_calibrated = calibrated_models.loc[
    calibrated_models[
        "ECE"
    ].idxmin()
]


# ============================================================
# CLASS-WISE RESULTS
# ============================================================

proposed_better = classwise_df[
    classwise_df[
        "Better_Model_F1"
    ] == "Proposed Model"
]

efficientnet_better = classwise_df[
    classwise_df[
        "Better_Model_F1"
    ] == "EfficientNet-B0"
]

equal_performance = classwise_df[
    classwise_df[
        "Better_Model_F1"
    ] == "Equal"
]


# ============================================================
# BUILD REPORT
# ============================================================

lines = []

lines.append("=" * 70)
lines.append("FINAL EXPERIMENTAL RESULTS SUMMARY")
lines.append("=" * 70)

lines.append("")
lines.append("1. TEST DATASET")
lines.append("-" * 70)
lines.append("Total Test Images: 576")
lines.append("Number of Classes: 6")
lines.append(
    "Classes: Acral_Lentiginous_Melanoma, "
    "Healthy_Nail, Onychogryphosis, "
    "blue_finger, clubbing, pitting"
)


lines.append("")
lines.append("2. OVERALL MODEL PERFORMANCE")
lines.append("-" * 70)

for _, row in comparison_df.iterrows():

    lines.append("")
    lines.append(
        f"Model: {row['Model']}"
    )

    lines.append(
        f"Accuracy: "
        f"{row['Accuracy_Percentage']:.4f}%"
    )

    lines.append(
        f"Macro Precision: "
        f"{row['Precision_Macro']:.4f}%"
    )

    lines.append(
        f"Macro Recall: "
        f"{row['Recall_Macro']:.4f}%"
    )

    lines.append(
        f"Macro F1-Score: "
        f"{row['F1_Macro']:.4f}%"
    )

    lines.append(
        f"MCC: "
        f"{row['MCC']:.4f}%"
    )

    lines.append(
        f"Inference Time: "
        f"{row['Inference_Time_ms']:.4f} ms/image"
    )

    lines.append(
        f"Parameters: "
        f"{row['Parameters_Millions']:.4f} million"
    )


lines.append("")
lines.append("3. BEST OVERALL MODELS")
lines.append("-" * 70)

lines.append(
    f"Highest Accuracy: "
    f"{best_accuracy['Model']} "
    f"({best_accuracy['Accuracy_Percentage']:.4f}%)"
)

lines.append(
    f"Highest Macro F1-Score: "
    f"{best_f1['Model']} "
    f"({best_f1['F1_Macro']:.4f}%)"
)

lines.append(
    f"Fastest Inference: "
    f"{fastest_model['Model']} "
    f"({fastest_model['Inference_Time_ms']:.4f} ms/image)"
)

lines.append(
    f"Smallest Model: "
    f"{smallest_model['Model']} "
    f"({smallest_model['Parameters_Millions']:.4f} million parameters)"
)


lines.append("")
lines.append("4. EFFICIENTNET-B0 VS PROPOSED MODEL")
lines.append("-" * 70)

lines.append(
    f"EfficientNet-B0 Accuracy: "
    f"{efficientnet['Accuracy_Percentage']:.4f}%"
)

lines.append(
    f"Proposed Model Accuracy: "
    f"{proposed['Accuracy_Percentage']:.4f}%"
)

lines.append(
    f"Accuracy Difference: "
    f"{accuracy_difference:.4f} percentage points"
)

lines.append("")

lines.append(
    f"EfficientNet-B0 Macro F1: "
    f"{efficientnet['F1_Macro']:.4f}%"
)

lines.append(
    f"Proposed Model Macro F1: "
    f"{proposed['F1_Macro']:.4f}%"
)

lines.append(
    f"Macro F1 Difference: "
    f"{f1_difference:.4f} percentage points"
)

lines.append("")

lines.append(
    f"EfficientNet-B0 Inference Time: "
    f"{efficientnet['Inference_Time_ms']:.4f} ms/image"
)

lines.append(
    f"Proposed Model Inference Time: "
    f"{proposed['Inference_Time_ms']:.4f} ms/image"
)

lines.append(
    f"Inference Time Difference: "
    f"{inference_difference:.4f} ms/image"
)

lines.append("")

lines.append(
    f"EfficientNet-B0 Parameters: "
    f"{efficientnet['Parameters_Millions']:.4f} million"
)

lines.append(
    f"Proposed Model Parameters: "
    f"{proposed['Parameters_Millions']:.4f} million"
)

lines.append(
    f"Additional Parameters in Proposed Model: "
    f"{parameter_difference:.4f} million"
)


lines.append("")
lines.append("5. STATISTICAL MODEL COMPARISON")
lines.append("-" * 70)

lines.append("McNemar's Test P-Value: 0.687500")
lines.append("Significance Level: 0.05")
lines.append("Result: Not Statistically Significant")

lines.append(
    "Interpretation: The difference between "
    "EfficientNet-B0 and the proposed model "
    "is not statistically significant."
)


lines.append("")
lines.append("6. PREDICTION-LEVEL COMPARISON")
lines.append("-" * 70)

lines.append("Total Test Images: 576")
lines.append("Both Models Correct: 563")
lines.append("Both Models Incorrect: 7")
lines.append("Only EfficientNet-B0 Correct: 4")
lines.append("Only Proposed Model Correct: 2")


lines.append("")
lines.append("7. CLASS-WISE PERFORMANCE")
lines.append("-" * 70)

lines.append(
    f"Classes where EfficientNet-B0 performed better: "
    f"{len(efficientnet_better)}"
)

lines.append(
    f"Classes where Proposed Model performed better: "
    f"{len(proposed_better)}"
)

lines.append(
    f"Classes with equal performance: "
    f"{len(equal_performance)}"
)

if len(proposed_better) > 0:

    lines.append("")

    lines.append(
        "Classes where the Proposed Model achieved "
        "higher F1-score:"
    )

    for _, row in proposed_better.iterrows():

        lines.append(
            f" - {row['Class']}"
        )


lines.append("")
lines.append("8. MODEL CALIBRATION")
lines.append("-" * 70)

for _, row in calibrated_models.iterrows():

    lines.append(
        f"{row['Model']}: "
        f"ECE = {row['ECE']:.6f}, "
        f"Brier Score = {row['Brier_Score']:.6f}"
    )

lines.append("")

lines.append(
    f"Best ECE Calibration: "
    f"{best_calibrated['Model']} "
    f"(ECE = {best_calibrated['ECE']:.6f})"
)

lines.append(
    "Lower ECE indicates better alignment between "
    "model confidence and observed accuracy."
)


lines.append("")
lines.append("9. PREDICTION CONFIDENCE")
lines.append("-" * 70)

lines.append(
    "EfficientNet-B0 Average Confidence: 92.23%"
)

lines.append(
    "EfficientNet-B0 Incorrect Prediction Confidence: 59.19%"
)

lines.append("")

lines.append(
    "Proposed Model Average Confidence: 98.92%"
)

lines.append(
    "Proposed Model Incorrect Prediction Confidence: 78.73%"
)


lines.append("")
lines.append("10. HIGH-CONFIDENCE ERROR ANALYSIS")
lines.append("-" * 70)

lines.append(
    "EfficientNet-B0 Errors >= 90% Confidence: 0"
)

lines.append(
    "Proposed Model Errors >= 90% Confidence: 5"
)

lines.append(
    "The proposed model produced a higher number "
    "of highly confident incorrect predictions."
)


lines.append("")
lines.append("11. FINAL SCIENTIFIC INTERPRETATION")
lines.append("-" * 70)

lines.append(
    "EfficientNet-B0 achieved the highest overall "
    "classification accuracy, Macro F1-score, MCC, "
    "and fastest inference time among the evaluated models."
)

lines.append("")

lines.append(
    "The proposed Attention-EfficientNet-B0 achieved "
    "comparable performance, with an accuracy difference "
    f"of only {accuracy_difference:.4f} percentage points "
    "relative to EfficientNet-B0."
)

lines.append("")

lines.append(
    "McNemar's statistical test indicated that the "
    "performance difference between EfficientNet-B0 "
    "and the proposed model was not statistically significant."
)

lines.append("")

lines.append(
    "The proposed model demonstrated substantially "
    "better Expected Calibration Error than EfficientNet-B0, "
    "suggesting improved overall confidence calibration."
)

lines.append("")

lines.append(
    "However, the proposed model also generated more "
    "high-confidence misclassifications, indicating that "
    "improved aggregate calibration does not eliminate "
    "individual overconfident errors."
)

lines.append("")

lines.append(
    "Overall, EfficientNet-B0 provides the strongest "
    "combination of classification performance and "
    "computational efficiency in the present experiment, "
    "while the proposed model demonstrates comparable "
    "classification performance and improved ECE calibration."
)

lines.append("")
lines.append("=" * 70)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(lines)
    )


print(
    "\n✓ Final experimental results summary created"
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("SUMMARY GENERATION COMPLETED SUCCESSFULLY")
print("=" * 70)

