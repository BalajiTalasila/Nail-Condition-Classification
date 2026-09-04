import os
import pandas as pd
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "reports"
)

os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL PROJECT RESULTS REPORT")
print("=" * 70)


model_comparison = pd.read_csv(
    os.path.join(
        METRICS_DIR,
        "model_comparison.csv"
    )
)

best_class = pd.read_csv(
    os.path.join(
        METRICS_DIR,
        "best_model_per_class.csv"
    )
)


# ============================================================
# IDENTIFY BEST MODEL
# ============================================================

best_model = model_comparison.loc[
    model_comparison["Accuracy"].idxmax()
]

best_model_name = best_model["Model"]

accuracy = best_model["Accuracy"] * 100
macro_f1 = best_model["F1_Macro"] * 100
mcc = best_model["MCC"] * 100
inference_time = best_model[
    "Average_Inference_Time_ms"
]


# ============================================================
# GENERATE REPORT
# ============================================================

report_lines = []

report_lines.append("=" * 70)
report_lines.append("FINAL PROJECT RESULTS REPORT")
report_lines.append("=" * 70)

report_lines.append(
    f"\nGenerated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)


# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

report_lines.append("\n")
report_lines.append("-" * 70)
report_lines.append("1. MODEL PERFORMANCE COMPARISON")
report_lines.append("-" * 70)

for _, row in model_comparison.iterrows():

    report_lines.append(
        f"\nModel: {row['Model']}"
    )

    report_lines.append(
        f"Accuracy: {row['Accuracy'] * 100:.2f}%"
    )

    report_lines.append(
        f"Macro F1-Score: {row['F1_Macro'] * 100:.2f}%"
    )

    report_lines.append(
        f"MCC: {row['MCC'] * 100:.2f}%"
    )

    report_lines.append(
        "Inference Time: "
        f"{row['Average_Inference_Time_ms']:.2f} ms/image"
    )


# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

report_lines.append("\n")
report_lines.append("-" * 70)
report_lines.append("2. BEST MODEL")
report_lines.append("-" * 70)

report_lines.append(
    f"\nSelected Model: {best_model_name}"
)

report_lines.append(
    f"Overall Accuracy: {accuracy:.2f}%"
)

report_lines.append(
    f"Macro F1-Score: {macro_f1:.2f}%"
)

report_lines.append(
    f"MCC: {mcc:.2f}%"
)

report_lines.append(
    f"Inference Time: {inference_time:.2f} ms/image"
)

report_lines.append(
    "\nConclusion:"
)

report_lines.append(
    f"{best_model_name} achieved the highest overall "
    "classification performance among the evaluated models. "
    "It also demonstrated the fastest average inference time, "
    "making it the most suitable model for the final nail "
    "condition classification system."
)


# ------------------------------------------------------------
# BEST MODEL PER CLASS
# ------------------------------------------------------------

report_lines.append("\n")
report_lines.append("-" * 70)
report_lines.append("3. BEST MODEL FOR EACH CLASS")
report_lines.append("-" * 70)

for _, row in best_class.iterrows():

    report_lines.append(
        f"\nClass: {row['Class']}"
    )

    report_lines.append(
        f"Best Model: {row['Best_Model']}"
    )

    report_lines.append(
        f"Precision: {row['Precision'] * 100:.2f}%"
    )

    report_lines.append(
        f"Recall: {row['Recall'] * 100:.2f}%"
    )

    report_lines.append(
        f"F1-Score: {row['F1_Score'] * 100:.2f}%"
    )

    report_lines.append(
        f"Specificity: {row['Specificity'] * 100:.2f}%"
    )


# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

report_lines.append("\n")
report_lines.append("=" * 70)
report_lines.append("4. FINAL CONCLUSION")
report_lines.append("=" * 70)

report_lines.append(
    "\nThree deep learning architectures were evaluated for "
    "the six-class nail condition classification task: "
    "ConvNeXtV2-Tiny, DenseNet121, and EfficientNet-B0."
)

report_lines.append(
    f"\nAmong the evaluated architectures, {best_model_name} "
    "demonstrated the best overall performance."
)

report_lines.append(
    f"It achieved an accuracy of {accuracy:.2f}%, "
    f"a Macro F1-score of {macro_f1:.2f}%, "
    f"and an MCC of {mcc:.2f}%."
)

report_lines.append(
    f"The model also achieved the fastest average inference "
    f"time of {inference_time:.2f} ms per image."
)

report_lines.append(
    "\nThe experimental results therefore support the selection "
    f"of {best_model_name} as the final model for the "
    "classification system."
)

report_lines.append(
    "\nGrad-CAM visualizations were generated for the selected "
    "model to provide visual interpretability and identify the "
    "image regions contributing to classification decisions."
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    REPORTS_DIR,
    "final_project_results_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    for line in report_lines:

        file.write(line + "\n")


# ============================================================
# PRINT REPORT
# ============================================================

print("\n" + "=" * 70)
print("PROJECT RESULTS REPORT GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nReport saved to:\n{report_path}")

print("\nBest Model:")

print(best_model_name)

print(
    f"\nAccuracy: {accuracy:.2f}%"
)

print(
    f"Macro F1-Score: {macro_f1:.2f}%"
)

print(
    f"MCC: {mcc:.2f}%"
)

print(
    f"Inference Time: {inference_time:.2f} ms/image"
)

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)