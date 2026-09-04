import os
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")

OUTPUT_FILE = os.path.join(
    METRICS_DIR,
    "final_results_summary.csv"
)


# ============================================================
# LOAD MODEL COMPARISON DATA
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL RESULTS SUMMARY")
print("=" * 70)

comparison_path = os.path.join(
    METRICS_DIR,
    "model_comparison.csv"
)

comparison_df = pd.read_csv(comparison_path)

print("\nModel Comparison Results:\n")

display_columns = [
    "Model",
    "Accuracy",
    "F1_Macro",
    "MCC",
    "Average_Inference_Time_ms"
]

print(comparison_df[display_columns].to_string(index=False))


# ============================================================
# CONVERT METRICS TO PERCENTAGES
# ============================================================

percentage_columns = [
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "F1_Weighted",
    "MCC"
]

summary_df = comparison_df.copy()

for column in percentage_columns:
    summary_df[column] = summary_df[column] * 100

summary_df = summary_df.round(2)


# ============================================================
# DETERMINE BEST MODEL
# ============================================================

best_accuracy_row = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

fastest_model_row = comparison_df.loc[
    comparison_df["Average_Inference_Time_ms"].idxmin()
]

print("\n" + "-" * 70)
print("BEST MODEL ANALYSIS")
print("-" * 70)

print(
    f"\nBest Model by Accuracy: "
    f"{best_accuracy_row['Model']}"
)

print(
    f"Accuracy: "
    f"{best_accuracy_row['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1-Score: "
    f"{best_accuracy_row['F1_Macro'] * 100:.2f}%"
)

print(
    f"MCC: "
    f"{best_accuracy_row['MCC'] * 100:.2f}%"
)

print(
    f"Inference Time: "
    f"{best_accuracy_row['Average_Inference_Time_ms']:.2f} ms/image"
)


print(
    f"\nFastest Model: "
    f"{fastest_model_row['Model']}"
)

print(
    f"Inference Time: "
    f"{fastest_model_row['Average_Inference_Time_ms']:.2f} ms/image"
)


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY SAVED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved to:\n{OUTPUT_FILE}")


# ============================================================
# GENERATE PER-CLASS COMPARISON
# ============================================================

model_files = {
    "ConvNeXtV2-Tiny":
        "convnextv2_tiny_class_metrics.csv",

    "DenseNet121":
        "densenet121_class_metrics.csv",

    "EfficientNet-B0":
        "efficientnet_b0_class_metrics.csv"
}


class_results = []

print("\nLoading per-class metrics...")

for model_name, filename in model_files.items():

    file_path = os.path.join(
        METRICS_DIR,
        filename
    )

    df = pd.read_csv(file_path)

    df["Model"] = model_name

    class_results.append(df)

    print(f"✓ Loaded {model_name}")


combined_class_df = pd.concat(
    class_results,
    ignore_index=True
)


per_class_output = os.path.join(
    METRICS_DIR,
    "per_class_model_comparison.csv"
)

combined_class_df.to_csv(
    per_class_output,
    index=False
)


print("\n✓ Per-class model comparison saved:")

print(per_class_output)


# ============================================================
# BEST MODEL PER CLASS
# ============================================================

best_class_results = []

for class_name in combined_class_df["Class"].unique():

    class_df = combined_class_df[
        combined_class_df["Class"] == class_name
    ]

    best_row = class_df.loc[
        class_df["F1_Score"].idxmax()
    ]

    best_class_results.append({
        "Class": class_name,
        "Best_Model": best_row["Model"],
        "Precision": best_row["Precision"],
        "Recall": best_row["Recall"],
        "F1_Score": best_row["F1_Score"],
        "Specificity": best_row["Specificity"]
    })


best_class_df = pd.DataFrame(
    best_class_results
)

best_class_output = os.path.join(
    METRICS_DIR,
    "best_model_per_class.csv"
)

best_class_df.to_csv(
    best_class_output,
    index=False
)


print("\n✓ Best model per class saved:")

print(best_class_output)


# ============================================================
# FINAL COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ALL FINAL RESULTS GENERATED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print("1. final_results_summary.csv")
print("2. per_class_model_comparison.csv")
print("3. best_model_per_class.csv")

print("\n")