import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

SOURCE_FILE = os.path.join(
    METRICS_DIR,
    "final_model_comparison.csv"
)

OUTPUT_FILE = os.path.join(
    METRICS_DIR,
    "final_results_summary.csv"
)

print("=" * 70)
print("UPDATING FINAL RESULTS SUMMARY")
print("=" * 70)

df = pd.read_csv(SOURCE_FILE)

required_columns = [
    "Model",
    "Accuracy_Percentage",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "F1_Weighted",
    "MCC",
    "Average_Inference_Time_ms"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

summary = pd.DataFrame({

    "Model": df["Model"],

    "Accuracy": (
        df["Accuracy_Percentage"]
    ).round(2),

    "Precision_Macro": (
        df["Precision_Macro"] * 100
    ).round(2),

    "Recall_Macro": (
        df["Recall_Macro"] * 100
    ).round(2),

    "F1_Macro": (
        df["F1_Macro"] * 100
    ).round(2),

    "F1_Weighted": (
        df["F1_Weighted"] * 100
    ).round(2),

    "MCC": (
        df["MCC"] * 100
    ).round(2),

    "Average_Inference_Time_ms": (
        df["Average_Inference_Time_ms"]
    ).round(2)

})

summary = summary.sort_values(
    by="Accuracy",
    ascending=False
)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nUpdated Final Results:")
print()
print(
    summary.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY UPDATED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved to:\n{OUTPUT_FILE}")
