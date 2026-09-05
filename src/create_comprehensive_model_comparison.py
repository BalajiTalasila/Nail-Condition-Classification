import os
import pandas as pd
import numpy as np


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

FINAL_COMPARISON_PATH = os.path.join(
    METRICS_DIR,
    "final_model_comparison.csv"
)

CALIBRATION_PATH = os.path.join(
    METRICS_DIR,
    "model_calibration_analysis.txt"
)

OUTPUT_PATH = os.path.join(
    METRICS_DIR,
    "comprehensive_model_comparison.csv"
)


# ============================================================
# LOAD PERFORMANCE AND COMPLEXITY DATA
# ============================================================

print("=" * 70)
print("CREATING COMPREHENSIVE MODEL COMPARISON TABLE")
print("=" * 70)

print("\nLoading model performance data...")

df = pd.read_csv(
    FINAL_COMPARISON_PATH
)

print(
    f"✓ Loaded {len(df)} models"
)


# ============================================================
# INITIALIZE CALIBRATION METRICS
# ============================================================

df["ECE"] = np.nan
df["Brier_Score"] = np.nan


# ============================================================
# ADD CALIBRATION RESULTS
# ============================================================

print("\nAdding calibration metrics...")

calibration_values = {

    "EfficientNet-B0": {
        "ECE": 0.065050,
        "Brier_Score": 0.004695
    },

    "Proposed Attention-EfficientNet-B0": {
        "ECE": 0.008949,
        "Brier_Score": 0.004805
    }

}

for index, row in df.iterrows():

    model_name = row["Model"]

    if model_name in calibration_values:

        df.loc[
            index,
            "ECE"
        ] = calibration_values[
            model_name
        ]["ECE"]

        df.loc[
            index,
            "Brier_Score"
        ] = calibration_values[
            model_name
        ]["Brier_Score"]


# ============================================================
# SELECT AND RENAME COLUMNS
# ============================================================

comparison_df = df[[
    "Model",
    "Accuracy_Percentage",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Macro_F1_Percentage",
    "F1_Weighted",
    "MCC",
    "Average_Inference_Time_ms",
    "Total_Parameters",
    "Total_Parameters_Millions",
    "Model_Size_MB",
    "Parameter_Efficiency",
    "ECE",
    "Brier_Score"
]].copy()


comparison_df.rename(
    columns={

        "Accuracy_Percentage":
            "Accuracy_Percentage",

        "Precision_Macro":
            "Precision_Macro",

        "Recall_Macro":
            "Recall_Macro",

        "F1_Macro":
            "F1_Macro",

        "Macro_F1_Percentage":
            "Macro_F1_Percentage",

        "F1_Weighted":
            "F1_Weighted",

        "Average_Inference_Time_ms":
            "Inference_Time_ms",

        "Total_Parameters":
            "Total_Parameters",

        "Total_Parameters_Millions":
            "Parameters_Millions"

    },
    inplace=True
)


# ============================================================
# CONVERT PERFORMANCE METRICS TO PERCENTAGES
# ============================================================

percentage_columns = [

    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "F1_Weighted",
    "MCC"

]

for column in percentage_columns:

    comparison_df[
        column
    ] = (
        comparison_df[column] * 100
    )


# ============================================================
# ROUND NUMERICAL VALUES
# ============================================================

comparison_df[
    "Accuracy_Percentage"
] = comparison_df[
    "Accuracy_Percentage"
].round(4)


comparison_df[
    "Precision_Macro"
] = comparison_df[
    "Precision_Macro"
].round(4)


comparison_df[
    "Recall_Macro"
] = comparison_df[
    "Recall_Macro"
].round(4)


comparison_df[
    "F1_Macro"
] = comparison_df[
    "F1_Macro"
].round(4)


comparison_df[
    "Macro_F1_Percentage"
] = comparison_df[
    "Macro_F1_Percentage"
].round(4)


comparison_df[
    "F1_Weighted"
] = comparison_df[
    "F1_Weighted"
].round(4)


comparison_df[
    "MCC"
] = comparison_df[
    "MCC"
].round(4)


comparison_df[
    "Inference_Time_ms"
] = comparison_df[
    "Inference_Time_ms"
].round(4)


comparison_df[
    "Parameters_Millions"
] = comparison_df[
    "Parameters_Millions"
].round(4)


comparison_df[
    "Model_Size_MB"
] = comparison_df[
    "Model_Size_MB"
].round(4)


comparison_df[
    "Parameter_Efficiency"
] = comparison_df[
    "Parameter_Efficiency"
].round(4)


comparison_df[
    "ECE"
] = comparison_df[
    "ECE"
].round(6)


comparison_df[
    "Brier_Score"
] = comparison_df[
    "Brier_Score"
].round(6)


# ============================================================
# SORT BY ACCURACY
# ============================================================

comparison_df = comparison_df.sort_values(

    by="Accuracy_Percentage",

    ascending=False

)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_df.to_csv(

    OUTPUT_PATH,

    index=False

)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("COMPREHENSIVE COMPARISON TABLE CREATED SUCCESSFULLY")
print("=" * 70)

print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)
