import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)

print("=" * 70)
print("CLASS-WISE MODEL PERFORMANCE COMPARISON")
print("=" * 70)

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

efficientnet_path = os.path.join(
    METRICS_DIR,
    "efficientnet_b0_test_predictions.csv"
)

proposed_path = os.path.join(
    METRICS_DIR,
    "proposed_model_test_predictions.csv"
)

output_csv_path = os.path.join(
    METRICS_DIR,
    "classwise_model_comparison.csv"
)

output_summary_path = os.path.join(
    METRICS_DIR,
    "classwise_model_comparison_summary.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading prediction files...")

efficientnet_df = pd.read_csv(
    efficientnet_path
)

proposed_df = pd.read_csv(
    proposed_path
)

print(
    f"EfficientNet-B0 samples: "
    f"{len(efficientnet_df)}"
)

print(
    f"Proposed model samples: "
    f"{len(proposed_df)}"
)


# ============================================================
# VALIDATE ALIGNMENT
# ============================================================

if not np.array_equal(
    efficientnet_df["True_Label"].values,
    proposed_df["True_Label"].values
):
    raise ValueError(
        "True labels are not aligned between models."
    )

print("✓ Prediction files are aligned")


# ============================================================
# CLASS LIST
# ============================================================

class_names = sorted(
    efficientnet_df["True_Label"].unique()
)

true_labels = (
    efficientnet_df["True_Label"]
)

efficientnet_predictions = (
    efficientnet_df["Predicted_Label"]
)

proposed_predictions = (
    proposed_df["Predicted_Label"]
)


# ============================================================
# CALCULATE CLASS-WISE METRICS
# ============================================================

eff_precision, eff_recall, eff_f1, eff_support = (
    precision_recall_fscore_support(
        true_labels,
        efficientnet_predictions,
        labels=class_names,
        zero_division=0
    )
)

prop_precision, prop_recall, prop_f1, prop_support = (
    precision_recall_fscore_support(
        true_labels,
        proposed_predictions,
        labels=class_names,
        zero_division=0
    )
)


# ============================================================
# CREATE COMPARISON DATAFRAME
# ============================================================

comparison_df = pd.DataFrame({

    "Class": class_names,

    "Support": eff_support,

    "EfficientNet_Precision": eff_precision,
    "Proposed_Precision": prop_precision,

    "Precision_Difference": (
        prop_precision - eff_precision
    ),

    "EfficientNet_Recall": eff_recall,
    "Proposed_Recall": prop_recall,

    "Recall_Difference": (
        prop_recall - eff_recall
    ),

    "EfficientNet_F1": eff_f1,
    "Proposed_F1": prop_f1,

    "F1_Difference": (
        prop_f1 - eff_f1
    )

})


# ============================================================
# DETERMINE BETTER MODEL PER CLASS
# ============================================================

comparison_df[
    "Better_Model_F1"
] = np.where(

    comparison_df["F1_Difference"] > 0,

    "Proposed Model",

    np.where(

        comparison_df["F1_Difference"] < 0,

        "EfficientNet-B0",

        "Equal"

    )

)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE F1-SCORE COMPARISON")
print("=" * 70)

for _, row in comparison_df.iterrows():

    print(
        f"\nClass: {row['Class']}"
    )

    print(
        f"Support: {row['Support']}"
    )

    print(
        f"EfficientNet-B0 F1: "
        f"{row['EfficientNet_F1'] * 100:.2f}%"
    )

    print(
        f"Proposed Model F1: "
        f"{row['Proposed_F1'] * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{row['F1_Difference'] * 100:+.2f}%"
    )

    print(
        f"Better Model: "
        f"{row['Better_Model_F1']}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

proposed_better = len(
    comparison_df[
        comparison_df["F1_Difference"] > 0
    ]
)

efficientnet_better = len(
    comparison_df[
        comparison_df["F1_Difference"] < 0
    ]
)

equal_performance = len(
    comparison_df[
        comparison_df["F1_Difference"] == 0
    ]
)

print(
    f"\nClasses where Proposed Model performs better: "
    f"{proposed_better}"
)

print(
    f"Classes where EfficientNet-B0 performs better: "
    f"{efficientnet_better}"
)

print(
    f"Classes with equal performance: "
    f"{equal_performance}"
)


# ============================================================
# SAVE CSV
# ============================================================

comparison_df.to_csv(
    output_csv_path,
    index=False
)

print(
    f"\n✓ Detailed comparison saved to:\n"
    f"{output_csv_path}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    output_summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CLASS-WISE MODEL PERFORMANCE COMPARISON\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    for _, row in comparison_df.iterrows():

        file.write(
            f"Class: {row['Class']}\n"
        )

        file.write(
            f"Support: {row['Support']}\n"
        )

        file.write(
            f"EfficientNet-B0 Precision: "
            f"{row['EfficientNet_Precision'] * 100:.2f}%\n"
        )

        file.write(
            f"Proposed Model Precision: "
            f"{row['Proposed_Precision'] * 100:.2f}%\n"
        )

        file.write(
            f"EfficientNet-B0 Recall: "
            f"{row['EfficientNet_Recall'] * 100:.2f}%\n"
        )

        file.write(
            f"Proposed Model Recall: "
            f"{row['Proposed_Recall'] * 100:.2f}%\n"
        )

        file.write(
            f"EfficientNet-B0 F1: "
            f"{row['EfficientNet_F1'] * 100:.2f}%\n"
        )

        file.write(
            f"Proposed Model F1: "
            f"{row['Proposed_F1'] * 100:.2f}%\n"
        )

        file.write(
            f"F1 Difference: "
            f"{row['F1_Difference'] * 100:+.2f}%\n"
        )

        file.write(
            f"Better Model: "
            f"{row['Better_Model_F1']}\n"
        )

        file.write(
            "-" * 50 + "\n"
        )

    file.write("\nOVERALL SUMMARY\n")

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        f"Proposed Model Better: "
        f"{proposed_better} classes\n"
    )

    file.write(
        f"EfficientNet-B0 Better: "
        f"{efficientnet_better} classes\n"
    )

    file.write(
        f"Equal Performance: "
        f"{equal_performance} classes\n"
    )


print(
    f"\n✓ Summary saved to:\n"
    f"{output_summary_path}"
)

print("\n" + "=" * 70)
print(
    "CLASS-WISE COMPARISON COMPLETED SUCCESSFULLY"
)
print("=" * 70)
