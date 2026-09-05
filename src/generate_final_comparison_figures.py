import os
import pandas as pd
import matplotlib.pyplot as plt


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

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

INPUT_FILE = os.path.join(
    METRICS_DIR,
    "final_results_summary.csv"
)

print("=" * 70)
print("GENERATING FINAL MODEL COMPARISON FIGURES")
print("=" * 70)

print("\nLoading final results...")

df = pd.read_csv(INPUT_FILE)

print("\nModels loaded:")

for model in df["Model"]:
    print(f" - {model}")


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_bar_chart(
    dataframe,
    value_column,
    ylabel,
    title,
    filename
):

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        dataframe["Model"],
        dataframe[value_column]
    )

    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(
        rotation=15,
        ha="right"
    )

    for bar, value in zip(
        bars,
        dataframe[value_column]
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {filename}"
    )


# ============================================================
# 1. ACCURACY COMPARISON
# ============================================================

save_bar_chart(
    dataframe=df,
    value_column="Accuracy",
    ylabel="Accuracy (%)",
    title="Model Accuracy Comparison",
    filename="final_accuracy_comparison.png"
)


# ============================================================
# 2. MACRO F1 COMPARISON
# ============================================================

save_bar_chart(
    dataframe=df,
    value_column="F1_Macro",
    ylabel="Macro F1-Score (%)",
    title="Macro F1-Score Comparison",
    filename="final_macro_f1_comparison.png"
)


# ============================================================
# 3. MCC COMPARISON
# ============================================================

save_bar_chart(
    dataframe=df,
    value_column="MCC",
    ylabel="Matthews Correlation Coefficient (%)",
    title="MCC Comparison",
    filename="final_mcc_comparison.png"
)


# ============================================================
# 4. INFERENCE TIME COMPARISON
# ============================================================

save_bar_chart(
    dataframe=df,
    value_column="Average_Inference_Time_ms",
    ylabel="Average Inference Time (ms/image)",
    title="Inference Time Comparison",
    filename="final_inference_time_comparison.png"
)


# ============================================================
# 5. COMBINED PERFORMANCE COMPARISON
# ============================================================

performance_columns = [
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro"
]

performance_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "Macro F1"
]

x_positions = range(
    len(performance_columns)
)

plt.figure(figsize=(11, 7))

for _, row in df.iterrows():

    values = [
        row[column]
        for column in performance_columns
    ]

    plt.plot(
        x_positions,
        values,
        marker="o",
        linewidth=2,
        label=row["Model"]
    )

plt.xticks(
    list(x_positions),
    performance_labels
)

plt.xlabel("Performance Metric")
plt.ylabel("Score (%)")

plt.title(
    "Overall Performance Comparison"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

output_path = os.path.join(
    FIGURES_DIR,
    "final_combined_performance_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Saved: final_combined_performance_comparison.png"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFigures saved in:\n{FIGURES_DIR}"
)

print("\nGenerated figures:")
print("1. final_accuracy_comparison.png")
print("2. final_macro_f1_comparison.png")
print("3. final_mcc_comparison.png")
print("4. final_inference_time_comparison.png")
print("5. final_combined_performance_comparison.png")
