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
# LOAD MODEL COMPARISON DATA
# ============================================================

INPUT_FILE = os.path.join(
    METRICS_DIR,
    "final_model_comparison.csv"
)

print("=" * 70)
print("MODEL COMPLEXITY VS PERFORMANCE ANALYSIS")
print("=" * 70)

print("\nLoading model comparison data...")

df = pd.read_csv(INPUT_FILE)

print(f"✓ Models loaded: {len(df)}")


# ============================================================
# DISPLAY DATA
# ============================================================

print("\nMODEL DATA")

for _, row in df.iterrows():

    print(
        f"\n{row['Model']}"
    )

    print(
        f"Parameters: "
        f"{row['Total_Parameters_Millions']:.4f} million"
    )

    print(
        f"Accuracy: "
        f"{row['Accuracy_Percentage']:.2f}%"
    )

    print(
        f"Inference Time: "
        f"{row['Average_Inference_Time_ms']:.2f} ms"
    )


# ============================================================
# PARAMETERS VS ACCURACY
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["Total_Parameters_Millions"],
    df["Accuracy_Percentage"],
    s=150
)

for _, row in df.iterrows():

    plt.annotate(
        row["Model"],
        (
            row["Total_Parameters_Millions"],
            row["Accuracy_Percentage"]
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel(
    "Total Parameters (Millions)"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.title(
    "Model Complexity vs Classification Performance"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

output_path = os.path.join(
    FIGURES_DIR,
    "model_complexity_vs_accuracy.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\n✓ Complexity vs accuracy figure saved:"
)

print(
    output_path
)


# ============================================================
# PARAMETERS VS INFERENCE TIME
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["Total_Parameters_Millions"],
    df["Average_Inference_Time_ms"],
    s=150
)

for _, row in df.iterrows():

    plt.annotate(
        row["Model"],
        (
            row["Total_Parameters_Millions"],
            row["Average_Inference_Time_ms"]
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel(
    "Total Parameters (Millions)"
)

plt.ylabel(
    "Average Inference Time (ms/image)"
)

plt.title(
    "Model Complexity vs Inference Time"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

output_path = os.path.join(
    FIGURES_DIR,
    "model_complexity_vs_inference_time.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\n✓ Complexity vs inference time figure saved:"
)

print(
    output_path
)


# ============================================================
# PARAMETER EFFICIENCY VS ACCURACY
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["Parameter_Efficiency"],
    df["Accuracy_Percentage"],
    s=150
)

for _, row in df.iterrows():

    plt.annotate(
        row["Model"],
        (
            row["Parameter_Efficiency"],
            row["Accuracy_Percentage"]
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel(
    "Parameter Efficiency"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.title(
    "Parameter Efficiency vs Classification Accuracy"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

output_path = os.path.join(
    FIGURES_DIR,
    "parameter_efficiency_vs_accuracy.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\n✓ Parameter efficiency figure saved:"
)

print(
    output_path
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPLEXITY ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFigures saved in:\n{FIGURES_DIR}"
)
