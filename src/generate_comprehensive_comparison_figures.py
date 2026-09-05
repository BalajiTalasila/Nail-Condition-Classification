import os
import pandas as pd
import matplotlib.pyplot as plt
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

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

INPUT_PATH = os.path.join(
    METRICS_DIR,
    "comprehensive_model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GENERATING COMPREHENSIVE MODEL COMPARISON FIGURE")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nModels loaded:")

for model in df["Model"]:
    print(f" - {model}")


# ============================================================
# SHORT MODEL NAMES
# ============================================================

short_names = {

    "EfficientNet-B0":
        "EfficientNet-B0",

    "ConvNeXtV2-Tiny":
        "ConvNeXtV2-Tiny",

    "Proposed Attention-EfficientNet-B0":
        "Proposed\nAttention-EfficientNet",

    "DenseNet121":
        "DenseNet121"

}

df["Short_Name"] = df["Model"].map(
    short_names
)


# ============================================================
# COMMON SETTINGS
# ============================================================

plt.rcParams.update({

    "font.size": 11

})

models = df["Short_Name"]
x_positions = np.arange(
    len(df)
)


# ============================================================
# FIGURE 1
# PERFORMANCE METRICS
# ============================================================

plt.figure(
    figsize=(12, 7)
)

width = 0.25

plt.bar(
    x_positions - width,
    df["Accuracy_Percentage"],
    width,
    label="Accuracy"
)

plt.bar(
    x_positions,
    df["F1_Macro"],
    width,
    label="Macro F1"
)

plt.bar(
    x_positions + width,
    df["MCC"],
    width,
    label="MCC"
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Performance Comparison of Deep Learning Models"
)

plt.xticks(
    x_positions,
    models
)

plt.ylim(
    90,
    100
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

performance_path = os.path.join(
    FIGURES_DIR,
    "comprehensive_performance_comparison.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\n✓ Saved comprehensive_performance_comparison.png"
)


# ============================================================
# FIGURE 2
# EFFICIENCY COMPARISON
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.scatter(
    df["Parameters_Millions"],
    df["Accuracy_Percentage"],
    s=150
)

for _, row in df.iterrows():

    plt.annotate(
        row["Short_Name"].replace(
            "\n",
            " "
        ),
        (
            row["Parameters_Millions"],
            row["Accuracy_Percentage"]
        ),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.xlabel(
    "Model Parameters (Millions)"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Model Complexity vs Classification Accuracy"
)

plt.grid(
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

complexity_path = os.path.join(
    FIGURES_DIR,
    "comprehensive_complexity_accuracy.png"
)

plt.savefig(
    complexity_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Saved comprehensive_complexity_accuracy.png"
)


# ============================================================
# FIGURE 3
# INFERENCE TIME
# ============================================================

plt.figure(
    figsize=(12, 7)
)

bars = plt.bar(
    models,
    df["Inference_Time_ms"]
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Average Inference Time (ms/image)"
)

plt.title(
    "Inference Time Comparison"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

for bar, value in zip(
    bars,
    df["Inference_Time_ms"]
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,

        bar.get_height(),

        f"{value:.2f}",

        ha="center",

        va="bottom"
    )

plt.tight_layout()

inference_path = os.path.join(
    FIGURES_DIR,
    "comprehensive_inference_time.png"
)

plt.savefig(
    inference_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Saved comprehensive_inference_time.png"
)


# ============================================================
# FIGURE 4
# MODEL SIZE AND PARAMETERS
# ============================================================

plt.figure(
    figsize=(12, 7)
)

width = 0.35

plt.bar(
    x_positions - width / 2,
    df["Parameters_Millions"],
    width,
    label="Parameters (Millions)"
)

plt.bar(
    x_positions + width / 2,
    df["Model_Size_MB"],
    width,
    label="Model Size (MB)"
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "Value"
)

plt.title(
    "Model Complexity Comparison"
)

plt.xticks(
    x_positions,
    models
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

size_path = os.path.join(
    FIGURES_DIR,
    "comprehensive_model_complexity.png"
)

plt.savefig(
    size_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Saved comprehensive_model_complexity.png"
)


# ============================================================
# FIGURE 5
# CALIBRATION COMPARISON
# ============================================================

calibration_df = df.dropna(
    subset=["ECE"]
).copy()


if len(calibration_df) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        calibration_df["Short_Name"],
        calibration_df["ECE"]
    )

    plt.xlabel(
        "Models"
    )

    plt.ylabel(
        "Expected Calibration Error"
    )

    plt.title(
        "Model Confidence Calibration Comparison"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    for index, value in enumerate(
        calibration_df["ECE"]
    ):

        plt.text(
            index,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    calibration_path = os.path.join(
        FIGURES_DIR,
        "comprehensive_calibration_comparison.png"
    )

    plt.savefig(
        calibration_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✓ Saved comprehensive_calibration_comparison.png"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFigures saved in:\n{FIGURES_DIR}"
)
