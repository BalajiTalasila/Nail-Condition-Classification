import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {
    "ConvNeXtV2-Tiny":
        "convnextv2_tiny_test_metrics.csv",

    "DenseNet121":
        "densenet121_test_metrics.csv",

    "EfficientNet-B0":
        "efficientnet_b0_test_metrics.csv"
}


# ============================================================
# LOAD MODEL METRICS
# ============================================================

def load_model_metrics():

    print("\nLoading model metrics...\n")

    comparison_data = []

    for model_name, filename in MODELS.items():

        file_path = os.path.join(
            METRICS_DIR,
            filename
        )

        if not os.path.exists(file_path):

            print(
                f"Warning: Metrics file not found: "
                f"{file_path}"
            )

            continue

        metrics = pd.read_csv(
            file_path
        )

        row = metrics.iloc[0].to_dict()

        row["Model"] = model_name

        comparison_data.append(
            row
        )

        print(
            f"✓ Loaded metrics for "
            f"{model_name}"
        )

    comparison_df = pd.DataFrame(
        comparison_data
    )

    return comparison_df


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

def save_comparison_table(comparison_df):

    output_path = os.path.join(
        METRICS_DIR,
        "model_comparison.csv"
    )

    columns = [
        "Model",
        "Accuracy",
        "Precision_Macro",
        "Recall_Macro",
        "F1_Macro",
        "F1_Weighted",
        "MCC",
        "Average_Inference_Time_ms"
    ]

    comparison_df = comparison_df[
        columns
    ]

    comparison_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n✓ Comparison table saved:\n"
        f"{output_path}"
    )

    return comparison_df


# ============================================================
# ACCURACY COMPARISON
# ============================================================

def generate_accuracy_comparison(df):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        df["Model"],
        df["Accuracy"] * 100
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Test Accuracy (%)"
    )

    plt.title(
        "Model Test Accuracy Comparison"
    )

    plt.ylim(
        90,
        100
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for index, value in enumerate(
        df["Accuracy"] * 100
    ):

        plt.text(
            index,
            value + 0.1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_accuracy_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Accuracy comparison saved:\n"
        f"{output_path}"
    )


# ============================================================
# F1 SCORE COMPARISON
# ============================================================

def generate_f1_comparison(df):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        df["Model"],
        df["F1_Macro"] * 100
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Macro F1 Score (%)"
    )

    plt.title(
        "Model Macro F1-Score Comparison"
    )

    plt.ylim(
        90,
        100
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for index, value in enumerate(
        df["F1_Macro"] * 100
    ):

        plt.text(
            index,
            value + 0.1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_f1_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ F1 comparison saved:\n"
        f"{output_path}"
    )


# ============================================================
# MCC COMPARISON
# ============================================================

def generate_mcc_comparison(df):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        df["Model"],
        df["MCC"]
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Matthews Correlation Coefficient"
    )

    plt.title(
        "Model MCC Comparison"
    )

    plt.ylim(
        0.90,
        1.0
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for index, value in enumerate(
        df["MCC"]
    ):

        plt.text(
            index,
            value + 0.002,
            f"{value:.4f}",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_mcc_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ MCC comparison saved:\n"
        f"{output_path}"
    )


# ============================================================
# INFERENCE TIME COMPARISON
# ============================================================

def generate_inference_time_comparison(df):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        df["Model"],
        df["Average_Inference_Time_ms"]
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Average Inference Time (ms/image)"
    )

    plt.title(
        "Model Inference Speed Comparison"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for index, value in enumerate(
        df["Average_Inference_Time_ms"]
    ):

        plt.text(
            index,
            value + 0.1,
            f"{value:.2f} ms",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_inference_time_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Inference time comparison saved:\n"
        f"{output_path}"
    )


# ============================================================
# COMBINED PERFORMANCE COMPARISON
# ============================================================

def generate_combined_comparison(df):

    metrics = [
        "Accuracy",
        "F1_Macro",
        "MCC"
    ]

    plot_df = df.copy()

    plt.figure(
        figsize=(12, 7)
    )

    x = range(
        len(plot_df)
    )

    width = 0.25

    for i, metric in enumerate(
        metrics
    ):

        values = plot_df[metric]

        if metric != "MCC":

            values = values * 100

        plt.bar(
            [
                position + i * width
                for position in x
            ],
            values,
            width=width,
            label=metric
        )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Performance"
    )

    plt.title(
        "Overall Model Performance Comparison"
    )

    plt.xticks(
        [
            position + width
            for position in x
        ],
        plot_df["Model"]
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "overall_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Overall comparison saved:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "MODEL PERFORMANCE COMPARISON"
    )

    print("=" * 70)

    comparison_df = load_model_metrics()

    if comparison_df.empty:

        print(
            "\nNo model metrics found."
        )

        return

    comparison_df = save_comparison_table(
        comparison_df
    )

    print(
        "\nGenerating comparison figures...\n"
    )

    generate_accuracy_comparison(
        comparison_df
    )

    generate_f1_comparison(
        comparison_df
    )

    generate_mcc_comparison(
        comparison_df
    )

    generate_inference_time_comparison(
        comparison_df
    )

    generate_combined_comparison(
        comparison_df
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL COMPARISON COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest model based on "
        "test accuracy:"
    )

    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    print(
        f"\nModel: "
        f"{best_model['Model']}"
    )

    print(
        f"Accuracy: "
        f"{best_model['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1-Score: "
        f"{best_model['F1_Macro'] * 100:.2f}%"
    )

    print(
        f"Inference Time: "
        f"{best_model['Average_Inference_Time_ms']:.2f} ms/image"
    )


if __name__ == "__main__":

    main()