import pandas as pd
from pathlib import Path

PROJECT_DIR = Path.cwd()
METRICS_DIR = PROJECT_DIR / "results" / "metrics"

FILES = {
    "EfficientNet-B0": "efficientnet_b0_training_history.csv",
    "DenseNet121": "densenet121_training_history.csv",
    "ConvNeXtV2-Tiny": "convnextv2_tiny_training_history.csv",
    "Proposed Attention-EfficientNet-B0": "proposed_attention_efficientnet_b0_training_history.csv",
}

print("=" * 70)
print("EXISTING TRAINING RUN AUDIT")
print("=" * 70)

for model_name, filename in FILES.items():

    path = METRICS_DIR / filename

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        continue

    df = pd.read_csv(path)

    print("\nColumns:")
    print(list(df.columns))

    print(f"\nNumber of recorded epochs: {len(df)}")

    epoch_candidates = [
        "epoch",
        "Epoch"
    ]

    epoch_column = None

    for candidate in epoch_candidates:
        if candidate in df.columns:
            epoch_column = candidate
            break

    if epoch_column:
        print(
            f"First epoch: {df[epoch_column].min()}"
        )
        print(
            f"Last epoch: {df[epoch_column].max()}"
        )

    print("\nFinal 5 epochs:")

    print(
        df.tail(5).to_string(
            index=False
        )
    )

print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)
