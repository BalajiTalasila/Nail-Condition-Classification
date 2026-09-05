import pandas as pd
from pathlib import Path

PROJECT_DIR = Path.cwd()
METRICS_DIR = PROJECT_DIR / "results" / "metrics"

FILES = [
    "efficientnet_b0_training_history.csv",
    "densenet121_training_history.csv",
    "convnextv2_tiny_training_history.csv",
    "proposed_attention_efficientnet_b0_training_history.csv",
]

REQUIRED_CONCEPTS = {
    "Epoch": ["epoch", "Epoch"],
    "Training Loss": ["train_loss", "Train_Loss"],
    "Validation Loss": ["validation_loss", "Validation_Loss", "val_loss", "Val_Loss"],
    "Training Accuracy": ["train_accuracy", "Train_Accuracy"],
    "Validation Accuracy": [
        "validation_accuracy",
        "Validation_Accuracy",
        "val_accuracy",
        "Val_Accuracy"
    ],
    "Learning Rate": ["learning_rate", "Learning_Rate"],
    "Epoch Time": ["epoch_time_seconds", "Epoch_Time_Seconds"],
}

print("=" * 70)
print("TRAINING HISTORY COMPLETENESS AUDIT")
print("=" * 70)

for filename in FILES:

    path = METRICS_DIR / filename

    print("\n" + "-" * 70)
    print(filename)
    print("-" * 70)

    if not path.exists():

        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    columns = list(df.columns)

    print("\nAvailable columns:")
    print(columns)

    print("\nRequired metrics:")

    missing = []

    for concept, candidates in REQUIRED_CONCEPTS.items():

        found = [
            column
            for column in candidates
            if column in columns
        ]

        if found:

            print(
                f"✓ {concept}: {found[0]}"
            )

        else:

            print(
                f"✗ {concept}: MISSING"
            )

            missing.append(concept)

    if not missing:

        print(
            "\nSTATUS: COMPLETE"
        )

    else:

        print(
            "\nSTATUS: INCOMPLETE"
        )

        print(
            "Missing:",
            ", ".join(missing)
        )

print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)
