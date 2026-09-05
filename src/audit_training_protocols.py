from pathlib import Path
import re

PROJECT_DIR = Path.cwd()
SRC_DIR = PROJECT_DIR / "src"

FILES = {
    "EfficientNet-B0": "train_efficientnet_b0.py",
    "DenseNet121": "train_densenet121.py",
    "ConvNeXtV2-Tiny": "train_convnextv2.py",
    "Proposed Attention-EfficientNet-B0": "train_proposed_model.py",
}

PATTERNS = [
    "BATCH_SIZE",
    "NUM_EPOCHS",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "PATIENCE",
    "IMAGE_SIZE",
    "DROPOUT",
]

print("=" * 70)
print("TRAINING PROTOCOL COMPARISON")
print("=" * 70)

for model_name, filename in FILES.items():

    path = SRC_DIR / filename

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    if not path.exists():
        print(f"ERROR: Missing file: {filename}")
        continue

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    for pattern in PATTERNS:

        matches = re.findall(
            rf"^{pattern}\s*=\s*.+$",
            text,
            flags=re.MULTILINE
        )

        if matches:
            for match in matches:
                print(match)

    print("\nOptimizer:")

    if "optim.AdamW" in text:
        print("AdamW")

    elif "optim.Adam" in text:
        print("Adam")

    elif "optim.SGD" in text:
        print("SGD")

    else:
        print("Could not automatically identify")

    print("\nLoss:")

    if "CrossEntropyLoss" in text:

        if "weight=class_weights" in text:
            print("Weighted CrossEntropyLoss")
        else:
            print("CrossEntropyLoss")

    else:
        print("Could not automatically identify")

    print("\nScheduler:")

    if "ReduceLROnPlateau" in text:
        print("ReduceLROnPlateau")

    elif "CosineAnnealing" in text:
        print("Cosine Annealing")

    else:
        print("Could not automatically identify")

print("\n" + "=" * 70)
print("PROTOCOL AUDIT COMPLETED")
print("=" * 70)
