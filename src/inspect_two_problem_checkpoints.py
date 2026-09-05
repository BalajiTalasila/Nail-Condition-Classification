from pathlib import Path
import torch

MODEL_DIR = Path("results/models")

for name in [
    "efficientnet_b0_best.pth",
    "proposed_attention_efficientnet_b0_best.pth"
]:
    path = MODEL_DIR / name

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    print("\nCheckpoint type:")
    print(type(checkpoint))

    if isinstance(checkpoint, dict):
        print("\nTop-level keys:")
        print(list(checkpoint.keys()))

        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"\nNested dictionary: {key}")
                print("First keys:")
                print(list(value.keys())[:20])

                for sub_key, tensor in value.items():
                    if hasattr(tensor, "shape"):
                        print(
                            f"{sub_key}: {tuple(tensor.shape)}"
                        )
            elif hasattr(value, "shape"):
                print(
                    f"{key}: {tuple(value.shape)}"
                )

    print()
