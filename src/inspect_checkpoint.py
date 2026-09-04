import torch
from pathlib import Path


MODEL_PATH = Path("results/models/efficientnet_b0_best.pth")


def main():
    print("=" * 70)
    print("INSPECTING EFFICIENTNET-B0 CHECKPOINT")
    print("=" * 70)

    print(f"\nModel path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print("\nERROR: Model file not found!")
        return

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    print("\nCheckpoint type:")
    print(type(checkpoint))

    if isinstance(checkpoint, dict):
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f" - {key}")

        print("\nDetailed information:")

        for key, value in checkpoint.items():
            print(f"\nKEY: {key}")
            print(f"TYPE: {type(value)}")

            if isinstance(value, dict):
                print(f"NUMBER OF ITEMS: {len(value)}")

                sample_keys = list(value.keys())[:10]

                print("FIRST KEYS:")
                for sample_key in sample_keys:
                    print(f"  {sample_key}")

            else:
                try:
                    print(f"VALUE: {value}")
                except Exception:
                    print("VALUE: [Could not display]")

    else:
        print("\nThe checkpoint is not a dictionary.")

    print("\n" + "=" * 70)
    print("CHECKPOINT INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    