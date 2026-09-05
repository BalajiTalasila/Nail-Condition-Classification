from pathlib import Path
import torch


PROJECT_DIR = Path.cwd()

MODELS_DIR = (
    PROJECT_DIR
    / "results"
    / "models"
)


MODEL_FILES = [
    "efficientnet_b0_best.pth",
    "densenet121_best.pth",
    "convnextv2_tiny_best.pth",
    "proposed_attention_efficientnet_b0_best.pth"
]


print("=" * 70)
print("MODEL CHECKPOINT INSPECTION")
print("=" * 70)


for filename in MODEL_FILES:

    checkpoint_path = (
        MODELS_DIR
        / filename
    )

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    if not checkpoint_path.exists():

        print("ERROR: Checkpoint not found.")

        continue

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        print(
            f"\nCheckpoint type: "
            f"{type(checkpoint)}"
        )

        if isinstance(
            checkpoint,
            dict
        ):

            print("\nCheckpoint keys:")

            for key in checkpoint.keys():

                print(
                    f"  - {key}"
                )

            if (
                "epoch"
                in checkpoint
            ):

                print(
                    f"\nSaved epoch: "
                    f"{checkpoint['epoch']}"
                )

            if (
                "validation_accuracy"
                in checkpoint
            ):

                print(
                    f"Validation accuracy: "
                    f"{checkpoint['validation_accuracy']}"
                )

            if (
                "validation_loss"
                in checkpoint
            ):

                print(
                    f"Validation loss: "
                    f"{checkpoint['validation_loss']}"
                )

            if (
                "class_names"
                in checkpoint
            ):

                print(
                    "\nSaved class names:"
                )

                print(
                    checkpoint[
                        "class_names"
                    ]
                )

            if (
                "model_state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

                print(
                    f"\nNumber of model parameters "
                    f"stored as tensors: "
                    f"{len(state_dict)}"
                )

                print(
                    "\nFirst 20 state dictionary keys:"
                )

                for key in list(
                    state_dict.keys()
                )[:20]:

                    print(
                        f"  {key}"
                    )

        else:

            print(
                "\nCheckpoint is not a dictionary."
            )

    except Exception as error:

        print(
            f"\nERROR: "
            f"{type(error).__name__}: "
            f"{error}"
        )


print("\n" + "=" * 70)
print("CHECKPOINT INSPECTION COMPLETED")
print("=" * 70)
