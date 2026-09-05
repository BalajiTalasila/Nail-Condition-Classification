from pathlib import Path
import torch

MODEL_DIR = Path("results/models")

for name in [
    "efficientnet_b0_best.pth",
    "proposed_attention_efficientnet_b0_best.pth"
]:
    path = MODEL_DIR / name
    checkpoint = torch.load(path, map_location="cpu")

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("Epoch:", checkpoint.get("epoch"))
    print("Model name:", checkpoint.get("model_name"))
    print("Class names:", checkpoint.get("class_names"))
    print("Validation accuracy:", checkpoint.get("validation_accuracy"))
    print("Validation loss:", checkpoint.get("validation_loss"))

    state_dict = checkpoint["model_state_dict"]

    print("\nTotal model parameters:", len(state_dict))

    print("\nImportant classifier keys:")

    for key, value in state_dict.items():
        if (
            "classifier" in key
            or "channel_attention" in key
        ):
            if hasattr(value, "shape"):
                print(f"{key}: {tuple(value.shape)}")
