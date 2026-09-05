from pathlib import Path
import torch

MODEL_PATH = Path(
    "results/models/proposed_attention_efficientnet_b0_best.pth"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

state_dict = checkpoint["model_state_dict"]

print("\nChecking proposed model checkpoint structure...\n")

required_prefixes = [
    "backbone.",
    "channel_attention.",
    "classifier."
]

for prefix in required_prefixes:
    matches = [
        key
        for key in state_dict.keys()
        if key.startswith(prefix)
    ]

    print(f"{prefix}")
    print(f"Number of matching keys: {len(matches)}")

    if len(matches) > 0:
        print("Status: FOUND")
    else:
        print("Status: NOT FOUND")

    print()

print("Checkpoint structure check completed.")
