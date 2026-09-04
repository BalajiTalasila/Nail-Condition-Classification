import timm
import torch


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# PRINT MODEL ARCHITECTURE
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENTNET-B0 MODEL ARCHITECTURE")
print("=" * 70)

print(model)


# ============================================================
# PRINT CONVOLUTIONAL LAYERS
# ============================================================

print("\n" + "=" * 70)
print("CONVOLUTIONAL LAYERS")
print("=" * 70)

for name, module in model.named_modules():

    if isinstance(module, torch.nn.Conv2d):

        print(
            f"{name}: "
            f"{module}"
        )