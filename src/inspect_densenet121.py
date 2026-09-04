import torch
import timm

print("\n" + "=" * 70)
print("DENSENET121 MODEL ARCHITECTURE")
print("=" * 70)

# Create DenseNet121 model
model = timm.create_model(
    "densenet121",
    pretrained=False,
    num_classes=6
)

# Print complete architecture
print(model)

print("\n" + "=" * 70)
print("CONVOLUTIONAL LAYERS")
print("=" * 70)

# Print all convolutional layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        print(f"{name}: {module}")