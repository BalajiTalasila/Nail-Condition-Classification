import sys
import torch
import torch.nn.functional as F
import timm

from pathlib import Path
from PIL import Image

from torchvision import transforms


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = (
    PROJECT_ROOT
    / "results"
    / "models"
)

MODEL_PATH = (
    MODELS_DIR
    / "efficientnet_b0_best.pth"
)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading model checkpoint..."
    )

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE
    )


    class_names = checkpoint[
        "class_names"
    ]


    model_name = checkpoint[
        "model_name"
    ]


    model = timm.create_model(

        model_name,

        pretrained=False,

        num_classes=len(
            class_names
        )
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    return (

        model,

        class_names
    )


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(

    model,

    image_path,

    class_names
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image_tensor = image_transform(

        image
    ).unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        DEVICE
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        probabilities = F.softmax(

            outputs,

            dim=1
        )


        confidence, predicted_index = torch.max(

            probabilities,

            dim=1
        )


    predicted_index = (

        predicted_index.item()
    )


    confidence = (

        confidence.item()
        * 100
    )


    predicted_class = (

        class_names[
            predicted_index
        ]
    )


    return (

        predicted_class,

        confidence,

        probabilities.cpu().numpy()[0]
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "EFFICIENTNET-B0 NAIL CONDITION PREDICTION"
    )

    print(
        "=" * 70
    )


    print(
        f"\nDevice: {DEVICE}"
    )


    if DEVICE.type == "cuda":

        print(

            "GPU: "

            + torch.cuda.get_device_name(
                0
            )
        )


    if not MODEL_PATH.exists():

        print(

            "\nERROR: Model checkpoint not found!"
        )

        print(
            MODEL_PATH
        )

        sys.exit(1)


    if len(sys.argv) < 2:

        print(

            "\nUsage:"
        )

        print(

            "python "
            ".\\src\\predict_efficientnet_b0.py "
            "<image_path>"
        )

        sys.exit(1)


    image_path = Path(

        sys.argv[1]
    )


    if not image_path.exists():

        print(

            "\nERROR: Image not found!"
        )

        print(
            image_path
        )

        sys.exit(1)


    model, class_names = (

        load_model()
    )


    print(

        "\nLoading image..."
    )


    predicted_class, confidence, probabilities = (

        predict_image(

            model,

            image_path,

            class_names
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "=" * 70
    )


    print(

        f"\nImage: "
        f"{image_path.name}"
    )


    print(

        f"\nPredicted Condition: "
        f"{predicted_class}"
    )


    print(

        f"Confidence: "
        f"{confidence:.2f}%"
    )


    print(

        "\nClass Probabilities:"
    )


    sorted_indices = (

        probabilities.argsort()[::-1]
    )


    for index in sorted_indices:

        probability = (

            probabilities[index]
            * 100
        )


        print(

            f"{class_names[index]}: "
            f"{probability:.4f}%"
        )


    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":

    main()
