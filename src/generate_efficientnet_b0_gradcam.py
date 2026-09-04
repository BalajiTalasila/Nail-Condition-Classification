import os
import random

import numpy as np
import torch
import timm

from PIL import Image
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEST_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "test"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "models",
    "efficientnet_b0_best.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "figures",
    "efficientnet_b0_gradcam"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


IMAGE_SIZE = 224
NUM_CLASSES = 6
SEED = 42

SAMPLES_PER_CLASS = 3


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

normalize = transforms.Normalize(

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


model_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    normalize
])


original_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading EfficientNet-B0 model...")

    model = timm.create_model(

        "efficientnet_b0",

        pretrained=False,

        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE
    )


    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model.eval()


    print(
        "✓ Model loaded successfully"
    )


    return model


# ============================================================
# GET IMAGE SAMPLES
# ============================================================

def get_samples(dataset):

    class_samples = {}


    for index, (
        image_path,
        class_index
    ) in enumerate(dataset.samples):


        if class_index not in class_samples:

            class_samples[
                class_index
            ] = []


        if len(
            class_samples[
                class_index
            ]
        ) < SAMPLES_PER_CLASS:


            class_samples[
                class_index
            ].append(
                image_path
            )


    return class_samples


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam():

    print("\n" + "=" * 70)

    print(
        "GENERATING EFFICIENTNET-B0 GRAD-CAM VISUALIZATIONS"
    )

    print("=" * 70)


    print(
        f"\nDevice: {DEVICE}"
    )


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print(
        "\nLoading test dataset..."
    )


    dataset = datasets.ImageFolder(
        TEST_DIR
    )


    class_names = dataset.classes


    print(
        f"Classes: {len(class_names)}"
    )


    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{index}: {class_name}"
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # SELECT TARGET LAYER
    # --------------------------------------------------------

    target_layers = [

        model.conv_head

    ]


    print(
        "\nGrad-CAM target layer:"
    )

    print(
        target_layers[0]
    )


    # --------------------------------------------------------
    # INITIALIZE GRAD-CAM
    # --------------------------------------------------------

    cam = GradCAM(

        model=model,

        target_layers=target_layers
    )


    # --------------------------------------------------------
    # SELECT IMAGE SAMPLES
    # --------------------------------------------------------

    class_samples = get_samples(
        dataset
    )


    total_images = sum(

        len(images)

        for images in class_samples.values()

    )


    print(
        f"\nGenerating Grad-CAM for "
        f"{total_images} images..."
    )


    # --------------------------------------------------------
    # PROCESS EACH CLASS
    # --------------------------------------------------------

    for class_index, image_paths in (
        class_samples.items()
    ):


        class_name = class_names[
            class_index
        ]


        print(
            f"\nProcessing class: "
            f"{class_name}"
        )


        class_output_dir = os.path.join(

            OUTPUT_DIR,

            class_name
        )


        os.makedirs(

            class_output_dir,

            exist_ok=True
        )


        # ----------------------------------------------------
        # PROCESS EACH IMAGE
        # ----------------------------------------------------

        for image_number, image_path in enumerate(

            image_paths,

            start=1
        ):


            # ------------------------------------------------
            # LOAD ORIGINAL IMAGE
            # ------------------------------------------------

            original_image = Image.open(

                image_path

            ).convert(

                "RGB"
            )


            # ------------------------------------------------
            # PREPARE RGB IMAGE
            # ------------------------------------------------

            rgb_image = original_transform(

                original_image

            ).numpy()


            rgb_image = np.transpose(

                rgb_image,

                (1, 2, 0)

            )


            # ------------------------------------------------
            # PREPARE MODEL INPUT
            # ------------------------------------------------

            input_tensor = model_transform(

                original_image

            )


            input_tensor = input_tensor.unsqueeze(

                0

            ).to(

                DEVICE

            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            with torch.no_grad():

                output = model(

                    input_tensor

                )


                probabilities = torch.softmax(

                    output,

                    dim=1

                )


                predicted_class = torch.argmax(

                    probabilities,

                    dim=1

                ).item()


                confidence = probabilities[

                    0,

                    predicted_class

                ].item()


            # ------------------------------------------------
            # GENERATE GRAD-CAM
            # ------------------------------------------------

            targets = [
                ClassifierOutputTarget(predicted_class)
            ]

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )

            grayscale_cam = grayscale_cam[
                0,
                :
            ]


            # ------------------------------------------------
            # CREATE OVERLAY
            # ------------------------------------------------

            visualization = show_cam_on_image(

                rgb_image,

                grayscale_cam,

                use_rgb=True

            )


            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            output_filename = (

                f"sample_{image_number}"

                f"_true_{class_name}"

                f"_pred_{class_names[predicted_class]}"

                f"_conf_{confidence:.4f}.png"

            )


            output_path = os.path.join(

                class_output_dir,

                output_filename

            )


            Image.fromarray(

                visualization

            ).save(

                output_path

            )


            print(

                f"✓ Sample {image_number} saved"

            )


            print(

                f"  True class: "

                f"{class_name}"

            )


            print(

                f"  Predicted: "

                f"{class_names[predicted_class]}"

            )


            print(

                f"  Confidence: "

                f"{confidence * 100:.2f}%"

            )


    print(

        "\n" + "=" * 70

    )


    print(

        "GRAD-CAM GENERATION COMPLETED"

    )


    print(

        "=" * 70

    )


    print(

        "\nResults saved to:"

    )


    print(

        OUTPUT_DIR

    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_gradcam()