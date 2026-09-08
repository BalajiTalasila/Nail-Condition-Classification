# ============================================================
# GRAD-CAM ERROR ANALYSIS
# EFFICIENTNET-B0 - LEAKAGE-SAFE GROUPED DATASET
# ============================================================

from pathlib import Path
import timm
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "models"
    / "efficientnet_b0_grouped_best.pth"
)

PREDICTIONS_PATH = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "efficientnet_b0_grouped_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "gradcam_grouped_errors"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 224

TARGET_TRUE_CLASS = "blue_finger"

TARGET_PREDICTED_CLASS = "clubbing"

MAX_IMAGES = 15


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained EfficientNet-B0 grouped model.

    The checkpoint was trained using the timm implementation of
    EfficientNet-B0, so the same architecture must be used here.
    """

    print("\nLoading EfficientNet-B0 model...")

    checkpoint_path = Path(
        "results/models/efficientnet_b0_grouped_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Extract checkpoint information
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "efficientnet_b0")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model architecture: {model_name}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # IMPORTANT:
    # Create the same timm EfficientNet architecture used for training
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names)
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    # Move model to device
    model = model.to(DEVICE)

    # Evaluation mode
    model.eval()

    print("Model loaded successfully.")

    return model, class_names


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None

        self.activations = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self.save_activations
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self.save_gradients
            )
        )


    def save_activations(
        self,
        module,
        input_data,
        output
    ):

        self.activations = output


    def save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate_cam(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        gradients = (
            self.gradients
        )

        activations = (
            self.activations
        )

        weights = torch.mean(
            gradients,
            dim=(
                2,
                3
            ),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze()

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        if (
            np.max(cam)
            - np.min(cam)
            > 0
        ):

            cam = (
                cam
                - np.min(cam)
            )

            cam = (
                cam
                / np.max(cam)
            )

        return cam


    def remove_hooks(
        self
    ):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(

    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
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

    ]

)


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    cam
):

    image_array = np.array(
        image
    )

    image_array = cv2.resize(

        image_array,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    )

    cam_resized = cv2.resize(

        cam,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    )

    heatmap = cv2.applyColorMap(

        np.uint8(
            255
            * cam_resized
        ),

        cv2.COLORMAP_JET

    )

    heatmap = cv2.cvtColor(

        heatmap,

        cv2.COLOR_BGR2RGB

    )

    overlay = cv2.addWeighted(

        image_array,

        0.6,

        heatmap,

        0.4,

        0

    )

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GRAD-CAM ERROR ANALYSIS"
    )

    print(
        "EFFICIENTNET-B0 GROUPED SPLIT"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nLoading predictions..."
    )

    predictions_df = pd.read_csv(
        PREDICTIONS_PATH
    )


    # --------------------------------------------------------
    # FILTER TARGET ERRORS
    # --------------------------------------------------------

    errors_df = predictions_df[

        (
            predictions_df[
                "true_class"
            ]
            == TARGET_TRUE_CLASS
        )

        &

        (
            predictions_df[
                "predicted_class"
            ]
            == TARGET_PREDICTED_CLASS
        )

    ].copy()


    errors_df = errors_df.sort_values(

        by="confidence_percent",

        ascending=False

    )


    errors_df = errors_df.head(
        MAX_IMAGES
    )


    print(

        f"\n{TARGET_TRUE_CLASS} → "
        f"{TARGET_PREDICTED_CLASS} errors: "
        f"{len(errors_df)}"

    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, class_names = load_model()


    class_to_index = {

        class_name: index

        for index, class_name

        in enumerate(
            class_names
        )

    }


    target_layer = model.conv_head


    grad_cam = GradCAM(

        model,

        target_layer

    )


    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for index, row in errors_df.iterrows():

        image_path = Path(

            row[
                "image_path"
            ]

        )


        print(

            f"\nProcessing: "
            f"{image_path.name}"

        )


        image = Image.open(

            image_path

        ).convert(
            "RGB"
        )


        input_tensor = transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )


        predicted_index = class_to_index[

            row[
                "predicted_class"
            ]

        ]


        cam = grad_cam.generate_cam(

            input_tensor,

            predicted_index

        )


        overlay = create_overlay(

            image,

            cam

        )


        original_image = np.array(

            image.resize(

                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )

            )

        )


        combined = np.hstack(

            [

                original_image,

                overlay

            ]

        )


        output_name = (

            f"{index}_"

            f"{image_path.stem}"

            f"_gradcam.png"

        )


        output_path = (

            OUTPUT_DIR

            / output_name

        )


        Image.fromarray(

            combined

        ).save(

            output_path

        )


        print(

            f"Saved: "
            f"{output_path}"

        )


    # --------------------------------------------------------
    # REMOVE HOOKS
    # --------------------------------------------------------

    grad_cam.remove_hooks()


    print(
        "\n"
        + "=" * 70
    )

    print(
        "GRAD-CAM ANALYSIS COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()