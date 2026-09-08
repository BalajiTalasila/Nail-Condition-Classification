# ============================================================
# GRAD-CAM ANALYSIS
# EFFICIENTNET-B0 - LEAKAGE-SAFE GROUPED SPLIT
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from PIL import Image

import matplotlib.pyplot as plt

from torchvision import transforms

import timm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"

MODELS_DIR = RESULTS_DIR / "models"

METRICS_DIR = RESULTS_DIR / "metrics"

FIGURES_DIR = RESULTS_DIR / "figures"

OUTPUT_DIR = FIGURES_DIR / "grouped_gradcam"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnet_b0"

IMAGE_SIZE = 224

NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

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

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0]
        )


    def generate_cam(
        self,
        input_tensor,
        target_class=None
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        if target_class is None:

            target_class = (
                torch.argmax(
                    output,
                    dim=1
                ).item()
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

            weights
            * activations,

            dim=1
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(

            cam.unsqueeze(1),

            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            mode="bilinear",

            align_corners=False
        )

        cam = (
            cam.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        cam = cam - np.min(
            cam
        )

        if np.max(cam) > 0:

            cam = (

                cam
                / np.max(cam)

            )

        probabilities = (

            torch.softmax(

                output,

                dim=1

            )

            .detach()

            .cpu()

            .numpy()[0]
        )

        return (

            cam,

            probabilities,

            target_class
        )


    def remove_hooks(
        self
    ):

        self.forward_hook.remove()

        self.backward_hook.remove()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading model checkpoint..."
    )

    checkpoint_path = (

        MODELS_DIR
        / "efficientnet_b0_grouped_best.pth"
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE,

        weights_only=False
    )

    model = timm.create_model(

        MODEL_NAME,

        pretrained=False,

        num_classes=NUM_CLASSES
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

    class_names = (

        checkpoint[
            "class_names"
        ]
    )

    print(
        "Model loaded successfully."
    )

    return (

        model,

        class_names
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image_path
):

    image = Image.open(

        image_path

    ).convert(
        "RGB"
    )

    original_image = image.resize(

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    input_tensor = (

        transform(
            original_image
        )

        .unsqueeze(0)

        .to(DEVICE)
    )

    return (

        original_image,

        input_tensor
    )


# ============================================================
# CREATE HEATMAP OVERLAY
# ============================================================

def create_overlay(
    image,
    cam
):

    image_array = (

        np.array(image)

        / 255.0
    )

    heatmap = plt.cm.jet(
        cam
    )[:, :, :3]

    overlay = (

        0.6
        * image_array

        +

        0.4
        * heatmap
    )

    overlay = np.clip(

        overlay,

        0,

        1
    )

    return overlay


# ============================================================
# VISUALIZE ONE IMAGE
# ============================================================

def visualize_image(

    image_path,

    true_class,

    predicted_class,

    confidence,

    model,

    gradcam,

    class_names,

    output_name

):

    try:

        original_image, input_tensor = (

            load_image(
                image_path
            )
        )

        cam, probabilities, predicted_index = (

            gradcam.generate_cam(
                input_tensor
            )
        )

        overlay = (

            create_overlay(

                original_image,

                cam
            )
        )

        top_indices = np.argsort(

            probabilities

        )[-3:][::-1]

        top_predictions = []

        for index in top_indices:

            top_predictions.append(

                f"{class_names[index]}: "
                f"{probabilities[index] * 100:.2f}%"
            )

        figure, axes = plt.subplots(

            1,

            3,

            figsize=(
                15,
                5
            )
        )

        # Original image

        axes[0].imshow(

            original_image
        )

        axes[0].set_title(

            "Original Image\n"
            f"True: {true_class}"
        )

        axes[0].axis(
            "off"
        )


        # Grad-CAM

        axes[1].imshow(

            cam,

            cmap="jet"
        )

        axes[1].set_title(

            "Grad-CAM Attention Map"
        )

        axes[1].axis(
            "off"
        )


        # Overlay

        axes[2].imshow(

            overlay
        )

        axes[2].set_title(

            "Prediction Overlay\n"
            f"Predicted: {predicted_class}\n"
            f"Confidence: {confidence:.2f}%"
        )

        axes[2].axis(
            "off"
        )


        figure.suptitle(

            " | ".join(
                top_predictions
            ),

            fontsize=10
        )

        plt.tight_layout()

        output_path = (

            OUTPUT_DIR

            / output_name
        )

        plt.savefig(

            output_path,

            dpi=200,

            bbox_inches="tight"
        )

        plt.close()

        print(

            f"Saved: "
            f"{output_path.name}"
        )

    except Exception as error:

        print(

            f"Error processing "
            f"{image_path}"
        )

        print(error)


# ============================================================
# CREATE BATCH GRAD-CAM FIGURE
# ============================================================

def create_batch_visualization(

    dataframe,

    title,

    model,

    gradcam,

    class_names,

    output_name,

    max_images=9

):

    dataframe = dataframe.head(
        max_images
    )

    total_images = len(
        dataframe
    )

    if total_images == 0:

        print(
            f"\nNo images found for: "
            f"{title}"
        )

        return


    columns = 3

    rows = int(

        np.ceil(
            total_images
            / columns
        )
    )

    figure, axes = plt.subplots(

        rows,

        columns,

        figsize=(

            15,

            5
            * rows

        )
    )

    axes = np.array(
        axes
    ).reshape(
        -1
    )


    for axis in axes:

        axis.axis(
            "off"
        )


    for index, (

        _,

        row

    ) in enumerate(

        dataframe.iterrows()

    ):

        try:

            image_path = row[
                "image_path"
            ]

            image, input_tensor = (

                load_image(
                    image_path
                )
            )

            cam, probabilities, predicted_index = (

                gradcam.generate_cam(
                    input_tensor
                )
            )

            overlay = create_overlay(

                image,

                cam
            )

            axes[index].imshow(

                overlay
            )

            axes[index].set_title(

                f"True: "
                f"{row['true_class']}\n"

                f"Predicted: "
                f"{row['predicted_class']}\n"

                f"Confidence: "
                f"{row['confidence_percent']:.2f}%",

                fontsize=9
            )

            axes[index].axis(
                "off"
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )


    figure.suptitle(

        title,

        fontsize=16,

        fontweight="bold"
    )

    plt.tight_layout()

    output_path = (

        OUTPUT_DIR

        / output_name
    )

    plt.savefig(

        output_path,

        dpi=200,

        bbox_inches="tight"
    )

    plt.close()

    print(

        f"Saved batch visualization: "
        f"{output_path.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "EFFICIENTNET-B0 GROUPED SPLIT GRAD-CAM ANALYSIS"
    )

    print("=" * 70)


    print(

        f"\nDevice: "
        f"{DEVICE}"
    )


    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    prediction_path = (

        METRICS_DIR

        / "efficientnet_b0_grouped_predictions.csv"
    )

    predictions = pd.read_csv(

        prediction_path
    )

    print(

        f"\nTotal predictions: "
        f"{len(predictions)}"
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, class_names = (

        load_model()
    )


    # --------------------------------------------------------
    # SELECT TARGET LAYER
    # --------------------------------------------------------

    target_layer = (

        model.conv_head
    )


    gradcam = GradCAM(

        model,

        target_layer
    )


    # ========================================================
    # ANALYSIS 1
    # BLUE_FINGER → CLUBBING
    # ========================================================

    print(

        "\nAnalyzing blue_finger → clubbing errors..."
    )

    blue_to_clubbing = predictions[

        (

            predictions[
                "true_class"
            ]

            == "blue_finger"

        )

        &

        (

            predictions[
                "predicted_class"
            ]

            == "clubbing"

        )

    ].sort_values(

        "confidence_percent",

        ascending=False
    )


    create_batch_visualization(

        dataframe=blue_to_clubbing,

        title=(
            "Grad-CAM: "
            "Blue Finger Predicted as Clubbing"
        ),

        model=model,

        gradcam=gradcam,

        class_names=class_names,

        output_name=(
            "gradcam_blue_finger_to_clubbing.png"
        ),

        max_images=9
    )


    # ========================================================
    # ANALYSIS 2
    # HIGH-CONFIDENCE ERRORS
    # ========================================================

    print(

        "\nAnalyzing high-confidence errors..."
    )

    high_confidence_errors = predictions[

        (

            predictions[
                "correct_prediction"
            ]

            == False

        )

        &

        (

            predictions[
                "confidence_percent"
            ]

            >= 90

        )

    ].sort_values(

        "confidence_percent",

        ascending=False
    )


    create_batch_visualization(

        dataframe=high_confidence_errors,

        title=(
            "Grad-CAM: "
            "High-Confidence Errors"
        ),

        model=model,

        gradcam=gradcam,

        class_names=class_names,

        output_name=(
            "gradcam_high_confidence_errors.png"
        ),

        max_images=9
    )


    # ========================================================
    # ANALYSIS 3
    # ONYCHOGRYPHOSIS → MELANOMA
    # ========================================================

    print(

        "\nAnalyzing Onychogryphosis "
        "→ Acral Lentiginous Melanoma errors..."
    )

    onych_to_melanoma = predictions[

        (

            predictions[
                "true_class"
            ]

            == "Onychogryphosis"

        )

        &

        (

            predictions[
                "predicted_class"
            ]

            == "Acral_Lentiginous_Melanoma"

        )

    ].sort_values(

        "confidence_percent",

        ascending=False
    )


    create_batch_visualization(

        dataframe=onych_to_melanoma,

        title=(
            "Grad-CAM: "
            "Onychogryphosis Predicted as Melanoma"
        ),

        model=model,

        gradcam=gradcam,

        class_names=class_names,

        output_name=(
            "gradcam_onychogryphosis_to_melanoma.png"
        ),

        max_images=9
    )


    # ========================================================
    # ANALYSIS 4
    # CORRECT BLUE_FINGER PREDICTIONS
    # ========================================================

    print(

        "\nAnalyzing correctly classified "
        "blue_finger images..."
    )

    correct_blue = predictions[

        (

            predictions[
                "true_class"
            ]

            == "blue_finger"

        )

        &

        (

            predictions[
                "predicted_class"
            ]

            == "blue_finger"

        )

    ].sort_values(

        "confidence_percent",

        ascending=False
    )


    create_batch_visualization(

        dataframe=correct_blue,

        title=(
            "Grad-CAM: "
            "Correctly Classified Blue Finger Images"
        ),

        model=model,

        gradcam=gradcam,

        class_names=class_names,

        output_name=(
            "gradcam_correct_blue_finger.png"
        ),

        max_images=9
    )


    # --------------------------------------------------------
    # REMOVE HOOKS
    # --------------------------------------------------------

    gradcam.remove_hooks()


    print("\n" + "=" * 70)

    print(
        "GRAD-CAM ANALYSIS COMPLETED"
    )

    print("=" * 70)

    print(
        "\nFigures saved in:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()