import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for CNN-based image classification models.
    """

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0]

    def generate_cam(
        self,
        image_tensor,
        target_class=None,
    ):

        self.model.zero_grad()

        outputs = self.model(
            image_tensor
        )

        if target_class is None:

            target_class = torch.argmax(
                outputs,
                dim=1
            ).item()


        class_score = outputs[
            0,
            target_class
        ]

        class_score.backward()


        gradients = self.gradients
        activations = self.activations


        if gradients is None:

            raise RuntimeError(
                "Gradients were not captured."
            )


        if activations is None:

            raise RuntimeError(
                "Activations were not captured."
            )


        # Calculate importance weight for
        # every feature map

        weights = torch.mean(
            gradients,
            dim=(
                2,
                3,
            ),
            keepdim=True,
        )


        # Weighted combination of
        # feature maps

        heatmap = torch.sum(
            weights * activations,
            dim=1,
        )


        heatmap = torch.relu(
            heatmap
        )


        heatmap = heatmap.squeeze()


        heatmap = (
            heatmap
            .detach()
            .cpu()
            .numpy()
        )


        # Normalize safely

        heatmap_min = np.min(
            heatmap
        )

        heatmap_max = np.max(
            heatmap
        )


        if (
            heatmap_max
            -
            heatmap_min
            >
            1e-8
        ):

            heatmap = (
                heatmap
                -
                heatmap_min
            )

            heatmap = (
                heatmap
                /
                (
                    heatmap_max
                    -
                    heatmap_min
                )
            )

        else:

            heatmap = np.zeros_like(
                heatmap
            )


        return heatmap

    def remove_hooks(
        self,
    ):

        if self.forward_hook is not None:

            self.forward_hook.remove()

        if self.backward_hook is not None:

            self.backward_hook.remove()


def create_gradcam_visualization(
    original_image,
    heatmap,
    alpha=0.45,
):

    # Convert PIL image to RGB NumPy array
    original_rgb = np.array(
        original_image.convert(
            "RGB"
        )
    )

    height, width = (
        original_rgb.shape[:2]
    )

    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(
        heatmap,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # Ensure values are between 0 and 1
    heatmap_resized = np.clip(
        heatmap_resized,
        0,
        1,
    )

    # Convert heatmap to uint8
    heatmap_uint8 = np.uint8(
        heatmap_resized * 255
    )

    # Apply color map
    heatmap_color_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # Convert BGR to RGB
    heatmap_color_rgb = cv2.cvtColor(
        heatmap_color_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Create overlay
    overlay = cv2.addWeighted(
        original_rgb,
        1 - alpha,
        heatmap_color_rgb,
        alpha,
        0,
    )

    return (
        original_rgb,
        heatmap_color_rgb,
        overlay,
    )