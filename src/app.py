import streamlit as st
import torch
import timm
import numpy as np

from PIL import Image
from pathlib import Path

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nail Condition Classification System",
    page_icon="💅",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )


    class_names = checkpoint[
        "class_names"
    ]


    image_size = checkpoint[
        "image_size"
    ]


    model = timm.create_model(

        "efficientnet_b0",

        pretrained=False,

        num_classes=len(class_names)
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(DEVICE)

    model.eval()


    return (
        model,
        class_names,
        image_size
    )


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

def get_model_transform(
    image_size
):

    return transforms.Compose([

        transforms.Resize(
            (
                image_size,
                image_size
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
    ])


def get_original_transform(
    image_size
):

    return transforms.Compose([

        transforms.Resize(
            (
                image_size,
                image_size
            )
        ),

        transforms.ToTensor()
    ])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(

    image,

    model,

    class_names,

    image_size
):


    transform = get_model_transform(

        image_size
    )


    image_tensor = transform(

        image

    )


    image_tensor = image_tensor.unsqueeze(

        0

    ).to(

        DEVICE

    )


    with torch.no_grad():

        outputs = model(

            image_tensor

        )


        probabilities = torch.softmax(

            outputs,

            dim=1

        )[0]


    probabilities = probabilities.cpu().numpy()


    predicted_index = np.argmax(

        probabilities
    )


    predicted_class = class_names[

        predicted_index

    ]


    confidence = (

        probabilities[
            predicted_index
        ]

        * 100
    )


    return (

        predicted_class,

        confidence,

        probabilities,

        predicted_index
    )


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(

    image,

    model,

    predicted_index,

    image_size
):


    # --------------------------------------------------------
    # PREPARE ORIGINAL IMAGE
    # --------------------------------------------------------

    original_transform = (

        get_original_transform(

            image_size
        )
    )


    rgb_image = original_transform(

        image

    ).numpy()


    rgb_image = np.transpose(

        rgb_image,

        (
            1,
            2,
            0
        )
    )


    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    model_transform = (

        get_model_transform(

            image_size
        )
    )


    input_tensor = model_transform(

        image

    )


    input_tensor = input_tensor.unsqueeze(

        0

    ).to(

        DEVICE
    )


    # --------------------------------------------------------
    # SELECT TARGET LAYER
    # --------------------------------------------------------

    target_layers = [

        model.conv_head

    ]


    # --------------------------------------------------------
    # INITIALIZE GRAD-CAM
    # --------------------------------------------------------

    cam = GradCAM(

        model=model,

        target_layers=target_layers
    )


    # --------------------------------------------------------
    # GENERATE CAM
    # --------------------------------------------------------

    targets = [

        ClassifierOutputTarget(

            predicted_index
        )
    ]


    grayscale_cam = cam(

        input_tensor=input_tensor,

        targets=targets
    )


    grayscale_cam = grayscale_cam[

        0,

        :
    ]


    # --------------------------------------------------------
    # CREATE VISUALIZATION
    # --------------------------------------------------------

    visualization = show_cam_on_image(

        rgb_image,

        grayscale_cam,

        use_rgb=True
    )


    return (

        rgb_image,

        grayscale_cam,

        visualization
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():


    # --------------------------------------------------------
    # APPLICATION TITLE
    # --------------------------------------------------------

    st.title(

        "💅 Nail Condition Classification System"

    )


    st.write(

        "AI-powered nail image classification using "

        "EfficientNet-B0 with Grad-CAM explainability."

    )


    st.divider()


    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header(

        "Model Information"

    )


    st.sidebar.write(

        "**Model:** EfficientNet-B0"

    )


    st.sidebar.write(

        "**Test Accuracy:** 98.44%"

    )


    st.sidebar.write(

        "**Macro F1-Score:** 98.36%"

    )


    st.sidebar.write(

        "**MCC:** 98.11%"

    )


    st.sidebar.write(

        "**Inference Time:** 2.14 ms/image"

    )


    st.sidebar.divider()


    st.sidebar.subheader(

        "Supported Classes"

    )


    supported_classes = [

        "Acral Lentiginous Melanoma",

        "Healthy Nail",

        "Onychogryphosis",

        "Blue Finger",

        "Clubbing",

        "Pitting"
    ]


    for condition in supported_classes:

        st.sidebar.write(

            f"• {condition}"
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model, class_names, image_size = (

            load_model()

        )


    except Exception as error:

        st.error(

            "Unable to load the trained model."

        )


        st.exception(

            error
        )


        return


    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(

        "Upload a nail image",

        type=[

            "jpg",

            "jpeg",

            "png"
        ]
    )


    # --------------------------------------------------------
    # PROCESS UPLOADED IMAGE
    # --------------------------------------------------------

    if uploaded_file is not None:


        image = Image.open(

            uploaded_file

        ).convert(

            "RGB"
        )


        # ----------------------------------------------------
        # DISPLAY ORIGINAL IMAGE
        # ----------------------------------------------------

        column1, column2 = st.columns(

            2
        )


        with column1:


            st.subheader(

                "Uploaded Image"

            )


            st.image(

                image,

                use_container_width=True
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with column2:


            st.subheader(

                "Prediction Result"

            )


            with st.spinner(

                "Analyzing image..."

            ):


                (

                    predicted_class,

                    confidence,

                    probabilities,

                    predicted_index

                ) = predict_image(

                    image,

                    model,

                    class_names,

                    image_size
                )


            display_class = (

                predicted_class

                .replace(

                    "_",

                    " "
                )
            )


            st.success(

                f"Predicted Condition: "

                f"{display_class}"

            )


            st.metric(

                "Prediction Confidence",

                f"{confidence:.2f}%"
            )


            # ------------------------------------------------
            # ALL PROBABILITIES
            # ------------------------------------------------

            st.subheader(

                "All Class Probabilities"

            )


            sorted_indices = np.argsort(

                probabilities

            )[::-1]


            for index in sorted_indices:


                class_name = (

                    class_names[index]

                    .replace(

                        "_",

                        " "
                    )
                )


                probability = (

                    probabilities[index]

                    * 100
                )


                st.write(

                    f"**{class_name}**"

                )


                st.progress(

                    float(

                        probabilities[index]

                    )
                )


                st.caption(

                    f"{probability:.2f}%"

                )


        # ====================================================
        # GRAD-CAM SECTION
        # ====================================================

        st.divider()


        st.header(

            "🔍 AI Visual Explanation using Grad-CAM"

        )


        st.write(

            "The Grad-CAM visualization highlights the "

            "regions of the image that were most important "

            "for the model's prediction."

        )


        with st.spinner(

            "Generating Grad-CAM visualization..."

        ):


            (

                gradcam_original,

                grayscale_cam,

                visualization

            ) = generate_gradcam(

                image,

                model,

                predicted_index,

                image_size
            )


        # ----------------------------------------------------
        # DISPLAY GRAD-CAM
        # ----------------------------------------------------

        cam_column1, cam_column2 = st.columns(

            2
        )


        with cam_column1:


            st.subheader(

                "Original Image"

            )


            st.image(

                gradcam_original,

                use_container_width=True
            )


        with cam_column2:


            st.subheader(

                "Grad-CAM Visualization"

            )


            st.image(

                visualization,

                use_container_width=True
            )


        st.info(

            "🔴 Red and yellow regions represent areas "

            "that contributed more strongly to the model's "

            "prediction. Blue regions contributed less."

        )


        # ====================================================
        # MEDICAL DISCLAIMER
        # ====================================================

        st.divider()


        st.warning(

            "⚠️ Medical Disclaimer: This application is "

            "developed for educational and research purposes. "

            "It should not be used as a replacement for "

            "professional medical diagnosis. Please consult "

            "a qualified healthcare professional for medical "

            "advice."

        )


    else:


        st.info(

            "👆 Upload a nail image to begin classification."

        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()