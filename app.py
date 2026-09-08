import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import pandas as pd

from PIL import Image
from torchvision import transforms
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# IMPORT PDF REPORT GENERATOR
# ============================================================

try:

    from report_generator import generate_prediction_report

    PDF_REPORT_AVAILABLE = True

    PDF_ERROR = None


except Exception as error:

    PDF_REPORT_AVAILABLE = False

    PDF_ERROR = str(error)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Nail Condition AI",

    page_icon="💅",

    layout="wide"

)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(

    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #666666;
        margin-bottom: 30px;
    }

    .condition-name {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: gray;
        margin-top: 50px;
        padding-top: 20px;
    }

    </style>
    """,

    unsafe_allow_html=True

)


# ============================================================
# PATH CONFIGURATION
# ============================================================

MODEL_PATH = (

    PROJECT_ROOT
    / "results"
    / "models"
    / "efficientnet_b0_best.pth"

)


METRICS_DIR = (

    PROJECT_ROOT
    / "results"
    / "metrics"

)


VISUALIZATIONS_DIR = (

    PROJECT_ROOT
    / "results"
    / "visualizations"

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
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([

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

])


# ============================================================
# CONDITION INFORMATION
# ============================================================

CONDITION_INFO = {

    "Acral_Lentiginous_Melanoma": {

        "display_name":
        "Acral Lentiginous Melanoma",

        "description":
        (
            "A condition involving abnormal dark pigmentation "
            "that can appear around or beneath the nail."
        ),

        "category":
        (
            "Abnormal nail pigmentation pattern detected "
            "by the model."
        )

    },


    "Healthy_Nail": {

        "display_name":
        "Healthy Nail",

        "description":
        (
            "The image contains characteristics commonly "
            "associated with a healthy appearing nail."
        ),

        "category":
        (
            "No visible abnormal pattern detected by the model."
        )

    },


    "Onychogryphosis": {

        "display_name":
        "Onychogryphosis",

        "description":
        (
            "A condition involving thickened and curved "
            "nail growth."
        ),

        "category":
        (
            "Nail thickening and abnormal growth pattern."
        )

    },


    "blue_finger": {

        "display_name":
        "Blue Finger",

        "description":
        (
            "A bluish discoloration pattern observed around "
            "the finger or nail region."
        ),

        "category":
        (
            "Color-related nail or finger appearance pattern."
        )

    },


    "clubbing": {

        "display_name":
        "Clubbing",

        "description":
        (
            "A condition involving changes in the shape and "
            "curvature of the fingers and nails."
        ),

        "category":
        (
            "Structural nail and finger shape pattern."
        )

    },


    "pitting": {

        "display_name":
        "Pitting",

        "description":
        (
            "A condition characterized by small depressions "
            "or pits on the nail surface."
        ),

        "category":
        (
            "Surface texture abnormality detected by the model."
        )

    }

}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource

def load_model():

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
# IMAGE PREDICTION
# ============================================================


# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================


# ============================================================
# IMAGE TRANSFORMATION FOR GRAD-CAM
# ============================================================


# ============================================================
# MODEL IMAGE TRANSFORM
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


def predict_image(

    model,

    image,

    class_names

):

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


    probabilities = (

        probabilities
        .cpu()
        .numpy()[0]
        * 100

    )


    return (

        predicted_class,

        confidence,

        probabilities

    )
    


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def get_confidence_level(

    confidence

):

    if confidence >= 90:

        return "High"


    elif confidence >= 70:

        return "Moderate"


    else:

        return "Low"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(

        "💅 Nail Condition AI"

    )


    st.caption(

        "Deep Learning Based Classification System"

    )


    st.divider()


    page = st.radio(

        "Navigation",

        [

            "Home",

            "Model Performance",

            "About Project"

        ]

    )


    st.divider()


    st.subheader(

        "System Information"

    )


    st.write(

        "**Model:** EfficientNet-B0"

    )


    st.write(

        "**Classes:** 6"

    )


    st.write(

        "**Input Size:** 224 x 224"

    )


    st.write(

        f"**Device:** {DEVICE.type.upper()}"

    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(

        """
        <div class="main-title">
        💅 Nail Condition Classification System
        </div>
        """,

        unsafe_allow_html=True

    )


    st.markdown(

        """
        <div class="sub-title">
        AI-powered nail image classification using EfficientNet-B0
        </div>
        """,

        unsafe_allow_html=True

    )


    st.divider()


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not MODEL_PATH.exists():

        st.error(

            "Model checkpoint not found."

        )


        st.code(

            str(MODEL_PATH)

        )


        st.stop()


    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(

        "Upload a nail image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if uploaded_file is not None:


        image = Image.open(

            uploaded_file

        ).convert(

            "RGB"

        )


        # ====================================================
        # IMAGE DISPLAY AND PREDICTION
        # ====================================================

        image_column, prediction_column = st.columns(

            [

                1,

                1

            ]

        )


        with image_column:


            st.subheader(

                "Uploaded Image"

            )


            st.image(

                image,

                caption="Uploaded Nail Image",

                width="stretch"

            )


        with prediction_column:


            st.subheader(

                "AI Prediction"

            )


            with st.spinner(

                "Analyzing nail image..."

            ):


                model, class_names = (

                    load_model()

                )


                predicted_class, confidence, probabilities = (

                    predict_image(

                        model,

                        image,

                        class_names

                    )

                )


                predicted_index = class_names.index(
                    predicted_class
                )


                try:

                    _, _, gradcam_image = generate_gradcam(
                        image,
                        model,
                        predicted_index,
                        224
                    )


                except Exception as error:

                    gradcam_image = None

                    st.warning(
                        f"Grad-CAM visualization could not be generated: {error}"
                    )


            display_name = (

                CONDITION_INFO[
                    predicted_class
                ][
                    "display_name"
                ]

            )


            confidence_level = (

                get_confidence_level(

                    confidence

                )

            )


            low_confidence_warning = (
                confidence < 70
            )


            st.success(

                "Analysis completed successfully."

            )


            st.markdown(

                f"""
                <div class="condition-name">
                {display_name}
                </div>
                """,

                unsafe_allow_html=True

            )


            st.metric(

                "Prediction Confidence",

                f"{confidence:.2f}%"

            )


            st.write(

                f"**Confidence Level:** "
                f"{confidence_level}"

            )
        # ====================================================

            if low_confidence_warning:

                st.warning(

                    "The model has relatively low confidence in this prediction. Please treat this result as an AI-generated estimate and not as a medical diagnosis."

                )



        # AI ATTENTION ANALYSIS - GRAD-CAM
        # ====================================================

        st.divider()

        st.subheader(
            "AI Attention Analysis (Grad-CAM)"
        )

        st.write(
            "The highlighted areas show which regions of the nail image influenced the AI prediction."
        )

        if gradcam_image is not None:

            gradcam_column1, gradcam_column2 = st.columns(2)

            with gradcam_column1:

                st.image(
                    image,
                    caption="Original Nail Image",
                    width="stretch"
                )

            with gradcam_column2:

                st.image(
                    gradcam_image,
                    caption="AI Attention Visualization",
                    width="stretch"
                )

            st.info(
                "Highlighted regions represent areas that contributed more strongly to the model prediction."
            )

        else:

            st.warning(
                "Grad-CAM visualization is unavailable for this image."
            )




        # ====================================================
        # CONDITION ANALYSIS
        # ====================================================

        st.divider()


        st.subheader(

            "Condition Analysis"

        )


        st.markdown(

            "#### Description"

        )


        st.write(

            CONDITION_INFO[
                predicted_class
            ][
                "description"
            ]

        )


        st.markdown(

            "#### Classification Category"

        )


        st.write(

            CONDITION_INFO[
                predicted_class
            ][
                "category"
            ]

        )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.divider()


        st.subheader(

            "Top 3 Model Predictions"

        )


        sorted_indices = (

            probabilities
            .argsort()
            [::-1]

        )


        top_3_indices = (

            sorted_indices[:3]

        )


        top_3_data = []


        for index in top_3_indices:


            class_name = (

                class_names[
                    index
                ]

            )


            readable_name = (

                CONDITION_INFO[
                    class_name
                ][
                    "display_name"
                ]

            )


            top_3_data.append({

                "Condition":
                readable_name,

                "Probability (%)":
                round(

                    float(
                        probabilities[index]
                    ),

                    2

                )

            })


        top_3_df = pd.DataFrame(

            top_3_data

        )


        st.dataframe(

            top_3_df,

            width="stretch",

            hide_index=True

        )


        # ====================================================
        # ALL CLASS PROBABILITIES
        # ====================================================

        st.divider()


        st.subheader(

            "All Class Probabilities"

        )


        probability_data = []


        report_probabilities = {}


        for index, class_name in enumerate(

            class_names

        ):


            readable_name = (

                CONDITION_INFO[
                    class_name
                ][
                    "display_name"
                ]

            )


            probability_value = (

                float(
                    probabilities[index]
                )

            )


            probability_data.append({

                "Condition":
                readable_name,

                "Probability (%)":
                probability_value

            })


            report_probabilities[
                readable_name
            ] = probability_value


        probability_df = pd.DataFrame(

            probability_data

        )


        probability_df = (

            probability_df
            .sort_values(

                by="Probability (%)",

                ascending=False

            )

        )


        st.bar_chart(

            probability_df,

            x="Condition",

            y="Probability (%)"

        )


        with st.expander(

            "View Detailed Probabilities"

        ):


            st.dataframe(

                probability_df,

                width="stretch",

                hide_index=True

            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        st.divider()


        st.subheader(

            "Download Prediction Report"

        )


        if PDF_REPORT_AVAILABLE:


            try:


                # --------------------------------------------
                # CONDITION INFORMATION FOR PDF
                # --------------------------------------------

                report_condition_info = (

                    "Description: "

                    + CONDITION_INFO[
                        predicted_class
                    ][
                        "description"
                    ]

                    + "\n\nClassification Category: "

                    + CONDITION_INFO[
                        predicted_class
                    ][
                        "category"
                    ]

                )


                # --------------------------------------------
                # GENERATE PDF
                # --------------------------------------------

                report_bytes = (

                    generate_prediction_report(

                        image_name=(
                            uploaded_file.name
                        ),

                        predicted_class=(
                            display_name
                        ),

                        confidence=(
                            confidence
                        ),

                        probabilities=(
                            report_probabilities
                        ),

                        condition_info=(
                            report_condition_info
                        )

                    )

                )


                # --------------------------------------------
                # DOWNLOAD BUTTON
                # --------------------------------------------

                st.download_button(

                    label=(
                        "📄 Download PDF Report"
                    ),

                    data=(
                        report_bytes
                    ),

                    file_name=(
                        "nail_condition_prediction_report.pdf"
                    ),

                    mime=(
                        "application/pdf"
                    ),

                    width="stretch"

                )


            except Exception as error:


                st.error(

                    "PDF report could not be generated."

                )


                st.code(

                    str(error)

                )


        else:


            st.warning(

                "PDF report generator could not be loaded."

            )


            st.code(

                PDF_ERROR

            )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":


    st.title(

        "📊 EfficientNet-B0 Model Performance"

    )


    st.write(

        "Performance evaluation and training results "
        "of the EfficientNet-B0 model."

    )


    # ========================================================
    # LOAD CLASS-WISE METRICS
    # ========================================================

    metrics_path = (

        METRICS_DIR
        / "efficientnet_b0_classwise_metrics.csv"

    )


    if metrics_path.exists():


        metrics_df = pd.read_csv(

            metrics_path

        )


        col1, col2, col3, col4 = st.columns(

            4

        )


        weighted_precision = (

            metrics_df[
                "precision"
            ].mean()
            * 100

        )


        weighted_recall = (

            metrics_df[
                "recall"
            ].mean()
            * 100

        )


        weighted_f1 = (

            metrics_df[
                "f1_score"
            ].mean()
            * 100

        )


        col1.metric(

            "Accuracy",

            "100.00%"

        )


        col2.metric(

            "Average Precision",

            f"{weighted_precision:.2f}%"

        )


        col3.metric(

            "Average Recall",

            f"{weighted_recall:.2f}%"

        )


        col4.metric(

            "Average F1 Score",

            f"{weighted_f1:.2f}%"

        )


    else:


        st.warning(

            "Metrics file not found."

        )


    # ========================================================
    # TRAINING PERFORMANCE
    # ========================================================

    st.divider()


    st.subheader(

        "📈 Training Performance"

    )


    loss_plot = (

        VISUALIZATIONS_DIR
        / "efficientnet_b0_loss_curve.png"

    )


    accuracy_plot = (

        VISUALIZATIONS_DIR
        / "efficientnet_b0_accuracy_curve.png"

    )


    col1, col2 = st.columns(

        2

    )


    with col1:


        if loss_plot.exists():


            st.image(

                str(loss_plot),

                caption=(
                    "Training and Validation Loss"
                ),

                width="stretch"

            )


        else:


            st.warning(

                "Loss visualization not found."

            )


    with col2:


        if accuracy_plot.exists():


            st.image(

                str(accuracy_plot),

                caption=(
                    "Training and Validation Accuracy"
                ),

                width="stretch"

            )


        else:


            st.warning(

                "Accuracy visualization not found."

            )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.divider()


    st.subheader(

        "🔢 Confusion Matrix"

    )


    confusion_plot = (

        VISUALIZATIONS_DIR
        / "efficientnet_b0_confusion_matrix.png"

    )


    if confusion_plot.exists():


        st.image(

            str(confusion_plot),

            width="stretch"

        )


    else:


        st.warning(

            "Confusion matrix visualization not found."

        )


    # ========================================================
    # CLASS-WISE PERFORMANCE
    # ========================================================

    st.divider()


    st.subheader(

        "📊 Class-wise Performance"

    )


    classwise_plot = (

        VISUALIZATIONS_DIR
        / "efficientnet_b0_combined_classwise_metrics.png"

    )


    if classwise_plot.exists():


        st.image(

            str(classwise_plot),

            width="stretch"

        )


    else:


        st.warning(

            "Class-wise performance visualization not found."

        )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":


    st.title(

        "ℹ️ About the Project"

    )


    st.subheader(

        "Project Overview"

    )


    st.write(

        "This project uses deep learning and computer vision "
        "to classify images of nails into six different "
        "classification categories."

    )


    st.subheader(

        "Deep Learning Architecture"

    )


    st.write(

        "EfficientNet-B0"

    )


    st.subheader(

        "Classification Categories"

    )


    categories = [

        "Acral Lentiginous Melanoma",

        "Healthy Nail",

        "Onychogryphosis",

        "Blue Finger",

        "Clubbing",

        "Pitting"

    ]


    for number, category in enumerate(

        categories,

        start=1

    ):


        st.write(

            f"{number}. {category}"

        )


    st.subheader(

        "Image Processing"

    )


    st.write(

        "Images are resized to 224 x 224 pixels and normalized "
        "before being passed to the EfficientNet-B0 deep learning model."

    )


    st.subheader(

        "Hardware"

    )


    if DEVICE.type == "cuda":


        gpu_name = (

            torch.cuda.get_device_name(
                0
            )

        )


        st.write(

            f"GPU Acceleration Enabled: {gpu_name}"

        )


    else:


        st.write(

            "The application is currently running on CPU."

        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(

    """
    <div class="footer">

    <hr>

    <h4>
    💅 Nail Condition Classification Project | EfficientNet-B0
    </h4>

    <p>
    Educational and research use only. This system is not a medical
    diagnostic tool and should not replace professional medical advice.
    </p>

    </div>
    """,

    unsafe_allow_html=True

)





# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================

