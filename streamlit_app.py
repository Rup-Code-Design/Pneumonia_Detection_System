# ============================================================
# streamlit_app.py
#
# PNEUMONIA DETECTION SYSTEM FROM X-RAY IMAGES
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Basic Validation
#       ↓
# 3-Class Modality Classifier
#       ↓
# ┌──────────────┬──────────────┬──────────────┐
# │ CT           │ MRI          │ X-ray        │
# │ STOP         │ STOP         │ CONTINUE     │
# └──────────────┴──────────────┴──────────────┘
#                                      ↓
#                              X-ray Verifier
#                                      ↓
#                              Pneumonia Model
#                                      ↓
#                              Normal/Pneumonia
#                                      ↓
#                                  PDF Report
#
#
# MODALITY MAPPING
#
# 0 = CT
# 1 = MRI
# 2 = X-ray
#
#
# PNEUMONIA MAPPING
#
# 0 = Normal
# 1 = Pneumonia
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import io
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATHS
# ============================================================

MODALITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "modality_classifier.keras"
)

XRAY_VERIFIER_PATH = os.path.join(
    BASE_DIR,
    "xray_verifier.keras"
)

PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# ICON
# ============================================================

ICON_PATH = os.path.join(
    BASE_DIR,
    "lung_xray_icon.png"
)


# ============================================================
# IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (224, 224)

XRAY_VERIFIER_IMAGE_SIZE = (224, 224)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# THRESHOLDS
# ============================================================

XRAY_VERIFIER_THRESHOLD = 0.50


# ============================================================
# MODALITY CLASS MAPPING
# ============================================================

MODALITY_CLASS_MAP = {
    0: "CT",
    1: "MRI",
    2: "X-ray"
}


MODALITY_XRAY_CLASS_INDEX = 2


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================

PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


if "pdf_report" not in st.session_state:
    st.session_state.pdf_report = None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

title_col1, title_col2 = st.columns(
    [1, 7],
    vertical_alignment="center"
)


with title_col1:

    if os.path.isfile(ICON_PATH):

        st.image(
            ICON_PATH,
            width=110
        )

    else:

        st.markdown(
            "<div style='font-size:70px;text-align:center;'>"
            "🫁"
            "</div>",
            unsafe_allow_html=True
        )


with title_col2:

    st.markdown(
        """
        <div class="main-title">
            PNEUMONIA DETECTION SYSTEM<br>
            FROM X-RAY IMAGES
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="subtitle">
        Automated medical image modality verification and
        pneumonia detection
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model_file(
    path,
    model_name
):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"""
{model_name} was not found.

Expected location:
{path}

Make sure the model file is committed to the
same GitHub repository as streamlit_app.py.
"""
        )


    if os.path.getsize(path) == 0:

        raise ValueError(
            f"{model_name} exists but is empty:\n{path}"
        )


# ============================================================
# LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    validate_model_file(
        MODALITY_MODEL_PATH,
        "Modality classifier"
    )

    try:

        model = tf.keras.models.load_model(
            MODALITY_MODEL_PATH,
            compile=False
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load modality_classifier.keras.\n\n"
            f"Original error:\n{e}"
        ) from e


    if model.output_shape[-1] != 3:

        raise ValueError(
            "The modality classifier must have exactly "
            "3 output classes.\n"
            f"Received: {model.output_shape}"
        )


    if model.input_shape[-1] != 3:

        raise ValueError(
            "The modality classifier must accept "
            "3-channel RGB input.\n"
            f"Received: {model.input_shape}"
        )


    return model


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_verifier():

    validate_model_file(
        XRAY_VERIFIER_PATH,
        "X-ray verifier"
    )

    try:

        model = tf.keras.models.load_model(
            XRAY_VERIFIER_PATH,
            compile=False
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load xray_verifier.keras.\n\n"
            f"Original error:\n{e}"
        ) from e


    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    validate_model_file(
        PNEUMONIA_MODEL_PATH,
        "Pneumonia model"
    )

    try:

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load best_xception_pneumonia_model.keras."
            "\n\n"
            f"Original error:\n{e}"
        ) from e


    return model


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    modality_model = load_modality_model()

    xray_verifier_model = load_xray_verifier()

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# PNEUMONIA OUTPUT CHECK
# ============================================================

if pneumonia_model.output_shape[-1] != 2:

    st.error(
        "The pneumonia model must have exactly "
        "2 output classes."
    )

    st.write(
        f"Received output shape: "
        f"{pneumonia_model.output_shape}"
    )

    st.stop()


# ============================================================
# PROBABILITY CONVERSION
# ============================================================

def convert_to_probabilities(scores):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Already probabilities
    # --------------------------------------------------------

    if (
        np.all(scores >= 0.0)
        and
        np.all(scores <= 1.0)
        and
        np.isclose(
            np.sum(scores),
            1.0,
            atol=1e-3
        )
    ):

        return scores


    # --------------------------------------------------------
    # Otherwise logits
    # --------------------------------------------------------

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# IMAGE COLOR CHECK
# ============================================================

def check_color_image(image):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )


    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]


    rg_difference = np.mean(
        np.abs(red - green)
    )


    gb_difference = np.mean(
        np.abs(green - blue)
    )


    rb_difference = np.mean(
        np.abs(red - blue)
    )


    average_difference = (
        rg_difference
        +
        gb_difference
        +
        rb_difference
    ) / 3.0


    return (
        average_difference > 8.0,
        float(average_difference)
    )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(image):

    width, height = image.size


    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )


    array = np.asarray(image)


    if array.size == 0:

        return (
            False,
            "Image is empty."
        )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT reject grayscale medical images.
    # CT, MRI and X-ray can all be grayscale.
    #
    # We only reject strongly colored images.
    # --------------------------------------------------------

    is_color, color_difference = (
        check_color_image(image)
    )


    if is_color:

        return (
            False,
            "Color image detected. "
            "Please upload a grayscale medical image."
        )


    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )


    if np.std(gray) < 8:

        return (
            False,
            "Image appears blank or invalid."
        )


    dark_ratio = np.mean(
        gray < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "Image is almost completely black."
        )


    bright_ratio = np.mean(
        gray > 245
    )


    if bright_ratio > 0.98:

        return (
            False,
            "Image is almost completely white."
        )


    return (
        True,
        "Image passed validation."
    )


# ============================================================
# ============================================================
# MODALITY PREPROCESSING
# ============================================================
#
# THIS IS THE IMPORTANT FIX.
#
# Your previous code used:
#
# Image.Resampling.NEAREST
#
# That does NOT reproduce the usual Keras
# flow_from_directory preprocessing.
#
# The default PIL interpolation used by the Keras
# directory loader is effectively BILINEAR for this path.
#
# Therefore use BILINEAR here.
#
# Training:
#
#     color_mode="rgb"
#     target_size=(224,224)
#     rescale=1/255
#
# Inference:
#
#     RGB
#     224x224
#     BILINEAR
#     /255
#
# ============================================================

def preprocess_modality_image(
    image,
    interpolation=Image.Resampling.BILINEAR
):

    # --------------------------------------------------------
    # EXIF orientation
    # --------------------------------------------------------

    image = ImageOps.exif_transpose(
        image
    )


    # --------------------------------------------------------
    # RGB
    #
    # Exactly 3 channels.
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # RESIZE
    #
    # IMPORTANT:
    #
    # BILINEAR instead of NEAREST.
    # --------------------------------------------------------

    image = image.resize(
        MODALITY_IMAGE_SIZE,
        interpolation
    )


    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SAME NORMALIZATION AS TRAINING
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )


    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# GET MODALITY PREDICTION
# ============================================================

def get_modality_prediction(
    image,
    interpolation
):

    image_array = preprocess_modality_image(
        image,
        interpolation
    )


    raw_prediction = modality_model.predict(
        image_array,
        verbose=0
    )


    raw_prediction = np.asarray(
        raw_prediction
    )


    if (
        raw_prediction.ndim != 2
        or
        raw_prediction.shape[1] != 3
    ):

        raise ValueError(
            "Modality classifier must output "
            "3 classes.\n"
            f"Received: {raw_prediction.shape}"
        )


    probabilities = (
        convert_to_probabilities(
            raw_prediction[0]
        )
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    return {
        "index": predicted_index,
        "class": MODALITY_CLASS_MAP[
            predicted_index
        ],
        "confidence": float(
            probabilities[predicted_index]
        ),
        "probabilities": probabilities,
        "raw_output": raw_prediction[0]
    }


# ============================================================
# MODALITY PREDICTION
# ============================================================
#
# PRIMARY:
#   BILINEAR
#
# FALLBACK:
#   BICUBIC
#   LANCZOS
#   NEAREST
#
# The primary result is always preferred when it produces
# a clear prediction.
#
# The fallback is especially useful for images whose
# interpolation is sensitive.
# ============================================================

def predict_modality(image):

    # --------------------------------------------------------
    # PRIMARY PREPROCESSING
    # --------------------------------------------------------

    primary = get_modality_prediction(
        image,
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # If primary is already highly confident, use it.
    #
    # This preserves the normal trained inference pipeline.
    # --------------------------------------------------------

    if primary["confidence"] >= 0.80:

        primary["preprocessing"] = "BILINEAR"

        primary["fallback_results"] = None

        return primary


    # --------------------------------------------------------
    # FALLBACK TESTS
    # --------------------------------------------------------

    preprocessing_methods = [
        (
            "BICUBIC",
            Image.Resampling.BICUBIC
        ),

        (
            "LANCZOS",
            Image.Resampling.LANCZOS
        ),

        (
            "NEAREST",
            Image.Resampling.NEAREST
        )
    ]


    candidates = [
        primary
    ]


    for name, interpolation in preprocessing_methods:

        result = get_modality_prediction(
            image,
            interpolation
        )

        result["preprocessing"] = name

        candidates.append(
            result
        )


    # --------------------------------------------------------
    # Choose strongest confidence.
    #
    # This is only reached when the primary model is
    # uncertain (<80%).
    # --------------------------------------------------------

    best_result = max(
        candidates,
        key=lambda x: x["confidence"]
    )


    best_result["fallback_results"] = candidates


    return best_result


# ============================================================
# GENERAL PREPROCESSING
#
# USED BY:
#   X-RAY VERIFIER
#   PNEUMONIA MODEL
#
# DO NOT CHANGE.
# ============================================================

def preprocess_image(
    image,
    target_size
):

    image = ImageOps.exif_transpose(
        image
    )


    image = image.convert(
        "RGB"
    )


    image = image.resize(
        target_size,
        Image.Resampling.LANCZOS
    )


    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    image_array /= 255.0


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def predict_xray_verification(image):

    image_array = preprocess_image(
        image,
        XRAY_VERIFIER_IMAGE_SIZE
    )


    prediction = (
        xray_verifier_model.predict(
            image_array,
            verbose=0
        )
    )


    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # SIGMOID
    #
    # 0 = NON-X-RAY
    # 1 = X-RAY
    # --------------------------------------------------------

    if prediction.shape[-1] == 1:

        probability = float(
            prediction[0][0]
        )


        if (
            probability < 0.0
            or
            probability > 1.0
        ):

            probability = float(
                tf.sigmoid(
                    prediction[0][0]
                ).numpy()
            )


        is_xray = (
            probability
            >= XRAY_VERIFIER_THRESHOLD
        )


        confidence = (
            probability
            if is_xray
            else
            1.0 - probability
        )


        return {
            "is_xray": is_xray,
            "confidence": float(confidence),
            "xray_probability": probability,
            "non_xray_probability": (
                1.0 - probability
            )
        }


    # --------------------------------------------------------
    # TWO CLASS SOFTMAX
    #
    # 0 = NON-X-RAY
    # 1 = X-RAY
    # --------------------------------------------------------

    if prediction.shape[-1] == 2:

        probabilities = (
            convert_to_probabilities(
                prediction[0]
            )
        )


        non_xray_probability = float(
            probabilities[0]
        )


        xray_probability = float(
            probabilities[1]
        )


        is_xray = (
            xray_probability
            >= XRAY_VERIFIER_THRESHOLD
        )


        confidence = (
            xray_probability
            if is_xray
            else
            non_xray_probability
        )


        return {
            "is_xray": is_xray,
            "confidence": float(confidence),
            "xray_probability": xray_probability,
            "non_xray_probability": (
                non_xray_probability
            )
        }


    raise ValueError(
        "Unexpected X-ray verifier output shape: "
        f"{prediction.shape}"
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(image):

    image_array = preprocess_image(
        image,
        PNEUMONIA_IMAGE_SIZE
    )


    prediction = (
        pneumonia_model.predict(
            image_array,
            verbose=0
        )
    )


    prediction = np.asarray(
        prediction
    )


    if (
        prediction.ndim != 2
        or
        prediction.shape[1] != 2
    ):

        raise ValueError(
            "Pneumonia model must output "
            "2 classes.\n"
            f"Received: {prediction.shape}"
        )


    probabilities = (
        convert_to_probabilities(
            prediction[0]
        )
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_class = (
        PNEUMONIA_CLASS_MAP[
            predicted_index
        ]
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    return {
        "class": predicted_class,
        "confidence": confidence,

        "normal_probability": float(
            probabilities[0]
        ),

        "pneumonia_probability": float(
            probabilities[1]
        ),

        "probabilities": probabilities
    }


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    modality_result,
    verifier_result,
    pneumonia_result=None
):

    buffer = io.BytesIO()


    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12
    )


    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=14,
        spaceAfter=15
    )


    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )


    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )


    story = []


    story.append(
        Paragraph(
            "PNEUMONIA DETECTION SYSTEM<br/>"
            "FROM X-RAY IMAGES",
            title_style
        )
    )


    story.append(
        Paragraph(
            "Medical Image Analysis Report",
            subtitle_style
        )
    )


    report_time = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )


    story.append(
        Paragraph(
            f"<b>Analysis Date:</b> {report_time}",
            normal_style
        )
    )


    story.append(
        Spacer(
            1,
            10
        )
    )


    image_buffer = io.BytesIO()


    image.save(
        image_buffer,
        format="PNG"
    )


    image_buffer.seek(0)


    report_image = RLImage(
        image_buffer,
        width=100 * mm,
        height=100 * mm
    )


    story.append(
        report_image
    )


    story.append(
        Spacer(
            1,
            12
        )
    )


    # ========================================================
    # MODALITY
    # ========================================================

    modality = modality_result["class"]


    modality_confidence = (
        modality_result["confidence"]
        * 100
    )


    modality_data = [
        ["Parameter", "Result"],

        [
            "Detected Modality",
            modality
        ],

        [
            "Modality Confidence",
            f"{modality_confidence:.2f}%"
        ]
    ]


    story.append(
        Paragraph(
            "1. Medical Image Modality",
            heading_style
        )
    )


    modality_table = Table(
        modality_data,
        colWidths=[
            70 * mm,
            80 * mm
        ]
    )


    modality_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )


    story.append(
        modality_table
    )


    # ========================================================
    # X-RAY VERIFICATION
    # ========================================================

    if verifier_result is not None:

        xray_probability = (
            verifier_result[
                "xray_probability"
            ]
            * 100
        )


        verifier_data = [
            ["Parameter", "Result"],

            [
                "X-ray Verification",
                (
                    "X-ray"
                    if verifier_result["is_xray"]
                    else "Not X-ray"
                )
            ],

            [
                "X-ray Probability",
                f"{xray_probability:.2f}%"
            ]
        ]


        story.append(
            Paragraph(
                "2. X-ray Verification",
                heading_style
            )
        )


        verifier_table = Table(
            verifier_data,
            colWidths=[
                70 * mm,
                80 * mm
            ]
        )


        verifier_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),

                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    )
                ]
            )
        )


        story.append(
            verifier_table
        )


    # ========================================================
    # PNEUMONIA
    # ========================================================

    if pneumonia_result is not None:

        diagnosis = (
            pneumonia_result["class"]
        )


        confidence = (
            pneumonia_result["confidence"]
            * 100
        )


        normal_probability = (
            pneumonia_result[
                "normal_probability"
            ]
            * 100
        )


        pneumonia_probability = (
            pneumonia_result[
                "pneumonia_probability"
            ]
            * 100
        )


        pneumonia_data = [
            ["Parameter", "Result"],

            [
                "Final Diagnosis",
                diagnosis
            ],

            [
                "Diagnosis Confidence",
                f"{confidence:.2f}%"
            ],

            [
                "Normal Probability",
                f"{normal_probability:.2f}%"
            ],

            [
                "Pneumonia Probability",
                f"{pneumonia_probability:.2f}%"
            ]
        ]


        story.append(
            Paragraph(
                "3. Pneumonia Detection",
                heading_style
            )
        )


        pneumonia_table = Table(
            pneumonia_data,
            colWidths=[
                40 * mm,
                60 * mm
            ]
        )


        pneumonia_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),

                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    )
                ]
            )
        )


        story.append(
            pneumonia_table
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Spacer(
            1,
            15
        )
    )


    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This application is a "
            "research prototype and is not intended to "
            "provide clinical diagnosis or replace "
            "professional medical evaluation.",
            normal_style
        )
    )


    document.build(
        story
    )


    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Pneumonia Detection System"
    )


    st.write(
        """
        **Pipeline**

        1. Upload image
        2. Basic validation
        3. Modality classification
        4. X-ray verification
        5. Pneumonia detection
        6. PDF report
        """
    )


    st.divider()


    st.write(
        "**Modality Classes**"
    )

    st.write("0 — CT")

    st.write("1 — MRI")

    st.write("2 — X-ray")


    st.divider()


    st.write(
        "**Pneumonia Classes**"
    )

    st.write("0 — Normal")

    st.write("1 — Pneumonia")


    st.divider()


    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# UPLOAD
# ============================================================

st.subheader(
    "Upload Medical Image"
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tif",
        "tiff"
    ],
    help=(
        "Upload a grayscale CT, MRI or X-ray image."
    )
)


# ============================================================
# PROCESS UPLOAD
# ============================================================

if uploaded_file is not None:

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )


        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )


        image = ImageOps.exif_transpose(
            image
        )


        image.load()


    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    st.subheader(
        "Uploaded Image"
    )


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    with st.expander(
        "Image information"
    ):

        st.write(
            f"Original mode: **{image.mode}**"
        )

        st.write(
            f"Original size: **{image.size[0]} × "
            f"{image.size[1]}**"
        )


    # ========================================================
    # ANALYZE
    # ========================================================

    analyze = st.button(
        "Verify Your Image",
        type="primary",
        use_container_width=True
    )


    if analyze:

        # ====================================================
        # STEP 1
        # ====================================================

        with st.spinner(
            "Validating image..."
        ):

            is_valid, validation_message = (
                validate_image(
                    image
                )
            )


        if not is_valid:

            st.error(
                validation_message
            )

            st.session_state.analysis_result = None

            st.session_state.pdf_report = None

            st.stop()


        st.success(
            "Image passed basic validation."
        )


        # ====================================================
        # STEP 2
        #
        # MODALITY
        # ====================================================

        with st.spinner(
            "Detecting CT / MRI / X-ray..."
        ):

            try:

                modality_result = (
                    predict_modality(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "Modality classification failed."
                )

                st.exception(e)

                st.stop()


        modality = (
            modality_result["class"]
        )


        modality_confidence = (
            modality_result["confidence"]
            * 100
        )


        # ====================================================
        # MODALITY RESULT
        # ====================================================

        st.markdown(
            "## Detected Image Type"
        )


        st.markdown(
            f"### {modality}"
        )


        st.write(
            f"Confidence: "
            f"**{modality_confidence:.4f}%**"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown(
            "### Modality Probabilities"
        )


        probabilities = (
            modality_result[
                "probabilities"
            ]
        )


        probability_columns = st.columns(3)


        for index, column in enumerate(
            probability_columns
        ):

            label = (
                MODALITY_CLASS_MAP[
                    index
                ]
            )


            probability = (
                float(
                    probabilities[index]
                )
                * 100
            )


            with column:

                st.metric(
                    label,
                    f"{probability:.4f}%"
                )


                st.progress(
                    min(
                        max(
                            probability / 100.0,
                            0.0
                        ),
                        1.0
                    )
                )


        # ====================================================
        # TECHNICAL INFORMATION
        # ====================================================

        with st.expander(
            "Technical modality information"
        ):

            st.write(
                f"Predicted class index: "
                f"**{modality_result['index']}**"
            )


            st.write(
                "Class mapping:"
            )


            st.code(
                """
0 = CT
1 = MRI
2 = X-ray
"""
            )


            st.write(
                f"Input size: "
                f"**{MODALITY_IMAGE_SIZE[0]} × "
                f"{MODALITY_IMAGE_SIZE[1]}**"
            )


            st.write(
                "Input channels: **RGB / 3 channels**"
            )


            st.write(
                "Normalization: **pixel / 255.0**"
            )


            st.write(
                "Primary interpolation: **BILINEAR**"
            )


            st.write(
                "Selected preprocessing: "
                f"**{modality_result.get('preprocessing', 'BILINEAR')}**"
            )


            st.write(
                "Raw model output:"
            )


            st.write(
                modality_result["raw_output"]
            )


        # ====================================================
        # FALLBACK INFORMATION
        # ====================================================

        if (
            modality_result.get(
                "fallback_results"
            )
            is not None
        ):

            with st.expander(
                "Modality preprocessing comparison"
            ):

                st.write(
                    "The primary BILINEAR preprocessing "
                    "was not sufficiently confident, so "
                    "additional interpolation methods were "
                    "tested."
                )


                for result in (
                    modality_result[
                        "fallback_results"
                    ]
                ):

                    st.write(
                        f"**{result['preprocessing']}** → "
                        f"{result['class']} — "
                        f"{result['confidence'] * 100:.4f}%"
                    )


        # ====================================================
        # STEP 3
        #
        # CT / MRI STOP
        # ====================================================

        if modality in (
            "CT",
            "MRI"
        ):

            st.warning(
                f"This image was identified as "
                f"**{modality}**."
            )


            st.info(
                "Pneumonia detection is available "
                "only for X-ray images. "
                "Analysis stopped."
            )


            st.session_state.analysis_result = {
                "modality": modality,
                "modality_confidence": (
                    modality_confidence
                ),
                "verifier": None,
                "pneumonia": None
            }


            st.session_state.pdf_report = (
                create_pdf_report(
                    image,
                    modality_result,
                    None,
                    None
                )
            )


            st.download_button(
                label=(
                    "Download Modality Report (PDF)"
                ),
                data=(
                    st.session_state.pdf_report
                ),
                file_name=(
                    "medical_image_modality_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )


            st.stop()


        # ====================================================
        # STEP 4
        #
        # X-RAY VERIFICATION
        # ====================================================

        if modality == "X-ray":

            st.markdown(
                "## X-ray Verification"
            )


            with st.spinner(
                "Verifying X-ray image..."
            ):

                try:

                    verifier_result = (
                        predict_xray_verification(
                            image
                        )
                    )

                except Exception as e:

                    st.error(
                        "X-ray verification failed."
                    )

                    st.exception(e)

                    st.stop()


            xray_probability = (
                verifier_result[
                    "xray_probability"
                ]
                * 100
            )


            st.write(
                f"X-ray probability: "
                f"**{xray_probability:.4f}%**"
            )


            st.progress(
                min(
                    max(
                        xray_probability / 100.0,
                        0.0
                    ),
                    1.0
                )
            )


            # =================================================
            # NOT X-RAY
            # =================================================

            if not verifier_result[
                "is_xray"
            ]:

                st.error(
                    "The X-ray verifier did not "
                    "confirm this image as an X-ray."
                )


                st.info(
                    "Pneumonia detection has been stopped."
                )


                st.session_state.analysis_result = {
                    "modality": modality,
                    "modality_confidence": (
                        modality_confidence
                    ),
                    "verifier": verifier_result,
                    "pneumonia": None
                }


                st.session_state.pdf_report = (
                    create_pdf_report(
                        image,
                        modality_result,
                        verifier_result,
                        None
                    )
                )


                st.download_button(
                    label=(
                        "Download Verification Report (PDF)"
                    ),
                    data=(
                        st.session_state.pdf_report
                    ),
                    file_name=(
                        "xray_verification_report.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True
                )


                st.stop()


            # =================================================
            # STEP 5
            #
            # PNEUMONIA
            # =================================================

            st.success(
                "X-ray verified successfully."
            )


            st.markdown(
                "## Pneumonia Detection"
            )


            with st.spinner(
                "Analyzing X-ray for pneumonia..."
            ):

                try:

                    pneumonia_result = (
                        predict_pneumonia(
                            image
                        )
                    )

                except Exception as e:

                    st.error(
                        "Pneumonia prediction failed."
                    )

                    st.exception(e)

                    st.stop()


            diagnosis = (
                pneumonia_result["class"]
            )


            diagnosis_confidence = (
                pneumonia_result["confidence"]
                * 100
            )


            # =================================================
            # FINAL RESULT
            # =================================================

            if diagnosis == "Pneumonia":

                st.error(
                    f"### Result: {diagnosis}"
                )

            else:

                st.success(
                    f"### Result: {diagnosis}"
                )


            st.write(
                f"Confidence: "
                f"**{diagnosis_confidence:.4f}%**"
            )


            # =================================================
            # PNEUMONIA PROBABILITIES
            # =================================================

            with st.expander(
                "View pneumonia probabilities"
            ):

                normal_probability = (
                    pneumonia_result[
                        "normal_probability"
                    ]
                    * 100
                )


                pneumonia_probability = (
                    pneumonia_result[
                        "pneumonia_probability"
                    ]
                    * 100
                )


                st.write(
                    f"**Normal: "
                    f"{normal_probability:.4f}%**"
                )


                st.progress(
                    min(
                        max(
                            normal_probability / 100.0,
                            0.0
                        ),
                        1.0
                    )
                )


                st.write(
                    f"**Pneumonia: "
                    f"{pneumonia_probability:.4f}%**"
                )


                st.progress(
                    min(
                        max(
                            pneumonia_probability / 100.0,
                            0.0
                        ),
                        1.0
                    )
                )


            # =================================================
            # PDF
            # =================================================

            pdf_bytes = create_pdf_report(
                image,
                modality_result,
                verifier_result,
                pneumonia_result
            )


            st.session_state.analysis_result = {
                "modality": modality,
                "modality_confidence": (
                    modality_confidence
                ),
                "verifier": verifier_result,
                "pneumonia": pneumonia_result
            }


            st.session_state.pdf_report = (
                pdf_bytes
            )


            st.divider()


            st.subheader(
                "Final Report"
            )


            st.download_button(
                label=(
                    "Download Final Report (PDF)"
                ),
                data=pdf_bytes,
                file_name=(
                    "pneumonia_detection_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )
