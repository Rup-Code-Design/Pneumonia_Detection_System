# ============================================================
# streamlit_app.py
#
# PneuX-ModNet
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
#       ↓
# X-ray Verifier
#       ↓
# Pneumonia Model
#       ↓
# Normal/Pneumonia
#       ↓
# IF PNEUMONIA
#       ↓
# Grad-CAM++ Localization
#       ↓
# PDF Report
#
#
# MODALITY MAPPING:
#
# 0 = CT
# 1 = MRI
# 2 = X-ray
#
# PNEUMONIA MAPPING:
#
# 0 = Normal
# 1 = Pneumonia
#
#
# IMPORTANT:
#
# Grad-CAM++ is NOT a separate trained model.
# It is calculated from the existing pneumonia CNN.
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
    page_title="PneuX-ModNet",
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
# ICON PATH
# ============================================================

ICON_PATH = os.path.join(
    BASE_DIR,
    "lung_xray_icon.png"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

MODALITY_IMAGE_SIZE = (
    224,
    224
)

XRAY_VERIFIER_IMAGE_SIZE = (
    224,
    224
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# GRAD-CAM++ SETTINGS
# ============================================================

GRADCAM_IMAGE_SIZE = (
    224,
    224
)

GRADCAM_OVERLAY_ALPHA = 0.45


# ============================================================
# THRESHOLDS
# ============================================================

COLOR_TOLERANCE = 8.0

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
# HTML / CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.15;
        margin-top: 5px;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        line-height: 1.5;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    .section-title {
        text-align: center;
        font-size: 26px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .modality-result {
        background-color: #eef4ff;
    }

    .normal-result {
        background-color: #eaf7ea;
    }

    .pneumonia-result {
        background-color: #fdeaea;
    }

    .info-text {
        text-align: center;
        font-size: 15px;
        line-height: 1.5;
    }

    .brand-name {
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 2px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE HEADER
# ============================================================

title_col1, title_col2 = st.columns(
    [0.75, 7.25],
    vertical_alignment="center"
)


# ============================================================
# HEADER ICON
# ============================================================

with title_col1:

    if os.path.isfile(ICON_PATH):

        st.image(
            ICON_PATH,
            width=90
        )

    else:

        st.markdown(
            """
            <div style="
                font-size:70px;
                text-align:center;
                line-height:1;
            ">
                🫁
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HEADER TEXT
# ============================================================

with title_col2:

    st.markdown(
        """
        <div class="main-title">
            PneuX-ModNet
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="brand-name">
            AI-Based Medical Image Modality Classification
            and Pneumonia Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SUBTITLE
# ============================================================

st.markdown(
    """
    <div class="subtitle">
        Automated classification of CT, MRI, and X-ray images,
        followed by X-ray verification and pneumonia detection.
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

Make sure the file is committed to the
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
            "The modality classifier must have "
            "exactly 3 output classes.\n"
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
            "Could not load "
            "best_xception_pneumonia_model.keras.\n\n"
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

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CHECK PNEUMONIA MODEL OUTPUT
# ============================================================

if pneumonia_model.output_shape[-1] != 2:

    st.error(
        "The pneumonia model must have "
        "2 output classes."
    )

    st.write(
        f"Received output shape: "
        f"{pneumonia_model.output_shape}"
    )

    st.stop()


# ============================================================
# CONVERT OUTPUT TO PROBABILITIES
# ============================================================

def convert_to_probabilities(
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

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

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# COLOR IMAGE CHECK
# ============================================================

def check_color_image(
    image
):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    rg_difference = np.mean(
        np.abs(
            red - green
        )
    )

    gb_difference = np.mean(
        np.abs(
            green - blue
        )
    )

    rb_difference = np.mean(
        np.abs(
            red - blue
        )
    )

    average_difference = (
        rg_difference
        +
        gb_difference
        +
        rb_difference
    ) / 3.0

    is_color = (
        average_difference
        >
        COLOR_TOLERANCE
    )

    return (
        is_color,
        float(average_difference)
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(
    image
):

    width, height = image.size

    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )

    array = np.asarray(
        image
    )

    if array.size == 0:

        return (
            False,
            "Image is empty."
        )

    is_color, difference = (
        check_color_image(
            image
        )
    )

    if is_color:

        return (
            False,
            "Color image detected. Please input grayscale medical image."
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
# GENERAL PREPROCESSING
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
# MODALITY PREPROCESSING
# ============================================================

def preprocess_modality_image(
    image
):

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        MODALITY_IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = (
        image_array / 255.0
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# MODALITY PREDICTION
# ============================================================

def predict_modality(
    image
):

    image_array = preprocess_modality_image(
        image
    )

    prediction = modality_model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )

    if (
        prediction.ndim != 2
        or
        prediction.shape[1] != 3
    ):

        raise ValueError(
            "Modality classifier must output "
            f"3 classes. Received: "
            f"{prediction.shape}"
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

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    modality = MODALITY_CLASS_MAP[
        predicted_index
    ]

    return {
        "index": predicted_index,
        "class": modality,
        "confidence": confidence,
        "probabilities": probabilities
    }


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def predict_xray_verification(
    image
):

    image_array = preprocess_image(
        image,
        XRAY_VERIFIER_IMAGE_SIZE
    )

    prediction = xray_verifier_model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )

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
            >=
            XRAY_VERIFIER_THRESHOLD
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
            >=
            XRAY_VERIFIER_THRESHOLD
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
            "non_xray_probability": non_xray_probability
        }

    raise ValueError(
        "Unexpected X-ray verifier output shape: "
        f"{prediction.shape}"
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    image
):

    image_array = preprocess_image(
        image,
        PNEUMONIA_IMAGE_SIZE
    )

    prediction = pneumonia_model.predict(
        image_array,
        verbose=0
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
            f"2 classes. Received: "
            f"{prediction.shape}"
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

    normal_probability = float(
        probabilities[0]
    )

    pneumonia_probability = float(
        probabilities[1]
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "normal_probability": normal_probability,
        "pneumonia_probability": pneumonia_probability,
        "probabilities": probabilities
    }


# ============================================================
# GRAD-CAM++ — LAYER UTILITIES
# ============================================================

def _is_4d_feature_layer(layer):
    """
    Checks whether a layer exposes a 4-D feature map:

        (batch, height, width, channels)
    """

    try:

        output = layer.output

        if isinstance(
            output,
            (list, tuple)
        ):

            for item in output:

                try:

                    shape = item.shape

                    if (
                        shape is not None
                        and
                        shape.rank == 4
                    ):

                        return True

                except Exception:

                    continue

            return False

        shape = output.shape

        if shape is None:

            return False

        if shape.rank == 4:

            return True

        try:

            shape_list = shape.as_list()

            return (
                len(shape_list) == 4
                and
                shape_list[-1] is not None
            )

        except Exception:

            return False

    except Exception:

        return False


# ============================================================
# FIND NESTED XCEPTION
# ============================================================

def _find_xception_backbone(
    model
):

    for layer in model.layers:

        class_name = (
            layer.__class__.__name__
            .lower()
        )

        layer_name = (
            getattr(
                layer,
                "name",
                ""
            )
            .lower()
        )

        combined_name = (
            class_name
            + " "
            + layer_name
        )

        if (
            "xception" in combined_name
            and
            isinstance(
                layer,
                tf.keras.Model
            )
        ):

            return layer

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            found = (
                _find_xception_backbone(
                    layer
                )
            )

            if found is not None:

                return found

    return None


# ============================================================
# FIND DEEPEST CONVOLUTIONAL LAYER
# ============================================================

def _find_deepest_conv_layer(
    model
):

    preferred_layers = []

    fallback_layers = []

    for layer in reversed(
        model.layers
    ):

        if not _is_4d_feature_layer(
            layer
        ):

            continue

        class_name = (
            layer.__class__.__name__
            .lower()
        )

        if (
            "conv2d" in class_name
            or
            "separableconv" in class_name
            or
            "depthwiseconv" in class_name
        ):

            preferred_layers.append(
                layer
            )

        else:

            fallback_layers.append(
                layer
            )

    if preferred_layers:

        return preferred_layers[0]

    if fallback_layers:

        return fallback_layers[0]

    return None


# ============================================================
# FIND GRAD-CAM TARGET LAYER
# ============================================================

def find_gradcam_target_layer(
    model
):

    # --------------------------------------------------------
    # 1. Search outer model.
    # --------------------------------------------------------

    target_layer = (
        _find_deepest_conv_layer(
            model
        )
    )

    if target_layer is not None:

        return target_layer


    # --------------------------------------------------------
    # 2. Search nested Xception.
    # --------------------------------------------------------

    xception_backbone = (
        _find_xception_backbone(
            model
        )
    )

    if xception_backbone is not None:

        target_layer = (
            _find_deepest_conv_layer(
                xception_backbone
            )
        )

        if target_layer is not None:

            return target_layer


    # --------------------------------------------------------
    # 3. Recursive search.
    # --------------------------------------------------------

    def recursive_search(
        current_model
    ):

        for layer in reversed(
            current_model.layers
        ):

            if isinstance(
                layer,
                tf.keras.Model
            ):

                target = (
                    _find_deepest_conv_layer(
                        layer
                    )
                )

                if target is not None:

                    return target

                target = recursive_search(
                    layer
                )

                if target is not None:

                    return target

        return None


    target_layer = recursive_search(
        model
    )

    if target_layer is not None:

        return target_layer


    raise ValueError(
        "Grad-CAM++ could not find a suitable "
        "4-D convolutional feature layer.\n\n"
        "The pneumonia model does not expose a usable "
        "intermediate convolutional feature map."
    )


# ============================================================
# FIND DIRECT PARENT MODEL
# ============================================================

def _find_parent_model_for_layer(
    model,
    target_layer
):

    for layer in model.layers:

        if layer is target_layer:

            return model

        if isinstance(
            layer,
            tf.keras.Model
        ):

            found = (
                _find_parent_model_for_layer(
                    layer,
                    target_layer
                )
            )

            if found is not None:

                return found

    return None


# ============================================================
# FIND NESTED MODEL CONTAINING TARGET
# ============================================================

def _find_nested_model_for_layer(
    model,
    target_layer
):

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            if target_layer in layer.layers:

                return layer

            found = (
                _find_nested_model_for_layer(
                    layer,
                    target_layer
                )
            )

            if found is not None:

                return found

    return None


# ============================================================
# BUILD FLAT GRAD-CAM MODEL
# ============================================================

def _build_flat_gradcam_model(
    model,
    target_layer
):

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                target_layer.output,
                model.output
            ]
        )

        return grad_model

    except Exception:

        return None


# ============================================================
# BUILD NESTED GRAD-CAM GRAPH
# ============================================================

def _build_nested_gradcam_components(
    model,
    target_layer
):
    """
    Handles the important case:

        outer model
              |
              v
        nested Xception
              |
              v
        classification head

    We construct:

        image
          |
          +----> target convolution
          |
          +----> nested Xception output
                              |
                              v
                       outer classification head

    Both feature extraction and prediction therefore remain
    inside the same TensorFlow computational graph.
    """

    nested_model = (
        _find_nested_model_for_layer(
            model,
            target_layer
        )
    )

    if nested_model is None:

        raise ValueError(
            "Could not locate the nested model "
            "containing the Grad-CAM++ target layer."
        )


    # --------------------------------------------------------
    # Nested model returning:
    #
    #   target feature map
    #   nested model output
    # --------------------------------------------------------

    try:

        nested_feature_model = (
            tf.keras.models.Model(
                inputs=nested_model.inputs,
                outputs=[
                    target_layer.output,
                    nested_model.output
                ],
                name="gradcam_nested_feature_model"
            )
        )

    except Exception as e:

        raise RuntimeError(
            "Could not create the internal Xception "
            "feature extraction graph.\n\n"
            f"Target layer: {target_layer.name}\n"
            f"Nested model: {nested_model.name}\n\n"
            f"Original error:\n{e}"
        ) from e


    # --------------------------------------------------------
    # Create a model representing everything after the
    # nested Xception model.
    #
    # This is the crucial fix for the previous implementation.
    # --------------------------------------------------------

    try:

        nested_output = nested_model.output

        head_model = tf.keras.models.Model(
            inputs=nested_output,
            outputs=model.output,
            name="gradcam_classification_head"
        )

    except Exception as e:

        raise RuntimeError(
            "Could not connect the nested Xception output "
            "to the pneumonia classification head.\n\n"
            f"Nested model: {nested_model.name}\n\n"
            f"Original error:\n{e}"
        ) from e


    return (
        nested_feature_model,
        head_model,
        nested_model
    )


# ============================================================
# GENERATE GRAD-CAM++ HEATMAP
# ============================================================

def generate_gradcam_plus_plus(
    image,
    target_class_index=1
):
    """
    Generate Grad-CAM++ from the existing pneumonia model.

    Mapping:

        0 = Normal
        1 = Pneumonia

    Supports:

        - Flat Functional models
        - Nested Xception Functional models

    The important correction is that, for nested Xception,
    the target convolutional activation and final pneumonia
    prediction are connected through the SAME TensorFlow graph.
    """

    # ========================================================
    # FIND TARGET LAYER
    # ========================================================

    target_layer = (
        find_gradcam_target_layer(
            pneumonia_model
        )
    )

    target_layer_name = (
        target_layer.name
    )


    # ========================================================
    # PREPROCESS
    # ========================================================

    image_array = preprocess_image(
        image,
        PNEUMONIA_IMAGE_SIZE
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )


    # ========================================================
    # METHOD 1 — NORMAL FLAT MODEL
    # ========================================================

    flat_grad_model = (
        _build_flat_gradcam_model(
            pneumonia_model,
            target_layer
        )
    )

    if flat_grad_model is not None:

        try:

            with tf.GradientTape(
                persistent=True
            ) as tape3:

                with tf.GradientTape(
                    persistent=True
                ) as tape2:

                    with tf.GradientTape(
                        persistent=True
                    ) as tape1:

                        conv_features, predictions = (
                            flat_grad_model(
                                image_tensor,
                                training=False
                            )
                        )

                        class_score = (
                            predictions[
                                :,
                                target_class_index
                            ]
                        )

                    first_derivative = (
                        tape1.gradient(
                            class_score,
                            conv_features
                        )
                    )

                second_derivative = (
                    tape2.gradient(
                        first_derivative,
                        conv_features
                    )
                )

            third_derivative = (
                tape3.gradient(
                    second_derivative,
                    conv_features
                )
            )

            del tape1
            del tape2
            del tape3

            if (
                first_derivative is not None
                and
                second_derivative is not None
                and
                third_derivative is not None
            ):

                return (
                    _calculate_gradcam_plus_plus_heatmap(
                        conv_features,
                        first_derivative,
                        second_derivative,
                        third_derivative
                    ),
                    target_layer_name
                )

        except Exception:
            pass


    # ========================================================
    # METHOD 2 — NESTED XCEPTION
    # ========================================================

    (
        nested_feature_model,
        head_model,
        nested_model
    ) = _build_nested_gradcam_components(
        pneumonia_model,
        target_layer
    )


    # ========================================================
    # NESTED GRAPH FORWARD PASS
    # ========================================================

    with tf.GradientTape(
        persistent=True
    ) as tape3:

        with tf.GradientTape(
            persistent=True
        ) as tape2:

            with tf.GradientTape(
                persistent=True
            ) as tape1:

                # --------------------------------------------
                # IMPORTANT:
                #
                # This produces BOTH:
                #
                #   1. target convolutional feature map
                #   2. Xception output
                #
                # from the SAME forward pass.
                # --------------------------------------------

                conv_features, nested_output = (
                    nested_feature_model(
                        image_tensor,
                        training=False
                    )
                )

                # --------------------------------------------
                # Feed the nested Xception output directly
                # into the outer classification head.
                # --------------------------------------------

                predictions = (
                    head_model(
                        nested_output,
                        training=False
                    )
                )

                predictions = tf.convert_to_tensor(
                    predictions
                )

                # --------------------------------------------
                # Handle possible shape:
                #
                # (batch, 2)
                # --------------------------------------------

                if predictions.shape.rank != 2:

                    raise RuntimeError(
                        "The pneumonia classification head "
                        "returned an unexpected output shape: "
                        f"{predictions.shape}"
                    )

                class_score = (
                    predictions[
                        :,
                        target_class_index
                    ]
                )

            first_derivative = (
                tape1.gradient(
                    class_score,
                    conv_features
                )
            )

        second_derivative = (
            tape2.gradient(
                first_derivative,
                conv_features
            )
        )

    third_derivative = (
        tape3.gradient(
            second_derivative,
            conv_features
        )
    )

    del tape1
    del tape2
    del tape3


    # ========================================================
    # GRADIENT SAFETY CHECKS
    # ========================================================

    if first_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the first "
            "gradient.\n\n"
            f"Target layer: {target_layer_name}\n"
            f"Nested model: {nested_model.name}"
        )

    if second_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the second "
            "gradient.\n\n"
            f"Target layer: {target_layer_name}\n"
            f"Nested model: {nested_model.name}"
        )

    if third_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the third "
            "gradient.\n\n"
            f"Target layer: {target_layer_name}\n"
            f"Nested model: {nested_model.name}"
        )


    # ========================================================
    # CALCULATE HEATMAP
    # ========================================================

    heatmap = (
        _calculate_gradcam_plus_plus_heatmap(
            conv_features,
            first_derivative,
            second_derivative,
            third_derivative
        )
    )


    return (
        heatmap,
        target_layer_name
    )


# ============================================================
# GRAD-CAM++ MATHEMATICAL CALCULATION
# ============================================================

def _calculate_gradcam_plus_plus_heatmap(
    conv_features,
    first_derivative,
    second_derivative,
    third_derivative
):
    """
    Grad-CAM++ implementation.

    Input feature tensor:

        (1, H, W, C)

    Output:

        normalized 2-D heatmap
    """

    # --------------------------------------------------------
    # Ensure float32.
    # --------------------------------------------------------

    conv_features = tf.cast(
        conv_features,
        tf.float32
    )

    first_derivative = tf.cast(
        first_derivative,
        tf.float32
    )

    second_derivative = tf.cast(
        second_derivative,
        tf.float32
    )

    third_derivative = tf.cast(
        third_derivative,
        tf.float32
    )


    # --------------------------------------------------------
    # Positive second derivatives.
    # --------------------------------------------------------

    positive_second = tf.maximum(
        second_derivative,
        0.0
    )


    epsilon = tf.constant(
        1e-8,
        dtype=tf.float32
    )


    # --------------------------------------------------------
    # Grad-CAM++ alpha coefficient.
    # --------------------------------------------------------

    denominator = (
        2.0 * positive_second
        +
        conv_features
        *
        third_derivative
    )

    alpha = (
        positive_second
        /
        (
            denominator
            +
            epsilon
        )
    )


    # --------------------------------------------------------
    # Positive first derivatives.
    # --------------------------------------------------------

    positive_gradients = tf.maximum(
        first_derivative,
        0.0
    )


    # --------------------------------------------------------
    # Channel weights.
    # --------------------------------------------------------

    weights = tf.reduce_sum(
        alpha * positive_gradients,
        axis=(1, 2)
    )


    # --------------------------------------------------------
    # Weighted feature maps.
    # --------------------------------------------------------

    weighted_features = (
        conv_features
        *
        weights[
            :,
            tf.newaxis,
            tf.newaxis,
            :
        ]
    )


    # --------------------------------------------------------
    # Combine channels.
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )


    # --------------------------------------------------------
    # Keep positive activations.
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0.0
    )


    heatmap = heatmap[0]


    # --------------------------------------------------------
    # Normalize.
    # --------------------------------------------------------

    heatmap_max = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        heatmap_max > 0,
        heatmap /
        (
            heatmap_max
            +
            epsilon
        ),
        tf.zeros_like(
            heatmap
        )
    )


    # --------------------------------------------------------
    # Convert to NumPy.
    # --------------------------------------------------------

    heatmap = heatmap.numpy()

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )


    return heatmap


# ============================================================
# CREATE GRAD-CAM++ OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):
    """
    Creates:

        1. Original image
        2. Heatmap
        3. Grad-CAM++ overlay

    Returns PIL images.
    """

    # --------------------------------------------------------
    # Prepare original image.
    # --------------------------------------------------------

    original = ImageOps.exif_transpose(
        image
    ).convert(
        "RGB"
    )

    original = original.resize(
        GRADCAM_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # Resize heatmap.
    # --------------------------------------------------------

    heatmap_uint8 = (
        np.clip(
            heatmap,
            0.0,
            1.0
        )
        * 255.0
    ).astype(
        np.uint8
    )

    heatmap_pil = Image.fromarray(
        heatmap_uint8,
        mode="L"
    )

    heatmap_pil = heatmap_pil.resize(
        GRADCAM_IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # Create colored heatmap.
    #
    # Use OpenCV if available.
    # Otherwise use a PIL-based gradient.
    # --------------------------------------------------------

    try:

        import cv2

        heatmap_array = np.asarray(
            heatmap_pil,
            dtype=np.uint8
        )

        colored_heatmap = cv2.applyColorMap(
            heatmap_array,
            cv2.COLORMAP_JET
        )

        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )

        heatmap_image = Image.fromarray(
            colored_heatmap
        )

    except Exception:

        # ----------------------------------------------------
        # Fallback heatmap implementation.
        # ----------------------------------------------------

        h = np.asarray(
            heatmap_pil,
            dtype=np.float32
        ) / 255.0

        r = np.clip(
            1.5 * h,
            0,
            1
        )

        g = np.clip(
            1.5 * (1.0 - np.abs(h - 0.5) * 2.0),
            0,
            1
        )

        b = np.clip(
            1.5 * (1.0 - h),
            0,
            1
        )

        rgb = np.stack(
            [r, g, b],
            axis=-1
        )

        heatmap_image = Image.fromarray(
            (rgb * 255).astype(
                np.uint8
            )
        )


    # --------------------------------------------------------
    # Create overlay.
    # --------------------------------------------------------

    overlay = Image.blend(
        original,
        heatmap_image.convert(
            "RGB"
        ),
        GRADCAM_OVERLAY_ALPHA
    )


    return (
        original,
        heatmap_image,
        overlay
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    modality_result,
    verifier_result,
    pneumonia_result=None,
    gradcam_overlay=None,
    gradcam_layer_name=None
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


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "PneuX-ModNet<br/>"
            "AI-Based Medical Image Analysis System",
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


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

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
    # MODALITY REPORT
    # ========================================================

    modality = (
        modality_result["class"]
    )

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

    story.append(
        Spacer(
            1,
            10
        )
    )


    # ========================================================
    # X-RAY VERIFICATION REPORT
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
    # PNEUMONIA REPORT
    # ========================================================

    if pneumonia_result is not None:

        diagnosis = (
            pneumonia_result["class"]
        )

        confidence = (
            pneumonia_result["confidence"]
            * 100
        )


        # ----------------------------------------------------
        # PNEUMONIA
        # ----------------------------------------------------

        if diagnosis == "Pneumonia":

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
                    "Pneumonia Probability",
                    f"{pneumonia_probability:.2f}%"
                ]
            ]


        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        else:

            normal_probability = (
                pneumonia_result[
                    "normal_probability"
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
                70 * mm,
                80 * mm
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
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    )
                ]
            )
        )

        story.append(
            pneumonia_table
        )

        story.append(
            Spacer(
                1,
                12
            )
        )


        # ====================================================
        # GRAD-CAM++ PDF SECTION
        # ====================================================

        if (
            diagnosis == "Pneumonia"
            and
            gradcam_overlay is not None
        ):

            story.append(
                Paragraph(
                    "4. Pneumonia Localization "
                    "(Grad-CAM++)",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    "The highlighted region represents "
                    "areas of the chest X-ray that "
                    "contributed strongly to the "
                    "Pneumonia prediction.",
                    normal_style
                )
            )

            story.append(
                Spacer(
                    1,
                    8
                )
            )

            gradcam_buffer = io.BytesIO()

            gradcam_overlay.save(
                gradcam_buffer,
                format="PNG"
            )

            gradcam_buffer.seek(0)

            gradcam_report_image = RLImage(
                gradcam_buffer,
                width=140 * mm,
                height=140 * mm
            )

            story.append(
                gradcam_report_image
            )

            story.append(
                Spacer(
                    1,
                    8
                )
            )

            if gradcam_layer_name:

                story.append(
                    Paragraph(
                        f"<b>Grad-CAM++ Feature Layer:</b> "
                        f"{gradcam_layer_name}",
                        normal_style
                    )
                )

            story.append(
                Spacer(
                    1,
                    8
                )
            )

            story.append(
                Paragraph(
                    "<b>Localization Note:</b> "
                    "Grad-CAM++ is an explainability "
                    "technique and highlights image "
                    "regions associated with the model's "
                    "prediction. It is not a pixel-level "
                    "clinical segmentation or a confirmed "
                    "boundary of disease.",
                    normal_style
                )
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


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# UPLOAD SECTION
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
        "Upload a grayscale medical image "
        "such as CT, MRI, or X-ray."
    )
)


# ============================================================
# ANALYZE
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
    # DISPLAY IMAGE
    # ========================================================

    st.subheader(
        "Uploaded Medical Image"
    )

    st.image(
        image,
        caption="Uploaded Medical Image",
        use_container_width=True
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "Check Your Image By Initiating AI Analysis",
        type="primary",
        use_container_width=True
    )


    if analyze:

        # ====================================================
        # STEP 1 — BASIC VALIDATION
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
        # STEP 2 — MODALITY CLASSIFICATION
        # ====================================================

        with st.spinner(
            "Classifying medical image modality..."
        ):

            try:

                modality_result = (
                    predict_modality(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "Medical image modality classification failed."
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
        #
        # WEB INTERFACE SHOWS ONLY CLASS.
        # ====================================================

        st.markdown(
            "## Detected Medical Image Type"
        )

        st.markdown(
            f"### {modality}"
        )


        # ====================================================
        # STEP 3 — CT / MRI STOP
        # ====================================================

        if modality in (
            "CT",
            "MRI"
        ):

            st.warning(
                f"This medical image was classified as "
                f"**{modality}**."
            )

            st.info(
                "Pneumonia detection is available only "
                "for chest X-ray images. Analysis has "
                "therefore stopped."
            )

            st.session_state.analysis_result = {
                "modality": modality,
                "modality_confidence": modality_confidence,
                "verifier": None,
                "pneumonia": None,
                "gradcam": None
            }

            st.session_state.pdf_report = (
                create_pdf_report(
                    image,
                    modality_result,
                    None,
                    None,
                    None,
                    None
                )
            )

            st.download_button(
                label="Download Modality Report (PDF)",
                data=st.session_state.pdf_report,
                file_name="medical_image_modality_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.stop()


        # ====================================================
        # STEP 4 — X-RAY VERIFICATION
        # ====================================================

        if modality == "X-ray":

            st.markdown(
                "## X-ray Verification"
            )

            with st.spinner(
                "Verifying chest X-ray image..."
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


            # =================================================
            # NOT X-RAY
            # =================================================

            if not verifier_result["is_xray"]:

                st.error(
                    "The X-ray verifier did not "
                    "confirm this image as an X-ray."
                )

                st.info(
                    "Pneumonia detection has been stopped."
                )

                st.session_state.analysis_result = {
                    "modality": modality,
                    "modality_confidence": modality_confidence,
                    "verifier": verifier_result,
                    "pneumonia": None,
                    "gradcam": None
                }

                st.session_state.pdf_report = (
                    create_pdf_report(
                        image,
                        modality_result,
                        verifier_result,
                        None,
                        None,
                        None
                    )
                )

                st.download_button(
                    label="Download X-ray Verification Report (PDF)",
                    data=st.session_state.pdf_report,
                    file_name="xray_verification_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                st.stop()


            # =================================================
            # STEP 5 — PNEUMONIA DETECTION
            # =================================================

            st.success(
                "Chest X-ray verified successfully."
            )

            st.markdown(
                "## Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing chest X-ray for pneumonia..."
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


            # =================================================
            # FINAL RESULT
            #
            # WEB INTERFACE SHOWS ONLY CLASS.
            # =================================================

            if diagnosis == "Pneumonia":

                st.error(
                    f"### Final Result: {diagnosis}"
                )

            else:

                st.success(
                    f"### Final Result: {diagnosis}"
                )


            # =================================================
            # STEP 6 — GRAD-CAM++
            #
            # ONLY RUN WHEN PNEUMONIA IS DETECTED.
            # =================================================

            gradcam_overlay = None

            gradcam_layer_name = None

            if diagnosis == "Pneumonia":

                st.markdown(
                    "## Pneumonia Localization"
                )

                with st.spinner(
                    "Generating Grad-CAM++ localization..."
                ):

                    try:

                        (
                            heatmap,
                            gradcam_layer_name
                        ) = (
                            generate_gradcam_plus_plus(
                                image,
                                target_class_index=1
                            )
                        )

                        (
                            original_gradcam_image,
                            heatmap_image,
                            gradcam_overlay
                        ) = create_gradcam_overlay(
                            image,
                            heatmap
                        )

                    except Exception as e:

                        st.error(
                            "Grad-CAM++ localization could not "
                            "be generated."
                        )

                        st.exception(e)

                        gradcam_overlay = None


                # ------------------------------------------------
                # DISPLAY GRAD-CAM++
                # ------------------------------------------------

                if gradcam_overlay is not None:

                    st.success(
                        "Pneumonia localization generated "
                        "using Grad-CAM++."
                    )

                    st.image(
                        gradcam_overlay,
                        caption=(
                            "Grad-CAM++ Pneumonia "
                            "Localization"
                        ),
                        use_container_width=True
                    )

                    st.info(
                        "Highlighted regions indicate areas "
                        "that contributed strongly to the "
                        "Pneumonia prediction."
                    )


            # =================================================
            # CREATE PDF
            # =================================================

            pdf_bytes = create_pdf_report(
                image,
                modality_result,
                verifier_result,
                pneumonia_result,
                gradcam_overlay,
                gradcam_layer_name
            )

            st.session_state.analysis_result = {
                "modality": modality,
                "modality_confidence": modality_confidence,
                "verifier": verifier_result,
                "pneumonia": pneumonia_result,
                "gradcam": gradcam_overlay,
                "gradcam_layer": gradcam_layer_name
            }

            st.session_state.pdf_report = (
                pdf_bytes
            )


            # =================================================
            # PDF DOWNLOAD
            # =================================================

            st.divider()

            st.subheader(
                "Analysis Report"
            )

            st.download_button(
                label="Download Final Report (PDF)",
                data=pdf_bytes,
                file_name=(
                    "pneumonia_detection_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )
