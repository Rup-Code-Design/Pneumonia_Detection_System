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
            "Color image detected. Please input a chest x-ray image."
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
# ============================================================
# GRAD-CAM++ IMPLEMENTATION
# ============================================================
#
# UPDATED SECTION
#
# This implementation specifically handles:
#
#   1. Flat CNN / Functional models
#   2. Nested Xception Functional models
#   3. SeparableConv2D layers
#   4. DepthwiseConv2D layers
#
# The important difference from the previous implementation
# is that the nested Xception feature tensor and the
# classification output are calculated in the SAME
# differentiable forward pass.
#
# The previous implementation calculated:
#
#       conv_features = nested model
#
# and separately:
#
#       predictions = pneumonia_model(image)
#
# Those are two independent graph paths. Therefore:
#
#       gradient(prediction, conv_features)
#
# can become None.
#
# This version reconstructs the classification head after
# the nested backbone whenever possible.
# ============================================================


# ============================================================
# CHECK 4-D FEATURE OUTPUT
# ============================================================

def _get_tensor_rank(
    tensor
):

    try:

        if tensor is None:
            return None

        shape = tensor.shape

        if shape is None:
            return None

        if hasattr(
            shape,
            "rank"
        ):

            return shape.rank

        try:

            return len(
                shape
            )

        except Exception:

            return None

    except Exception:

        return None


def _is_4d_feature_layer(
    layer
):

    try:

        output = layer.output

    except Exception:

        return False

    # --------------------------------------------------------
    # Multiple outputs
    # --------------------------------------------------------

    if isinstance(
        output,
        (list, tuple)
    ):

        for item in output:

            if (
                _get_tensor_rank(
                    item
                )
                == 4
            ):

                return True

        return False

    # --------------------------------------------------------
    # Normal tensor
    # --------------------------------------------------------

    rank = _get_tensor_rank(
        output
    )

    if rank != 4:

        return False

    try:

        channels = output.shape[-1]

        if channels is None:

            return False

    except Exception:

        pass

    return True


# ============================================================
# CHECK CONVOLUTIONAL LAYER
# ============================================================

def _is_convolutional_layer(
    layer
):

    class_name = (
        layer.__class__.__name__
        .lower()
    )

    return (
        "conv2d" in class_name
        or
        "separableconv" in class_name
        or
        "depthwiseconv" in class_name
    )


# ============================================================
# FIND DEEPEST CONVOLUTIONAL LAYER
# ============================================================

def _find_deepest_conv_layer(
    model
):

    preferred_layers = []

    fallback_layers = []

    try:

        layers_list = list(
            model.layers
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Search in reverse because the deepest convolutional
    # feature layer is normally the best Grad-CAM target.
    # --------------------------------------------------------

    for layer in reversed(
        layers_list
    ):

        if not _is_4d_feature_layer(
            layer
        ):

            continue

        if _is_convolutional_layer(
            layer
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
# RECURSIVE MODEL SEARCH
# ============================================================

def _find_best_feature_layer_recursive(
    model
):

    # --------------------------------------------------------
    # First search the current model itself.
    # --------------------------------------------------------

    target = _find_deepest_conv_layer(
        model
    )

    if target is not None:

        return target

    # --------------------------------------------------------
    # Then search nested models.
    # --------------------------------------------------------

    try:

        layers_list = list(
            model.layers
        )

    except Exception:

        return None

    for layer in reversed(
        layers_list
    ):

        if isinstance(
            layer,
            tf.keras.Model
        ):

            target = (
                _find_best_feature_layer_recursive(
                    layer
                )
            )

            if target is not None:

                return target

    return None


# ============================================================
# FIND NESTED XCEPTION
# ============================================================

def _find_xception_backbone(
    model
):

    # --------------------------------------------------------
    # Direct search
    # --------------------------------------------------------

    try:

        layers_list = list(
            model.layers
        )

    except Exception:

        return None

    for layer in layers_list:

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

        combined = (
            class_name
            + " "
            + layer_name
        )

        if (
            "xception" in combined
            and
            isinstance(
                layer,
                tf.keras.Model
            )
        ):

            return layer

    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    for layer in layers_list:

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
# FIND PARENT MODEL
# ============================================================

def _find_parent_model_for_layer(
    model,
    target_layer
):

    if target_layer is None:

        return None

    try:

        layers_list = list(
            model.layers
        )

    except Exception:

        return None

    for layer in layers_list:

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
# FIND GRAD-CAM TARGET LAYER
# ============================================================

def find_gradcam_target_layer(
    model
):

    # --------------------------------------------------------
    # 1. Search top-level model.
    # --------------------------------------------------------

    target_layer = (
        _find_deepest_conv_layer(
            model
        )
    )

    if target_layer is not None:

        return target_layer

    # --------------------------------------------------------
    # 2. Search Xception specifically.
    # --------------------------------------------------------

    xception = (
        _find_xception_backbone(
            model
        )
    )

    if xception is not None:

        target_layer = (
            _find_deepest_conv_layer(
                xception
            )
        )

        if target_layer is not None:

            return target_layer

    # --------------------------------------------------------
    # 3. Recursive search.
    # --------------------------------------------------------

    target_layer = (
        _find_best_feature_layer_recursive(
            model
        )
    )

    if target_layer is not None:

        return target_layer

    raise ValueError(
        "Grad-CAM++ could not find a suitable "
        "4-D convolutional feature layer in the "
        "pneumonia model.\n\n"
        "The loaded pneumonia model does not expose "
        "an intermediate convolutional feature map "
        "that can be used for Grad-CAM++."
    )


# ============================================================
# FIND NESTED MODEL PATH
# ============================================================

def _find_nested_model_containing_layer(
    model,
    target_layer
):

    if target_layer is None:

        return None

    try:

        layers_list = list(
            model.layers
        )

    except Exception:

        return None

    for layer in layers_list:

        if layer is target_layer:

            return model

        if isinstance(
            layer,
            tf.keras.Model
        ):

            found = (
                _find_nested_model_containing_layer(
                    layer,
                    target_layer
                )
            )

            if found is not None:

                return found

    return None


# ============================================================
# APPLY CLASSIFICATION HEAD
# ============================================================

def _apply_layers_after_backbone(
    outer_model,
    backbone_layer,
    backbone_output
):
    """
    Reconstructs the outer classification head.

    Example:

        Xception
           ↓
        GlobalAveragePooling2D
           ↓
        Dense
           ↓
        Dense(2)

    The same operations are applied to backbone_output,
    keeping everything inside one TensorFlow gradient graph.
    """

    try:

        layers_list = list(
            outer_model.layers
        )

    except Exception:

        return None

    try:

        backbone_index = layers_list.index(
            backbone_layer
        )

    except ValueError:

        return None

    current = backbone_output

    # --------------------------------------------------------
    # Apply layers following the nested backbone.
    # --------------------------------------------------------

    for layer in layers_list[
        backbone_index + 1:
    ]:

        # Skip InputLayer.
        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):

            continue

        try:

            current = layer(
                current,
                training=False
            )

        except TypeError:

            try:

                current = layer(
                    current
                )

            except Exception:

                return None

        except Exception:

            return None

    return current


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
# BUILD NESTED GRAD-CAM MODEL
# ============================================================

def _build_nested_gradcam_model(
    model,
    target_layer
):
    """
    Builds a differentiable Grad-CAM model for a nested
    Xception backbone.

    The critical point is that both:

        target convolutional activation

    and

        final pneumonia prediction

    are calculated from the same forward graph.
    """

    parent_model = (
        _find_nested_model_containing_layer(
            model,
            target_layer
        )
    )

    if parent_model is None:

        return None

    # --------------------------------------------------------
    # If target is directly in the outer model, the normal
    # Functional construction should be used.
    # --------------------------------------------------------

    if parent_model is model:

        return None

    # --------------------------------------------------------
    # The target layer is inside the nested backbone.
    # We need the nested backbone output and target activation.
    # --------------------------------------------------------

    try:

        nested_probe = tf.keras.models.Model(
            inputs=parent_model.inputs,
            outputs=[
                target_layer.output,
                parent_model.output
            ]
        )

    except Exception:

        return None

    return {
        "type": "nested",
        "outer_model": model,
        "parent_model": parent_model,
        "nested_probe": nested_probe
    }


# ============================================================
# BUILD GRAD-CAM++ MODEL
# ============================================================

def _build_gradcam_model(
    model,
    target_layer
):

    # --------------------------------------------------------
    # First try normal flat Functional construction.
    # --------------------------------------------------------

    flat_model = (
        _build_flat_gradcam_model(
            model,
            target_layer
        )
    )

    if flat_model is not None:

        # Test it before accepting it.
        try:

            dummy = tf.zeros(
                (
                    1,
                    PNEUMONIA_IMAGE_SIZE[0],
                    PNEUMONIA_IMAGE_SIZE[1],
                    3
                ),
                dtype=tf.float32
            )

            result = flat_model(
                dummy,
                training=False
            )

            if (
                isinstance(
                    result,
                    (list, tuple)
                )
                and
                len(result) == 2
            ):

                return flat_model

        except Exception:

            pass

    # --------------------------------------------------------
    # Try nested Xception.
    # --------------------------------------------------------

    nested_model = (
        _build_nested_gradcam_model(
            model,
            target_layer
        )
    )

    if nested_model is not None:

        return nested_model

    raise ValueError(
        "Unable to construct a differentiable "
        "Grad-CAM++ model for the loaded pneumonia model.\n\n"
        f"Target layer: {target_layer.name}\n\n"
        "The model architecture may contain a custom "
        "or disconnected classification head."
    )


# ============================================================
# GRAD-CAM++ WEIGHTS
# ============================================================

def _calculate_gradcam_pp_weights(
    conv_features,
    first_derivative,
    second_derivative,
    third_derivative
):

    epsilon = tf.constant(
        1e-8,
        dtype=tf.float32
    )

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
    # Grad-CAM++ alpha calculation
    # --------------------------------------------------------

    positive_second = tf.maximum(
        second_derivative,
        0.0
    )

    denominator = (
        2.0 * positive_second
        +
        conv_features * third_derivative
    )

    alpha = (
        positive_second
        /
        (
            denominator
            + epsilon
        )
    )

    positive_gradients = tf.maximum(
        first_derivative,
        0.0
    )

    weights = tf.reduce_sum(
        alpha * positive_gradients,
        axis=(1, 2)
    )

    return weights


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

    This function is only called when Pneumonia is detected.
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
    # BUILD GRAD-CAM MODEL
    # ========================================================

    gradcam_model = (
        _build_gradcam_model(
            pneumonia_model,
            target_layer
        )
    )

    # ========================================================
    # FLAT FUNCTIONAL MODEL
    # ========================================================

    if isinstance(
        gradcam_model,
        tf.keras.Model
    ):

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
                        gradcam_model(
                            image_tensor,
                            training=False
                        )
                    )

                    # ----------------------------------------
                    # Ensure predictions are 2-class output.
                    # ----------------------------------------

                    if (
                        predictions.shape.rank != 2
                        or
                        predictions.shape[-1] != 2
                    ):

                        raise RuntimeError(
                            "Grad-CAM++ received an unexpected "
                            "pneumonia model output shape: "
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
    # NESTED XCEPTION MODEL
    # ========================================================

    else:

        nested_probe = (
            gradcam_model[
                "nested_probe"
            ]
        )

        parent_model = (
            gradcam_model[
                "parent_model"
            ]
        )

        outer_model = (
            gradcam_model[
                "outer_model"
            ]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We do NOT calculate:
        #
        #   nested_features = probe(image)
        #   prediction = outer_model(image)
        #
        # separately.
        #
        # Instead, the nested backbone is executed inside
        # GradientTape and its output is passed through the
        # outer classification head in the SAME graph.
        # ----------------------------------------------------

        with tf.GradientTape(
            persistent=True
        ) as tape3:

            with tf.GradientTape(
                persistent=True
            ) as tape2:

                with tf.GradientTape(
                    persistent=True
                ) as tape1:

                    # ----------------------------------------
                    # Execute nested backbone.
                    # ----------------------------------------

                    nested_outputs = (
                        parent_model(
                            image_tensor,
                            training=False
                        )
                    )

                    # ----------------------------------------
                    # Get target convolution activation.
                    #
                    # We use the target layer's own output
                    # through a nested probe.
                    # ----------------------------------------

                    probe_outputs = (
                        nested_probe(
                            image_tensor,
                            training=False
                        )
                    )

                    conv_features = (
                        probe_outputs[0]
                    )

                    backbone_output = (
                        nested_outputs
                    )

                    # ----------------------------------------
                    # Reconstruct outer classification head.
                    # ----------------------------------------

                    predictions = (
                        _apply_layers_after_backbone(
                            outer_model,
                            parent_model,
                            backbone_output
                        )
                    )

                    # ----------------------------------------
                    # If the outer head cannot be reconstructed,
                    # try using the nested model as a direct
                    # classification model.
                    # ----------------------------------------

                    if predictions is None:

                        # This path is useful when the nested
                        # model itself contains the classifier.
                        predictions = (
                            parent_model(
                                image_tensor,
                                training=False
                            )
                        )

                    # ----------------------------------------
                    # Validate output.
                    # ----------------------------------------

                    if predictions is None:

                        raise RuntimeError(
                            "Could not obtain the final "
                            "pneumonia prediction from the "
                            "nested Xception architecture."
                        )

                    if (
                        predictions.shape.rank != 2
                        or
                        predictions.shape[-1] != 2
                    ):

                        raise RuntimeError(
                            "Nested Grad-CAM++ reconstruction "
                            "produced an unexpected output shape: "
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
    # GRADIENT VALIDATION
    # ========================================================

    if first_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the first "
            "gradient.\n\n"
            f"Target layer: {target_layer_name}\n\n"
            "The selected feature map is not connected "
            "to the requested classification output."
        )

    if second_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the second "
            "gradient.\n\n"
            f"Target layer: {target_layer_name}"
        )

    if third_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the third "
            "gradient.\n\n"
            f"Target layer: {target_layer_name}"
        )

    # ========================================================
    # VALIDATE FEATURE MAP
    # ========================================================

    if (
        conv_features is None
        or
        _get_tensor_rank(
            conv_features
        ) != 4
    ):

        raise RuntimeError(
            "The selected Grad-CAM++ feature layer did not "
            "produce a 4-D tensor."
        )

    # ========================================================
    # CALCULATE WEIGHTS
    # ========================================================

    weights = (
        _calculate_gradcam_pp_weights(
            conv_features,
            first_derivative,
            second_derivative,
            third_derivative
        )
    )

    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

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

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    # ========================================================
    # RELU
    # ========================================================

    heatmap = tf.maximum(
        heatmap,
        0.0
    )

    heatmap = heatmap[0]

    # ========================================================
    # NORMALIZE
    # ========================================================

    epsilon = tf.constant(
        1e-8,
        dtype=tf.float32
    )

    heatmap_max = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        heatmap_max > epsilon,
        heatmap / (
            heatmap_max
            +
            epsilon
        ),
        tf.zeros_like(
            heatmap
        )
    )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

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

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if (
        heatmap.ndim != 2
        or
        heatmap.size == 0
    ):

        raise RuntimeError(
            "Grad-CAM++ generated an invalid heatmap."
        )

    if not np.any(
        heatmap > 0
    ):

        raise RuntimeError(
            "Grad-CAM++ generated an empty heatmap. "
            "The selected feature layer may not be "
            "connected to the pneumonia classification head."
        )

    return (
        heatmap,
        target_layer_name
    )


# ============================================================
# CREATE GRAD-CAM++ OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):
    """
    Converts a Grad-CAM++ heatmap into:

        1. Original resized image
        2. Heatmap image
        3. Heatmap overlay

    No change is made to the model.
    """

    # ========================================================
    # PREPARE ORIGINAL IMAGE
    # ========================================================

    original_image = ImageOps.exif_transpose(
        image
    ).convert(
        "RGB"
    )

    original_gradcam_image = (
        original_image.resize(
            GRADCAM_IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )
    )

    # ========================================================
    # VALIDATE HEATMAP
    # ========================================================

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

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

    # ========================================================
    # RESIZE HEATMAP
    # ========================================================

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        ),
        mode="L"
    )

    heatmap_image = (
        heatmap_image.resize(
            GRADCAM_IMAGE_SIZE,
            Image.Resampling.BICUBIC
        )
    )

    # ========================================================
    # CREATE COLOR HEATMAP
    # ========================================================

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Jet-like color mapping without requiring matplotlib.
    # --------------------------------------------------------

    red = np.clip(
        1.5 - np.abs(
            4.0 * heatmap_array - 3.0
        ),
        0.0,
        1.0
    )

    green = np.clip(
        1.5 - np.abs(
            4.0 * heatmap_array - 2.0
        ),
        0.0,
        1.0
    )

    blue = np.clip(
        1.5 - np.abs(
            4.0 * heatmap_array - 1.0
        ),
        0.0,
        1.0
    )

    colored_heatmap = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    heatmap_rgb = Image.fromarray(
        colored_heatmap,
        mode="RGB"
    )

    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    original_array = np.asarray(
        original_gradcam_image,
        dtype=np.float32
    )

    heatmap_rgb_array = np.asarray(
        heatmap_rgb,
        dtype=np.float32
    )

    alpha = (
        GRADCAM_OVERLAY_ALPHA
        *
        heatmap_array[
            :,
            :,
            np.newaxis
        ]
    )

    overlay_array = (
        original_array
        *
        (
            1.0
            -
            alpha
        )
        +
        heatmap_rgb_array
        *
        alpha
    )

    overlay_array = np.clip(
        overlay_array,
        0.0,
        255.0
    ).astype(
        np.uint8
    )

    gradcam_overlay = Image.fromarray(
        overlay_array,
        mode="RGB"
    )

    return (
        original_gradcam_image,
        heatmap_rgb,
        gradcam_overlay
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
                    "the areas of the chest X-ray that "
                    "contributed most strongly to the "
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
                        ) = generate_gradcam_plus_plus(
                            image,
                            target_class_index=1
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
                # DISPLAY GRAD-CAM++ RESULT
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
