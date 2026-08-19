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

from PIL import Image, ImageOps, ImageFilter

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

# Base overlay transparency.
GRADCAM_OVERLAY_ALPHA = 0.70

# Heatmap processing.
GRADCAM_LOW_PERCENTILE = 55.0
GRADCAM_HIGH_PERCENTILE = 99.0

# Suppress very weak activations.
GRADCAM_MIN_ACTIVATION = 0.20

# Gamma below 1 makes moderately strong regions more visible.
GRADCAM_GAMMA = 0.70

# Gaussian smoothing for cleaner localization.
GRADCAM_BLUR_RADIUS = 1.2


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

    /* ======================================================
       MAIN APPLICATION TITLE
       ====================================================== */

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.15;
        margin-top: 5px;
        margin-bottom: 8px;
    }


    /* ======================================================
       APPLICATION SUBTITLE
       ====================================================== */

    .subtitle {
        text-align: center;
        font-size: 17px;
        line-height: 1.5;
        margin-top: 4px;
        margin-bottom: 25px;
    }


    /* ======================================================
       SECTION TITLE
       ====================================================== */

    .section-title {
        text-align: center;
        font-size: 26px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }


    /* ======================================================
       RESULT BOX
       ====================================================== */

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        margin-top: 15px;
        margin-bottom: 15px;
    }


    /* ======================================================
       MODALITY RESULT
       ====================================================== */

    .modality-result {
        background-color: #eef4ff;
    }


    /* ======================================================
       NORMAL RESULT
       ====================================================== */

    .normal-result {
        background-color: #eaf7ea;
    }


    /* ======================================================
       PNEUMONIA RESULT
       ====================================================== */

    .pneumonia-result {
        background-color: #fdeaea;
    }


    /* ======================================================
       SMALL INFORMATION TEXT
       ====================================================== */

    .info-text {
        text-align: center;
        font-size: 15px;
        line-height: 1.5;
    }


    /* ======================================================
       BRAND NAME
       ====================================================== */

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
# ============================================================
# GRAD-CAM++ SECTION
# ============================================================
#
# This section has been redesigned to correctly handle:
#
# 1. Flat Functional models
# 2. Nested Xception Functional models
# 3. Deeply nested Functional models
# 4. SeparableConv2D layers used by Xception
# 5. Activation layers following the final convolution
#
# Most importantly, the nested-model path keeps the feature
# activation and classification prediction inside the SAME
# differentiable graph.
# ============================================================


# ============================================================
# GRAD-CAM++ LAYER INFORMATION
# ============================================================

def _safe_rank_from_shape(
    shape
):
    """
    Safely obtains tensor rank across TensorFlow/Keras
    versions.
    """

    if shape is None:

        return None

    try:

        rank = shape.rank

        if rank is not None:

            return int(rank)

    except Exception:

        pass

    try:

        shape_list = shape.as_list()

        return len(shape_list)

    except Exception:

        pass

    try:

        return len(shape)

    except Exception:

        return None


def _get_layer_output_tensors(
    layer
):
    """
    Returns all symbolic output tensors exposed by a Keras
    layer.

    Keras versions can expose outputs slightly differently,
    so several safe access paths are used.
    """

    tensors = []

    # --------------------------------------------------------
    # Standard .outputs
    # --------------------------------------------------------

    try:

        outputs = layer.outputs

        if outputs is not None:

            if isinstance(
                outputs,
                (list, tuple)
            ):

                tensors.extend(
                    outputs
                )

            else:

                tensors.append(
                    outputs
                )

    except Exception:

        pass

    # --------------------------------------------------------
    # Standard .output
    # --------------------------------------------------------

    try:

        output = layer.output

        if output is not None:

            if isinstance(
                output,
                (list, tuple)
            ):

                tensors.extend(
                    output
                )

            else:

                tensors.append(
                    output
                )

    except Exception:

        pass

    # --------------------------------------------------------
    # Inbound nodes fallback
    # --------------------------------------------------------

    try:

        for node in layer._inbound_nodes:

            try:

                node_outputs = (
                    node.output_tensors
                )

            except Exception:

                try:

                    node_outputs = (
                        node.outputs
                    )

                except Exception:

                    node_outputs = None

            if node_outputs is None:

                continue

            if isinstance(
                node_outputs,
                (list, tuple)
            ):

                tensors.extend(
                    node_outputs
                )

            else:

                tensors.append(
                    node_outputs
                )

    except Exception:

        pass

    # --------------------------------------------------------
    # Remove duplicate tensor objects.
    # --------------------------------------------------------

    unique = []

    seen = set()

    for tensor in tensors:

        try:

            identifier = id(tensor)

            if identifier not in seen:

                seen.add(
                    identifier
                )

                unique.append(
                    tensor
                )

        except Exception:

            continue

    return unique


def _layer_has_4d_output(
    layer
):
    """
    Determines whether a layer exposes a 4-D tensor:
        (batch, height, width, channels)
    """

    tensors = (
        _get_layer_output_tensors(
            layer
        )
    )

    for tensor in tensors:

        try:

            rank = _safe_rank_from_shape(
                tensor.shape
            )

            if rank == 4:

                shape = tensor.shape

                try:

                    channels = (
                        shape[-1]
                    )

                    if channels is not None:

                        return True

                except Exception:

                    return True

        except Exception:

            continue

    return False


# ============================================================
# RECURSIVE MODEL WALKER
# ============================================================

def _walk_model_layers(
    model,
    depth=0,
    visited=None
):
    """
    Recursively walks every nested Keras Model/Layer.

    Returns tuples:
        (layer, depth)

    This is more reliable for nested Xception models than
    searching only the outer model.layers list.
    """

    if visited is None:

        visited = set()

    results = []

    try:

        model_id = id(model)

        if model_id in visited:

            return results

        visited.add(
            model_id
        )

    except Exception:

        pass

    try:

        layers = list(
            model.layers
        )

    except Exception:

        layers = []

    for layer in layers:

        try:

            layer_id = id(layer)

            if layer_id in visited:

                continue

            visited.add(
                layer_id
            )

        except Exception:

            pass

        results.append(
            (
                layer,
                depth
            )
        )

        # ----------------------------------------------------
        # Recurse into nested Functional/Sequential models.
        # ----------------------------------------------------

        if isinstance(
            layer,
            tf.keras.Model
        ):

            results.extend(
                _walk_model_layers(
                    layer,
                    depth=depth + 1,
                    visited=visited
                )
            )

    return results


# ============================================================
# TARGET LAYER SCORING
# ============================================================

def _gradcam_layer_score(
    layer,
    depth
):
    """
    Gives a score to candidate feature layers.

    Xception uses SeparableConv2D extensively, therefore those
    layers receive a strong preference.

    The final activation after the last SeparableConv2D is
    also a valid feature map and receives a high score.
    """

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

    score = 0.0

    # --------------------------------------------------------
    # Strong preference for convolutional feature layers.
    # --------------------------------------------------------

    if "separableconv2d" in class_name:

        score += 1000.0

    elif "conv2d" in class_name:

        score += 950.0

    elif "depthwiseconv2d" in class_name:

        score += 900.0

    elif "conv" in class_name:

        score += 800.0

    # --------------------------------------------------------
    # Activation following convolution.
    # --------------------------------------------------------

    if (
        "activation" in class_name
        or
        "relu" in class_name
    ):

        score += 500.0

    # --------------------------------------------------------
    # Xception-specific naming.
    # --------------------------------------------------------

    if "block14" in combined:

        score += 500.0

    elif "block13" in combined:

        score += 450.0

    elif "block12" in combined:

        score += 400.0

    elif "block11" in combined:

        score += 350.0

    # --------------------------------------------------------
    # Separable convolution names.
    # --------------------------------------------------------

    if "sepconv" in combined:

        score += 250.0

    # --------------------------------------------------------
    # Avoid pooling / flatten / dense layers.
    # --------------------------------------------------------

    if (
        "pool" in combined
        or
        "flatten" in combined
        or
        "dense" in combined
    ):

        score -= 500.0

    # --------------------------------------------------------
    # Deeper layers are preferred.
    # --------------------------------------------------------

    score += (
        float(depth)
        * 10.0
    )

    # --------------------------------------------------------
    # Later Xception blocks are preferred.
    # --------------------------------------------------------

    import re

    block_match = re.search(
        r"block(\d+)",
        combined
    )

    if block_match:

        try:

            block_number = int(
                block_match.group(1)
            )

            score += (
                block_number
                * 20.0
            )

        except Exception:

            pass

    return score


# ============================================================
# FIND BEST GRAD-CAM++ TARGET
# ============================================================

def find_gradcam_target_layer(
    model
):
    """
    Robustly finds the deepest useful 4-D feature layer.

    Unlike the previous implementation, this function does not
    assume that the Xception backbone is directly exposed by a
    layer named 'xception'.

    It recursively inspects every nested model.
    """

    candidates = []

    all_layers = _walk_model_layers(
        model
    )

    for layer, depth in all_layers:

        if not _layer_has_4d_output(
            layer
        ):

            continue

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

        score = _gradcam_layer_score(
            layer,
            depth
        )

        # ----------------------------------------------------
        # Prefer layers that are clearly convolutional.
        # ----------------------------------------------------

        if (
            "conv" in class_name
            or
            "activation" in class_name
            or
            "relu" in class_name
        ):

            score += 100.0

        candidates.append(
            (
                score,
                depth,
                layer_name,
                layer
            )
        )

    if not candidates:

        # ----------------------------------------------------
        # Second fallback:
        # inspect model layers using their input/output
        # specifications.
        # ----------------------------------------------------

        for layer in model.layers:

            try:

                output_shape = (
                    layer.output_shape
                )

                rank = _safe_rank_from_shape(
                    output_shape
                )

                if rank == 4:

                    candidates.append(
                        (
                            100.0,
                            0,
                            getattr(
                                layer,
                                "name",
                                "unknown"
                            ),
                            layer
                        )
                    )

            except Exception:

                continue

    if not candidates:

        raise ValueError(
            "Grad-CAM++ could not find a suitable "
            "4-D convolutional feature layer.\n\n"
            "The pneumonia model does not expose a usable "
            "intermediate convolutional feature map."
        )

    # --------------------------------------------------------
    # Highest scoring layer.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        ),
        reverse=True
    )

    selected_layer = (
        candidates[0][3]
    )

    return selected_layer


# ============================================================
# FIND PARENT MODEL OF TARGET LAYER
# ============================================================

def _find_parent_model_for_layer(
    model,
    target_layer,
    visited=None
):
    """
    Finds the nested model that directly contains the target
    layer.
    """

    if visited is None:

        visited = set()

    model_id = id(model)

    if model_id in visited:

        return None

    visited.add(
        model_id
    )

    try:

        layers = list(
            model.layers
        )

    except Exception:

        return None

    for layer in layers:

        if layer is target_layer:

            return model

        if isinstance(
            layer,
            tf.keras.Model
        ):

            found = (
                _find_parent_model_for_layer(
                    layer,
                    target_layer,
                    visited
                )
            )

            if found is not None:

                return found

    return None


# ============================================================
# PRIMARY TENSOR HELPER
# ============================================================

def _primary_tensor(
    value
):
    """
    Returns the first tensor when a model has multiple outputs.
    """

    if isinstance(
        value,
        (list, tuple)
    ):

        if len(value) == 0:

            return None

        return _primary_tensor(
            value[0]
        )

    return value


# ============================================================
# BUILD FLAT GRAD-CAM GRAPH
# ============================================================

def _try_build_direct_gradcam_model(
    model,
    target_layer
):
    """
    Attempts to build:

        model input
             ↓
        target feature map
             ↓
        classification output

    directly from the original Functional graph.
    """

    try:

        target_output = (
            _primary_tensor(
                target_layer.output
            )
        )

        model_output = (
            _primary_tensor(
                model.output
            )
        )

        if (
            target_output is None
            or
            model_output is None
        ):

            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                target_output,
                model_output
            ],
            name="gradcam_direct_model"
        )

        # ----------------------------------------------------
        # Test graph.
        # ----------------------------------------------------

        test_input = tf.zeros(
            [
                1,
                PNEUMONIA_IMAGE_SIZE[0],
                PNEUMONIA_IMAGE_SIZE[1],
                3
            ],
            dtype=tf.float32
        )

        test_features, test_prediction = (
            grad_model(
                test_input,
                training=False
            )
        )

        if (
            _safe_rank_from_shape(
                test_features.shape
            )
            != 4
        ):

            return None

        if (
            _safe_rank_from_shape(
                test_prediction.shape
            )
            != 2
        ):

            return None

        return grad_model

    except Exception:

        return None


# ============================================================
# BUILD NESTED GRAD-CAM GRAPH
# ============================================================

def _build_nested_gradcam_graph(
    model,
    target_layer
):
    """
    Correctly builds a differentiable graph for a nested
    Xception model.

    The important point is that:

        target feature map
               ↓
        backbone output
               ↓
        classifier head
               ↓
        class score

    are all calculated within the SAME graph.

    This fixes the main problem in the previous implementation.
    """

    parent_model = (
        _find_parent_model_for_layer(
            model,
            target_layer
        )
    )

    if parent_model is None:

        return None

    # --------------------------------------------------------
    # Target layer output.
    # --------------------------------------------------------

    target_output = (
        _primary_tensor(
            target_layer.output
        )
    )

    if target_output is None:

        return None

    # --------------------------------------------------------
    # Parent model input/output.
    # --------------------------------------------------------

    parent_input = (
        _primary_tensor(
            parent_model.input
        )
    )

    parent_output = (
        _primary_tensor(
            parent_model.output
        )
    )

    if (
        parent_input is None
        or
        parent_output is None
    ):

        return None

    # --------------------------------------------------------
    # Check that parent output is connected to outer model.
    #
    # Example:
    #
    # input
    #   ↓
    # Xception
    #   ↓
    # GAP
    #   ↓
    # Dense
    #   ↓
    # output
    # --------------------------------------------------------

    try:

        head_model = tf.keras.models.Model(
            inputs=parent_output,
            outputs=_primary_tensor(
                model.output
            ),
            name="gradcam_classifier_head"
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Build feature model that returns both:
    #
    # 1. target convolutional feature
    # 2. parent Xception output
    #
    # Both originate from the SAME backbone execution.
    # --------------------------------------------------------

    try:

        feature_model = tf.keras.models.Model(
            inputs=parent_input,
            outputs=[
                target_output,
                parent_output
            ],
            name="gradcam_nested_feature_model"
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Test nested graph.
    # --------------------------------------------------------

    try:

        test_input = tf.zeros(
            [
                1,
                PNEUMONIA_IMAGE_SIZE[0],
                PNEUMONIA_IMAGE_SIZE[1],
                3
            ],
            dtype=tf.float32
        )

        test_features, test_backbone_output = (
            feature_model(
                test_input,
                training=False
            )
        )

        if (
            _safe_rank_from_shape(
                test_features.shape
            )
            != 4
        ):

            return None

        test_prediction = head_model(
            test_backbone_output,
            training=False
        )

        if (
            _safe_rank_from_shape(
                test_prediction.shape
            )
            != 2
        ):

            return None

        return {
            "feature_model": feature_model,
            "head_model": head_model,
            "parent_model": parent_model
        }

    except Exception:

        return None


# ============================================================
# BUILD COMPLETE GRAD-CAM GRAPH
# ============================================================

def _build_gradcam_graph(
    model,
    target_layer
):
    """
    Attempts the safest graph construction methods in order.
    """

    # --------------------------------------------------------
    # Method 1:
    # Direct outer graph.
    # --------------------------------------------------------

    direct_model = (
        _try_build_direct_gradcam_model(
            model,
            target_layer
        )
    )

    if direct_model is not None:

        return {
            "type": "direct",
            "model": direct_model
        }

    # --------------------------------------------------------
    # Method 2:
    # Nested backbone + classifier head.
    # --------------------------------------------------------

    nested_graph = (
        _build_nested_gradcam_graph(
            model,
            target_layer
        )
    )

    if nested_graph is not None:

        return {
            "type": "nested",
            **nested_graph
        }

    raise RuntimeError(
        "Grad-CAM++ found a 4-D feature layer but "
        "could not connect that feature layer to the "
        "pneumonia classification output."
    )


# ============================================================
# GRAD-CAM++ CORE CALCULATION
# ============================================================

def _calculate_gradcam_plus_plus(
    conv_features,
    first_derivative,
    second_derivative,
    third_derivative
):
    """
    Grad-CAM++ weight calculation.

    Inputs:
        conv_features:
            [batch, height, width, channels]

        derivatives:
            Same spatial/channel dimensions.
    """

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

    epsilon = tf.constant(
        1e-8,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Positive second derivative.
    # --------------------------------------------------------

    positive_second = tf.maximum(
        second_derivative,
        0.0
    )

    # --------------------------------------------------------
    # Grad-CAM++ alpha.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Positive gradients.
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
    # Positive activations only.
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
        heatmap_max > epsilon,
        heatmap / (
            heatmap_max
            + epsilon
        ),
        tf.zeros_like(
            heatmap
        )
    )

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
# GRAD-CAM++ GENERATION
# ============================================================

def generate_gradcam_plus_plus(
    image,
    target_class_index=1
):
    """
    Generates Grad-CAM++ for:

        0 = Normal
        1 = Pneumonia

    This function correctly maintains the computational
    connection between the selected feature map and the
    pneumonia classification output.
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
        getattr(
            target_layer,
            "name",
            "unknown"
        )
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
    # BUILD GRAPH
    # ========================================================

    graph = _build_gradcam_graph(
        pneumonia_model,
        target_layer
    )

    graph_type = graph[
        "type"
    ]

    # ========================================================
    # DIRECT MODEL
    # ========================================================

    if graph_type == "direct":

        grad_model = graph[
            "model"
        ]

        # ----------------------------------------------------
        # Nested persistent tapes are required because
        # Grad-CAM++ uses first, second and third derivatives.
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

                    conv_features, predictions = (
                        grad_model(
                            image_tensor,
                            training=False
                        )
                    )

                    class_score = predictions[
                        :,
                        target_class_index
                    ]

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

    elif graph_type == "nested":

        feature_model = graph[
            "feature_model"
        ]

        head_model = graph[
            "head_model"
        ]

        # ----------------------------------------------------
        # CRITICAL:
        #
        # feature_model produces:
        #
        #     target feature map
        #     backbone output
        #
        # Then head_model produces the prediction FROM THAT
        # SAME backbone output.
        #
        # Therefore gradients flow correctly:
        #
        # prediction
        #     ↓
        # backbone output
        #     ↓
        # target feature map
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

                    conv_features, backbone_output = (
                        feature_model(
                            image_tensor,
                            training=False
                        )
                    )

                    predictions = head_model(
                        backbone_output,
                        training=False
                    )

                    predictions = tf.convert_to_tensor(
                        predictions
                    )

                    class_score = predictions[
                        :,
                        target_class_index
                    ]

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

    else:

        raise RuntimeError(
            "Unknown Grad-CAM++ graph type."
        )

    # ========================================================
    # GRADIENT VALIDATION
    # ========================================================

    if first_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the first "
            "gradient for the selected feature layer.\n\n"
            f"Target layer: {target_layer_name}"
        )

    if second_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the second "
            "gradient for the selected feature layer.\n\n"
            f"Target layer: {target_layer_name}"
        )

    if third_derivative is None:

        raise RuntimeError(
            "Grad-CAM++ could not calculate the third "
            "gradient for the selected feature layer.\n\n"
            f"Target layer: {target_layer_name}"
        )

    # ========================================================
    # FEATURE SHAPE VALIDATION
    # ========================================================

    if (
        _safe_rank_from_shape(
            conv_features.shape
        )
        != 4
    ):

        raise RuntimeError(
            "The selected Grad-CAM++ feature activation "
            "is not 4-D.\n\n"
            f"Received shape: {conv_features.shape}\n"
            f"Target layer: {target_layer_name}"
        )

    # ========================================================
    # CALCULATE HEATMAP
    # ========================================================

    heatmap = (
        _calculate_gradcam_plus_plus(
            conv_features,
            first_derivative,
            second_derivative,
            third_derivative
        )
    )

    # ========================================================
    # CHECK HEATMAP
    # ========================================================

    if heatmap.size == 0:

        raise RuntimeError(
            "Grad-CAM++ produced an empty heatmap."
        )

    if not np.isfinite(
        heatmap
    ).any():

        raise RuntimeError(
            "Grad-CAM++ produced an invalid heatmap."
        )

    maximum = float(
        np.max(
            heatmap
        )
    )

    if maximum <= 1e-7:

        raise RuntimeError(
            "Grad-CAM++ produced an almost-zero "
            "localization map.\n\n"
            f"Target layer: {target_layer_name}"
        )

    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    heatmap = (
        heatmap
        /
        (
            maximum
            + 1e-8
        )
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    return (
        heatmap,
        target_layer_name
    )


# ============================================================
# IMPROVE HEATMAP CONTRAST
# ============================================================

def enhance_gradcam_heatmap(
    heatmap
):
    """
    Improves visual contrast.

    The raw Grad-CAM++ map often contains a large number of
    weak activations. Simply resizing that raw map makes the
    localization look washed out.

    This function:
        1. Removes the weakest activation range.
        2. Stretches the useful activation range.
        3. Applies gamma correction.
        4. Clips very weak areas.
    """

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

    maximum = float(
        np.max(
            heatmap
        )
    )

    if maximum <= 1e-8:

        return np.zeros_like(
            heatmap
        )

    # --------------------------------------------------------
    # Percentile-based contrast.
    # --------------------------------------------------------

    positive_values = (
        heatmap[
            heatmap > 0
        ]
    )

    if positive_values.size > 10:

        low_value = float(
            np.percentile(
                positive_values,
                GRADCAM_LOW_PERCENTILE
            )
        )

        high_value = float(
            np.percentile(
                positive_values,
                GRADCAM_HIGH_PERCENTILE
            )
        )

    else:

        low_value = 0.0

        high_value = maximum

    if (
        high_value
        <=
        low_value + 1e-8
    ):

        low_value = 0.0

        high_value = maximum

    # --------------------------------------------------------
    # Contrast stretch.
    # --------------------------------------------------------

    enhanced = (
        heatmap
        -
        low_value
    ) / (
        high_value
        -
        low_value
        +
        1e-8
    )

    enhanced = np.clip(
        enhanced,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Gamma correction.
    # --------------------------------------------------------

    enhanced = np.power(
        enhanced,
        GRADCAM_GAMMA
    )

    # --------------------------------------------------------
    # Remove weak activation.
    # --------------------------------------------------------

    threshold = (
        GRADCAM_MIN_ACTIVATION
    )

    weak_mask = (
        enhanced
        <
        threshold
    )

    enhanced[
        weak_mask
    ] = 0.0

    # --------------------------------------------------------
    # Re-normalize after threshold.
    # --------------------------------------------------------

    final_max = float(
        np.max(
            enhanced
        )
    )

    if final_max > 1e-8:

        enhanced /= (
            final_max
            + 1e-8
        )

    return np.clip(
        enhanced,
        0.0,
        1.0
    )


# ============================================================
# HEATMAP COLORIZATION
# ============================================================

def colorize_gradcam_heatmap(
    heatmap
):
    """
    Converts a normalized heatmap to a high-contrast
    blue -> cyan -> green -> yellow -> red visualization.

    Red represents the strongest activation.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # HSV-like color interpolation.
    #
    # Hue:
    #
    # 0.66 = blue
    # 0.50 = cyan
    # 0.33 = green
    # 0.16 = yellow
    # 0.00 = red
    # --------------------------------------------------------

    hue = (
        0.66
        *
        (
            1.0
            -
            heatmap
        )
    )

    saturation = np.ones_like(
        heatmap
    )

    value = np.ones_like(
        heatmap
    )

    h = (
        hue
        * 6.0
    )

    sector = np.floor(
        h
    ).astype(
        np.int32
    )

    fraction = (
        h
        -
        np.floor(h)
    )

    p = (
        value
        *
        (
            1.0
            -
            saturation
        )
    )

    q = (
        value
        *
        (
            1.0
            -
            saturation
            *
            fraction
        )
    )

    t = (
        value
        *
        (
            1.0
            -
            saturation
            *
            (
                1.0
                -
                fraction
            )
        )
    )

    r = np.zeros_like(
        heatmap
    )

    g = np.zeros_like(
        heatmap
    )

    b = np.zeros_like(
        heatmap
    )

    mask = (
        sector == 0
    )

    r[mask] = value[mask]
    g[mask] = t[mask]
    b[mask] = p[mask]

    mask = (
        sector == 1
    )

    r[mask] = q[mask]
    g[mask] = value[mask]
    b[mask] = p[mask]

    mask = (
        sector == 2
    )

    r[mask] = p[mask]
    g[mask] = value[mask]
    b[mask] = t[mask]

    mask = (
        sector == 3
    )

    r[mask] = p[mask]
    g[mask] = q[mask]
    b[mask] = value[mask]

    mask = (
        sector == 4
    )

    r[mask] = t[mask]
    g[mask] = p[mask]
    b[mask] = value[mask]

    mask = (
        sector >= 5
    )

    r[mask] = value[mask]
    g[mask] = p[mask]
    b[mask] = q[mask]

    rgb = np.stack(
        [
            r,
            g,
            b
        ],
        axis=-1
    )

    rgb = (
        np.clip(
            rgb,
            0.0,
            1.0
        )
        * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb,
        mode="RGB"
    )


# ============================================================
# CREATE GRAD-CAM++ OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):
    """
    Creates three images:

        1. Original image
        2. Standalone Grad-CAM++ heatmap
        3. High-contrast Grad-CAM++ overlay

    The heatmap is resized using bicubic interpolation and
    lightly smoothed to avoid blocky 7x7 Xception maps.
    """

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original = ImageOps.exif_transpose(
        image
    ).convert(
        "RGB"
    )

    # ========================================================
    # ENHANCE HEATMAP
    # ========================================================

    enhanced_heatmap = (
        enhance_gradcam_heatmap(
            heatmap
        )
    )

    # ========================================================
    # CONVERT HEATMAP TO IMAGE
    # ========================================================

    heatmap_image = colorize_gradcam_heatmap(
        enhanced_heatmap
    )

    # ========================================================
    # RESIZE TO ORIGINAL IMAGE
    # ========================================================

    heatmap_image = (
        heatmap_image.resize(
            original.size,
            Image.Resampling.BICUBIC
        )
    )

    # ========================================================
    # SMOOTH VERY SLIGHTLY
    # ========================================================

    heatmap_image = (
        heatmap_image.filter(
            ImageFilter.GaussianBlur(
                radius=GRADCAM_BLUR_RADIUS
            )
        )
    )

    # ========================================================
    # RESIZE HEATMAP MASK
    # ========================================================

    mask_image = Image.fromarray(
        (
            enhanced_heatmap
            * 255.0
        ).astype(
            np.uint8
        ),
        mode="L"
    )

    mask_image = (
        mask_image.resize(
            original.size,
            Image.Resampling.BICUBIC
        )
    )

    mask_array = np.asarray(
        mask_image,
        dtype=np.float32
    ) / 255.0

    # ========================================================
    # STRONGER ALPHA IN HIGH-ACTIVATION AREAS
    # ========================================================

    alpha_array = np.power(
        np.clip(
            mask_array,
            0.0,
            1.0
        ),
        0.75
    )

    alpha_array *= (
        GRADCAM_OVERLAY_ALPHA
    )

    # --------------------------------------------------------
    # Completely suppress very weak regions.
    # --------------------------------------------------------

    alpha_array[
        mask_array < 0.10
    ] = 0.0

    alpha_image = Image.fromarray(
        (
            np.clip(
                alpha_array,
                0.0,
                1.0
            )
            * 255.0
        ).astype(
            np.uint8
        ),
        mode="L"
    )

    # ========================================================
    # CREATE RGBA HEATMAP
    # ========================================================

    heatmap_rgba = (
        heatmap_image.convert(
            "RGBA"
        )
    )

    heatmap_rgba.putalpha(
        alpha_image
    )

    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    original_rgba = (
        original.convert(
            "RGBA"
        )
    )

    overlay = Image.alpha_composite(
        original_rgba,
        heatmap_rgba
    )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        original,
        heatmap_image,
        overlay.convert(
            "RGB"
        )
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

            # ------------------------------------------------
            # Keep image within A4 page width.
            # ------------------------------------------------

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
        #
        # WEB INTERFACE SHOWS ONLY THE DETECTED CLASS.
        # ====================================================

        st.markdown(
            "## Detected Medical Image Type"
        )

        st.markdown(
            f"### {modality}"
        )


        # ====================================================
        # NO MODALITY PROBABILITIES ON WEB INTERFACE
        # NO TECHNICAL INFORMATION ON WEB INTERFACE
        # ====================================================


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
            # WEB INTERFACE SHOWS ONLY THE CLASS.
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
            # STEP 6 — GRAD-CAM++ LOCALIZATION
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

                        heatmap, gradcam_layer_name = (
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
                # DISPLAY GRAD-CAM++ RESULT
                # ------------------------------------------------

                if gradcam_overlay is not None:

                    st.success(
                        "Pneumonia localization generated "
                        "using Grad-CAM++."
                    )

                    # ------------------------------------------------
                    # Display original + standalone heatmap +
                    # final overlay.
                    #
                    # This makes it much easier to verify whether
                    # the heatmap is actually meaningful.
                    # ------------------------------------------------

                    gradcam_col1, gradcam_col2 = st.columns(
                        2
                    )

                    with gradcam_col1:

                        st.image(
                            heatmap_image,
                            caption=(
                                "Grad-CAM++ Activation Heatmap"
                            ),
                            use_container_width=True
                        )

                    with gradcam_col2:

                        st.image(
                            gradcam_overlay,
                            caption=(
                                "Grad-CAM++ Pneumonia "
                                "Localization"
                            ),
                            use_container_width=True
                        )

                    st.info(
                        "Red and yellow regions indicate the "
                        "strongest areas associated with the "
                        "Pneumonia prediction. The visualization "
                        "is an explainability map, not a clinical "
                        "segmentation."
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
