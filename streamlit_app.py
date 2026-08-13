# ============================================================
# streamlit_app.py
# Pneumonia Detection System
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Basic Validation
#       ↓
# Grayscale / Color Validation
#       ↓
# 3-Class Medical Image Modality Classifier
#       ↓
# ┌───────────────┬───────────────┬───────────────┐
# │ Chest X-ray   │ CT            │ MRI           │
# │ Continue      │ Reject        │ Reject        │
# └───────────────┴───────────────┴───────────────┘
#       ↓
# Pneumonia Model
#       ↓
# Normal / Pneumonia
#
# ============================================================


import os
import io

import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from fpdf import FPDF

from model_builder import build_model
from modality_model_builder import build_modality_classifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia AI",
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
# MODEL FILES
# ============================================================
#
# These filenames correspond to the files in your GitHub
# repository.
#
# Repository:
# Pneumonia_Detection_System
#
# ============================================================

@st.cache_resource
def load_modality_model():

    validate_model_file(
        MODALITY_MODEL_PATH,
        "Medical image modality classifier weights"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_modality_classifier(
        input_shape=(224, 224, 3),
        num_classes=3
    )

    # --------------------------------------------------------
    # NEVER allow None
    # --------------------------------------------------------

    if model is None:
        raise RuntimeError(
            "build_modality_classifier() returned None. "
            "Check modality_model_builder.py."
        )

    # --------------------------------------------------------
    # Verify architecture
    # --------------------------------------------------------

    if model.output_shape[-1] != 3:
        raise ValueError(
            "Modality classifier must have exactly "
            "3 outputs. "
            f"Received: {model.output_shape}"
        )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    try:

        model.load_weights(
            MODALITY_MODEL_PATH
        )

    except Exception as e:

        raise RuntimeError(
            "\n\n"
            "MODALITY MODEL WEIGHT LOADING FAILED.\n\n"
            "The file exists, but its architecture does not "
            "match modality_model_builder.py.\n\n"
            f"Weight file:\n{MODALITY_MODEL_PATH}\n\n"
            f"Original error:\n{e}"
        ) from e

    return model

# ============================================================
# IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (
    224,
    224
)


PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# THRESHOLDS
# ============================================================

COLOR_TOLERANCE = 5.0

MODALITY_CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# CLASS MAPPINGS
# ============================================================

# ------------------------------------------------------------
# MODALITY CLASSIFIER
# ------------------------------------------------------------
#
# IMPORTANT:
#
# This MUST match the class_indices used during training.
#
# 0 = Chest X-ray
# 1 = CT
# 2 = MRI
#
# ------------------------------------------------------------

MODALITY_CLASS_MAP = {

    0: "CHEST_XRAY",

    1: "CT",

    2: "MRI"
}


MODALITY_XRAY_CLASS_INDEX = 0


# ------------------------------------------------------------
# PNEUMONIA CLASSIFIER
# ------------------------------------------------------------
#
# Expected:
#
# 0 = Normal
# 1 = Pneumonia
#
# ------------------------------------------------------------

PNEUMONIA_CLASS_MAP = {

    0: "Normal",

    1: "Pneumonia"
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_model_file(
    path,
    model_name
):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"{model_name} not found:\n"
            f"{path}\n\n"
            f"Make sure the model file is in the "
            f"same GitHub directory as streamlit_app.py."
        )


    if os.path.getsize(path) == 0:

        raise ValueError(
            f"{model_name} exists but is empty:\n"
            f"{path}"
        )


# ============================================================
# LOAD MODALITY CLASSIFIER
# ============================================================

@st.cache_resource
def load_modality_model():

    validate_model_file(
        MODALITY_MODEL_PATH,
        "Medical image modality classifier weights"
    )


    # --------------------------------------------------------
    # Build exactly the same architecture used during
    # modality-classifier training.
    # --------------------------------------------------------

    model = build_modality_classifier(
        input_shape=(224, 224, 3),
        num_classes=3
    )


    # --------------------------------------------------------
    # Load trained weights.
    # --------------------------------------------------------

    model.load_weights(
        MODALITY_MODEL_PATH
    )


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


    # --------------------------------------------------------
    # The repository contains the complete .keras model.
    #
    # First try loading it directly.
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

        return model


    except Exception as complete_model_error:

        # ----------------------------------------------------
        # If the .keras file is actually weights-only,
        # fall back to rebuilding the architecture.
        # ----------------------------------------------------

        try:

            model = build_model(
                input_shape=(224, 224, 3),
                num_classes=2
            )


            model.load_weights(
                PNEUMONIA_MODEL_PATH
            )


            return model


        except Exception as weights_error:

            raise RuntimeError(
                "Unable to load the pneumonia model.\n\n"
                "Complete-model loading error:\n"
                f"{complete_model_error}\n\n"
                "Weights-loading error:\n"
                f"{weights_error}"
            )


# ============================================================
# LOAD ALL REQUIRED MODELS
# ============================================================

try:

    modality_model = load_modality_model()

    pneumonia_model = load_pneumonia_model()


except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY MODEL OUTPUT SHAPES
# ============================================================

# ------------------------------------------------------------
# Modality model
# ------------------------------------------------------------

if modality_model.output_shape[-1] != 3:

    st.error(
        "The modality classifier must have exactly "
        "3 output classes."
    )

    st.write(
        f"Received output shape: "
        f"{modality_model.output_shape}"
    )

    st.stop()


# ------------------------------------------------------------
# Pneumonia model
# ------------------------------------------------------------

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
# CONVERT SCORES TO PROBABILITIES
# ============================================================

def convert_to_probabilities(
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Already softmax probabilities
    # --------------------------------------------------------

    if (

        np.all(
            scores >= 0.0
        )

        and

        np.all(
            scores <= 1.0
        )

        and

        np.isclose(
            np.sum(scores),
            1.0,
            atol=1e-3
        )

    ):

        return scores


    # --------------------------------------------------------
    # Otherwise treat as logits.
    # --------------------------------------------------------

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# IMAGE COLOR CHECK
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
        > COLOR_TOLERANCE
    )


    return (
        is_color,
        float(
            average_difference
        )
    )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(
    image
):

    width, height = image.size


    # --------------------------------------------------------
    # Minimum resolution
    # --------------------------------------------------------

    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )


    # --------------------------------------------------------
    # Empty image
    # --------------------------------------------------------

    array = np.asarray(
        image
    )


    if array.size == 0:

        return (
            False,
            "Image is empty."
        )


    # --------------------------------------------------------
    # Color check
    # --------------------------------------------------------

    is_color, color_difference = (
        check_color_image(
            image
        )
    )


    if is_color:

        return (
            False,
            "Color image detected."
        )


    # --------------------------------------------------------
    # Grayscale image
    # --------------------------------------------------------

    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Blank image
    # --------------------------------------------------------

    if np.std(gray) < 8:

        return (
            False,
            "Image appears blank or invalid."
        )


    # --------------------------------------------------------
    # Almost completely black
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "Image is almost completely black."
        )


    # --------------------------------------------------------
    # Almost completely white
    # --------------------------------------------------------

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
        "Image passed basic validation."
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image,
    target_size
):

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
# MODALITY PREDICTION
# ============================================================

def predict_modality(
    image
):

    image_array = preprocess_image(
        image,
        MODALITY_IMAGE_SIZE
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


    modality = (
        MODALITY_CLASS_MAP.get(
            predicted_index,
            "UNKNOWN"
        )
    )


    return {

        "index":
            predicted_index,

        "class":
            modality,

        "confidence":
            confidence,

        "probabilities":
            probabilities
    }


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
        PNEUMONIA_CLASS_MAP.get(
            predicted_index,
            "UNKNOWN"
        )
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

        "class":
            predicted_class,

        "confidence":
            confidence,

        "normal_probability":
            normal_probability,

        "pneumonia_probability":
            pneumonia_probability,

        "probabilities":
            probabilities
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)


st.markdown(
    """
### AI Processing Pipeline

**Uploaded Image**
↓  
**Basic Validation**
↓  
**Grayscale Verification**
↓  
**3-Class Medical Image Modality Classification**
↓  
**Chest X-ray / CT / MRI**
↓  
**Pneumonia Classification**
↓  
**Normal / Pneumonia**
"""
)


st.info(
    "Only grayscale chest X-ray images are accepted. "
    "Color images, CT scans, MRI images, and other "
    "medical image modalities are rejected."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Pneumonia AI Pipeline"
    )


    st.write(
        """
        Image Upload
        ↓
        Image Validation
        ↓
        Modality Classifier
        ↓
        Chest X-ray Verification
        ↓
        Pneumonia Detection
        """
    )


    st.divider()


    st.write(
        "**Modality Classes**"
    )


    st.write(
        "0 — Chest X-ray"
    )


    st.write(
        "1 — CT"
    )


    st.write(
        "2 — MRI"
    )


    st.divider()


    st.write(
        "**Pneumonia Classes**"
    )


    st.write(
        "0 — Normal"
    )


    st.write(
        "1 — Pneumonia"
    )


    st.divider()


    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Modality classifier:**"
    )

    st.code(
        "best_modality_classifier.weights.h5"
    )


    st.write(
        "**Pneumonia model:**"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )


    st.divider()


    st.write(
        "**Modality input:**"
    )

    st.write(
        "224 × 224 × 3"
    )


    st.write(
        "**Pneumonia input:**"
    )

    st.write(
        "224 × 224 × 3"
    )


    st.divider()


    st.write(
        "**Modality mapping:**"
    )

    st.code(
        """
0 = CHEST_XRAY
1 = CT
2 = MRI
"""
    )


    st.write(
        "**Pneumonia mapping:**"
    )

    st.code(
        """
0 = Normal
1 = Pneumonia
"""
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    help=(
        "Upload a grayscale chest X-ray image."
    )
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )


        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )


        image.load()


    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    is_valid, validation_message = (
        validate_image(
            image
        )
    )


    if not is_valid:

        st.error(
            f"❌ {validation_message}"
        )


        if (
            validation_message
            == "Color image detected."
        ):

            st.warning()
