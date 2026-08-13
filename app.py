# ============================================================
# app.py
# Pneumonia Detection System
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Basic Validation
#       ↓
# Color Image Rejection
#       ↓
# 3-Class Modality Classifier
#       ↓
# ┌───────────────┬───────────────┬───────────────┐
# │ Chest X-ray   │ CT            │ MRI           │
# │ Continue      │ Reject        │ Reject        │
# └───────────────┴───────────────┴───────────────┘
#       ↓
# 2-Class X-ray Verifier
#       ↓
# ┌───────────────┬───────────────┐
# │ X-RAY         │ NON-XRAY      │
# │ Continue      │ Reject        │
# └───────────────┴───────────────┘
#       ↓
# Pneumonia Model
#       ↓
# Normal / Pneumonia
#
# ============================================================


import os
import io

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from fpdf import FPDF

from model_builder import build_model
from xray_model_builder import build_xray_classifier
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XRAY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xray_verifier.weights.h5"
)

PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_exception_pneumonia_model.keras"
)


# ============================================================
# MODEL PATHS
# ============================================================

XRAY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xray_verifier.weights.h5"
)


PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xception_pneumonia_model.keras"
)


MODALITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_modality_classifier.weights.h5"
)


# ============================================================
# IMAGE SIZES
# ============================================================

XRAY_IMAGE_SIZE = (
    128,
    128
)


PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


MODALITY_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# IMAGE VALIDATION SETTINGS
# ============================================================

# Difference between RGB channels.
#
# If the average channel difference is above this value,
# the image is treated as a color image and rejected.

COLOR_TOLERANCE = 5.0


# Minimum confidence required from the 3-class
# modality classifier.

MODALITY_CONFIDENCE_THRESHOLD = 0.90


# Minimum confidence required from the
# 2-class X-ray verifier.

XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# CLASS MAPPINGS
# ============================================================

# ------------------------------------------------------------
# X-RAY VERIFIER
# ------------------------------------------------------------
#
# According to your xray_model_builder.py:
#
# Dense(2, activation="softmax")
#
# Therefore:
#
# 0 = X-RAY
# 1 = NON-XRAY
#

XRAY_CLASS_MAP = {

    0: "X-RAY",

    1: "NON-XRAY"
}


XRAY_CLASS_INDEX = 0

NON_XRAY_CLASS_INDEX = 1


# ------------------------------------------------------------
# MODALITY CLASSIFIER
# ------------------------------------------------------------
#
# IMPORTANT:
#
# This version assumes your modality classifier
# was trained with THREE classes:
#
# 0 = Chest X-ray
# 1 = CT
# 2 = MRI
#
# The training class_indices MUST match this mapping.
#

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
            f"{model_name} not found:\n{path}\n\n"
            f"Place the required model file in the "
            f"same folder as app.py."
        )


    if os.path.getsize(path) == 0:

        raise ValueError(
            f"{model_name} exists but is empty:\n{path}"
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
    # Build architecture
    # --------------------------------------------------------

    model = build_modality_classifier(
        input_shape=(224, 224, 3),
        num_classes=3
    )


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_weights(
        MODALITY_MODEL_PATH
    )


    return model


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    validate_model_file(
        XRAY_MODEL_PATH,
        "X-ray verifier weights"
    )


    # --------------------------------------------------------
    # Build exactly the same architecture used during training
    # --------------------------------------------------------

    model = build_xray_classifier(
        input_shape=(128, 128, 3)
    )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_weights(
        XRAY_MODEL_PATH
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
    # First try loading as a complete Keras model.
    #
    # This is appropriate if:
    #
    # model.save("best_xception_pneumonia_model.keras")
    #
    # was used during training.
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

        return model

    except Exception:

        pass


    # --------------------------------------------------------
    # If the .keras file is actually weights-only,
    # build the architecture and load weights.
    # --------------------------------------------------------

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=2
    )


    model.load_weights(
        PNEUMONIA_MODEL_PATH
    )


    return model


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    modality_model = load_modality_model()

    xray_model = load_xray_model()

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
# Modality classifier
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
# X-ray verifier
# ------------------------------------------------------------

if xray_model.output_shape[-1] != 2:

    st.error(
        "The X-ray verifier must have exactly "
        "2 output classes."
    )

    st.write(
        f"Received output shape: "
        f"{xray_model.output_shape}"
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
# UTILITY — CONVERT SCORES TO PROBABILITIES
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
    # Otherwise treat as logits
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

    # --------------------------------------------------------
    # Convert to RGB only for channel comparison.
    # --------------------------------------------------------

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )


    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]


    # --------------------------------------------------------
    # Average difference between channels
    # --------------------------------------------------------

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
    # Resolution
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


    if (
        array.size == 0
    ):

        return (
            False,
            "Image is empty."
        )


    # --------------------------------------------------------
    # Check for color image
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
    # Convert to grayscale
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
# X-RAY VERIFICATION
# ============================================================

def verify_xray(
    image
):

    image_array = preprocess_image(
        image,
        XRAY_IMAGE_SIZE
    )


    prediction = xray_model.predict(
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
            "X-ray verifier must output "
            f"2 classes. Received: "
            f"{prediction.shape}"
        )


    probabilities = (
        convert_to_probabilities(
            prediction[0]
        )
    )


    xray_probability = float(
        probabilities[
            XRAY_CLASS_INDEX
        ]
    )


    non_xray_probability = float(
        probabilities[
            NON_XRAY_CLASS_INDEX
        ]
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_class = (
        XRAY_CLASS_MAP.get(
            predicted_index,
            "UNKNOWN"
        )
    )


    # --------------------------------------------------------
    # X-ray acceptance
    # --------------------------------------------------------

    is_xray = (

        predicted_class == "X-RAY"

        and

        xray_probability
        >= XRAY_CONFIDENCE_THRESHOLD

    )


    return {

        "result":
            "X-RAY"
            if is_xray
            else
            "NON-XRAY",

        "is_xray":
            is_xray,

        "xray_probability":
            xray_probability,

        "non_xray_probability":
            non_xray_probability,

        "predicted_index":
            predicted_index,

        "predicted_class":
            predicted_class,

        "confidence":
            (
                xray_probability
                if is_xray
                else
                non_xray_probability
            ),

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
**Color Image Rejection**
↓  
**3-Class Modality Classification**
↓  
**Chest X-ray / CT / MRI**
↓  
**2-Class X-ray Verification**
↓  
**Pneumonia Classification**
↓  
**Normal / Pneumonia**
"""
)


st.info(
    "Only grayscale chest X-ray images are accepted. "
    "Color images, CT scans, MRI images, and non-X-ray "
    "images are rejected."
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
        Color Validation
        ↓
        Modality Classifier
        ↓
        X-ray Verification
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
        "**X-ray Verifier Classes**"
    )


    st.write(
        "0 — X-RAY"
    )


    st.write(
        "1 — NON-XRAY"
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
        "**X-ray verifier:**"
    )

    st.code(
        "best_xray_verifier.weights.h5"
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
        "**X-ray verifier input:**"
    )

    st.write(
        "128 × 128 × 3"
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
        "**X-ray verifier mapping:**"
    )

    st.code(
        """
0 = X-RAY
1 = NON-XRAY
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

            st.warning(
                "Color images are not accepted. "
                "Please upload a grayscale chest X-ray."
            )

        else:

            st.warning(
                "Please upload a valid grayscale "
                "medical image."
            )


        st.stop()


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True
    ):

        # ====================================================
        # STEP 1 — COLOR VALIDATION
        # ====================================================

        st.subheader(
            "Step 1 — Image Validation"
        )


        is_color, color_difference = (
            check_color_image(
                image
            )
        )


        st.write(
            f"RGB channel difference: "
            f"{color_difference:.3f}"
        )


        if is_color:

            st.error(
                "❌ Color image detected."
            )


            st.warning(
                "This system accepts only "
                "grayscale chest X-ray images."
            )


            history_entry = (
                f"Rejected - Color image - "
                f"{uploaded_file.name}"
            )


            if (
                history_entry
                not in st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        st.success(
            "Grayscale image confirmed."
        )


        # ====================================================
        # STEP 2 — MODALITY CLASSIFICATION
        # ====================================================

        st.subheader(
            "Step 2 — Medical Image Modality Verification"
        )


        with st.spinner(
            "Determining image modality..."
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


        modality_class = (
            modality_result[
                "class"
            ]
        )


        modality_confidence = (
            modality_result[
                "confidence"
            ]
        )


        modality_probabilities = (
            modality_result[
                "probabilities"
            ]
        )


        # ----------------------------------------------------
        # DISPLAY MODALITY RESULT
        # ----------------------------------------------------

        st.write(
            f"**Detected modality:** "
            f"{modality_class}"
        )


        st.write(
            f"**Confidence:** "
            f"{modality_confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # MODALITY PROBABILITIES
        # ----------------------------------------------------

        modality_columns = st.columns(3)


        for i, column in enumerate(
            modality_columns
        ):

            with column:

                class_name = (
                    MODALITY_CLASS_MAP[i]
                )


                probability = float(
                    modality_probabilities[i]
                )


                st.metric(
                    class_name,
                    f"{probability * 100:.2f}%"
                )


                st.progress(
                    float(
                        np.clip(
                            probability,
                            0.0,
                            1.0
                        )
                    )
                )


        # ====================================================
        # REJECT CT
        # ====================================================

        if modality_class == "CT":

            st.error(
                "❌ CT scan detected."
            )


            st.warning(
                "This system accepts only "
                "chest X-ray images."
            )


            history_entry = (
                f"Rejected - CT - "
                f"{uploaded_file.name}"
            )


            if (
                history_entry
                not in st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        # ====================================================
        # REJECT MRI
        # ====================================================

        if modality_class == "MRI":

            st.error(
                "❌ MRI image detected."
            )


            st.warning(
                "This system accepts only "
                "chest X-ray images."
            )


            history_entry = (
                f"Rejected - MRI - "
                f"{uploaded_file.name}"
            )


            if (
                history_entry
                not in st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        # ====================================================
        # UNKNOWN MODALITY
        # ====================================================

        if modality_class != "CHEST_XRAY":

            st.error(
                "❌ Unsupported image modality."
            )


            st.stop()


        # ====================================================
        # LOW MODALITY CONFIDENCE
        # ====================================================

        if (
            modality_confidence
            < MODALITY_CONFIDENCE_THRESHOLD
        ):

            st.error(
                "❌ Modality classification confidence "
                "is too low."
            )


            st.warning(
                f"Chest X-ray was predicted, but confidence "
                f"was only "
                f"{modality_confidence * 100:.2f}%. "
                f"Required confidence: "
                f"{MODALITY_CONFIDENCE_THRESHOLD * 100:.0f}%."
            )


            history_entry = (
                f"Rejected - Low modality confidence - "
                f"{uploaded_file.name}"
            )


            if (
                history_entry
                not in st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        # ====================================================
        # CHEST X-RAY CONFIRMED BY MODALITY CLASSIFIER
        # ====================================================

        st.success(
            "✅ Chest X-ray modality detected."
        )


        # ====================================================
        # STEP 3 — X-RAY VERIFIER
        # ====================================================

        st.subheader(
            "Step 3 — Chest X-ray Verification"
        )


        with st.spinner(
            "Verifying chest X-ray..."
        ):

            try:

                xray_result = (
                    verify_xray(
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
            xray_result[
                "xray_probability"
            ]
        )


        non_xray_probability = (
            xray_result[
                "non_xray_probability"
            ]
        )


        xray_confidence = (
            xray_result[
                "confidence"
            ]
        )


        xray_is_valid = (
            xray_result[
                "is_xray"
            ]
        )


        # ----------------------------------------------------
        # X-RAY PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "X-ray Probability",
                f"{xray_probability * 100:.2f}%"
            )


            st.progress(
                float(
                    np.clip(
                        xray_probability,
                        0.0,
                        1.0
                    )
                )
            )


        with col2:

            st.metric(
                "Non-X-ray Probability",
                f"{non_xray_probability * 100:.2f}%"
            )


            st.progress(
                float(
                    np.clip(
                        non_xray_probability,
                        0.0,
                        1.0
                    )
                )
            )


        # ====================================================
        # REJECT NON-X-RAY
        # ====================================================

        if not xray_is_valid:

            st.error(
                "❌ This image failed X-ray verification."
            )


            st.warning(
                "Pneumonia detection has been stopped."
            )


            history_entry = (
                f"Rejected - X-ray verification failed - "
                f"{uploaded_file.name}"
            )


            if (
                history_entry
                not in st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        # ====================================================
        # X-RAY VERIFIED
        # ====================================================

        st.success(
            "✅ Chest X-ray verified."
        )


        st.write(
            f"X-ray verification confidence: "
            f"**{xray_confidence * 100:.2f}%**"
        )


        # ====================================================
        # STEP 4 — PNEUMONIA DETECTION
        # ====================================================

        st.subheader(
            "Step 4 — Pneumonia Detection"
        )


        with st.spinner(
            "Analyzing verified chest X-ray..."
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
            pneumonia_result[
                "class"
            ]
        )


        diagnosis_confidence = (
            pneumonia_result[
                "confidence"
            ]
        )


        normal_probability = (
            pneumonia_result[
                "normal_probability"
            ]
        )


        pneumonia_probability = (
            pneumonia_result[
                "pneumonia_probability"
            ]
        )


        # ====================================================
        # DIAGNOSIS
        # ====================================================

        if diagnosis == "Pneumonia":

            st.error(
                "Diagnosis: Pneumonia"
            )

        else:

            st.success(
                "Diagnosis: Normal"
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.subheader(
            "Final Result"
        )


        st.write(
            f"**Diagnosis:** {diagnosis}"
        )


        st.metric(
            "Prediction Confidence",
            f"{diagnosis_confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # PNEUMONIA PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Normal",
                f"{normal_probability * 100:.2f}%"
            )


            st.progress(
                float(
                    np.clip(
                        normal_probability,
                        0.0,
                        1.0
                    )
                )
            )


        with col2:

            st.metric(
                "Pneumonia",
                f"{pneumonia_probability * 100:.2f}%"
            )


            st.progress(
                float(
                    np.clip(
                        pneumonia_probability,
                        0.0,
                        1.0
                    )
                )
            )


        # ====================================================
        # HISTORY
        # ====================================================

        history_entry = (
            f"{diagnosis} - "
            f"{uploaded_file.name}"
        )


        if (
            history_entry
            not in st.session_state.history
        ):

            st.session_state.history.append(
                history_entry
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "Technical Details"
        ):

            st.write(
                "**Image modality:** Chest X-ray"
            )


            st.write(
                f"Modality confidence: "
                f"{modality_confidence * 100:.2f}%"
            )


            st.write(
                f"X-ray verification confidence: "
                f"{xray_confidence * 100:.2f}%"
            )


            st.write(
                f"X-ray probability: "
                f"{xray_probability * 100:.2f}%"
            )


            st.write(
                f"Non-X-ray probability: "
                f"{non_xray_probability * 100:.2f}%"
            )


            st.write(
                f"Normal probability: "
                f"{normal_probability * 100:.2f}%"
            )


            st.write(
                f"Pneumonia probability: "
                f"{pneumonia_probability * 100:.2f}%"
            )


            st.divider()


            st.write(
                "**Modality classes:**"
            )


            st.code(
                """
0 = CHEST_XRAY
1 = CT
2 = MRI
"""
            )


            st.write(
                "**X-ray verifier classes:**"
            )


            st.code(
                """
0 = X-RAY
1 = NON-XRAY
"""
            )


            st.write(
                "**Pneumonia classes:**"
            )


            st.code(
                """
0 = Normal
1 = Pneumonia
"""
            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        st.divider()

        st.subheader(
            "Diagnostic Report"
        )


        # ----------------------------------------------------
        # Clean filename
        # ----------------------------------------------------

        clean_filename = (
            uploaded_file.name
            .encode(
                "ascii",
                "ignore"
            )
            .decode(
                "ascii"
            )
        )


        if not clean_filename:

            clean_filename = "uploaded_image"


        # ----------------------------------------------------
        # Create PDF
        # ----------------------------------------------------

        pdf = FPDF()

        pdf.add_page()


        pdf.set_font(
            "Arial",
            "B",
            18
        )


        pdf.cell(
            0,
            15,
            "Pneumonia AI Diagnostic Report",
            ln=True,
            align="C"
        )


        pdf.line(
            10,
            25,
            200,
            25
        )


        pdf.ln(10)


        # ----------------------------------------------------
        # Report helper
        # ----------------------------------------------------

        def pdf_field(
            label,
            value
        ):

            pdf.set_font(
                "Arial",
                "B",
                11
            )


            pdf.cell(
                55,
                9,
                label,
                ln=False
            )


            pdf.set_font(
                "Arial",
                "",
                11
            )


            pdf.cell(
                0,
                9,
                str(value),
                ln=True
            )


        # ----------------------------------------------------
        # Report fields
        # ----------------------------------------------------

        pdf_field(
            "File Name:",
            clean_filename
        )


        pdf_field(
            "Image Modality:",
            "Chest X-ray"
        )


        pdf_field(
            "Modality Confidence:",
            f"{modality_confidence * 100:.2f}%"
        )


        pdf_field(
            "X-ray Verification:",
            "X-RAY"
        )


        pdf_field(
            "X-ray Confidence:",
            f"{xray_confidence * 100:.2f}%"
        )


        pdf_field(
            "Normal Probability:",
            f"{normal_probability * 100:.2f}%"
        )


        pdf_field(
            "Pneumonia Probability:",
            f"{pneumonia_probability * 100:.2f}%"
        )


        pdf_field(
            "Diagnosis:",
            diagnosis
        )


        pdf_field(
            "Diagnosis Confidence:",
            f"{diagnosis_confidence * 100:.2f}%"
        )


        pdf.ln(15)


        pdf.set_font(
            "Arial",
            "I",
            9
        )


        pdf.multi_cell(
            0,
            7,
            (
                "Disclaimer: This AI-generated result "
                "is intended for research purposes only "
                "and does not replace professional "
                "medical diagnosis."
            )
        )


        # ----------------------------------------------------
        # Generate PDF bytes
        # ----------------------------------------------------

        pdf_output = bytes(
            pdf.output()
        )


        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        st.download_button(
            label="Download Diagnostic Report",
            data=pdf_output,
            file_name=(
                f"Report_{clean_filename}.pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )
