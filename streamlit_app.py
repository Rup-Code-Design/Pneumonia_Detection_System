# ============================================================
# streamlit_app.py
#
# Pneumonia Detection System
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Basic Image Validation
#       ↓
# Colour Image Rejection
#       ↓
# Medical Modality Classifier
#       ↓
# CHEST X-RAY / CT / MRI
#       ↓
# If CHEST X-RAY
#       ↓
# X-ray Verifier
#       ↓
# X-RAY / NON-XRAY
#       ↓
# If verified X-RAY
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

from pathlib import Path
from PIL import Image
from fpdf import FPDF


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

BASE_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# MODEL FILES
# ============================================================
#
# IMPORTANT:
#
# We are loading COMPLETE .keras models.
#
# This avoids rebuilding the architecture and accidentally
# creating a model that does not exactly match the trained
# weights.
#
# Repository files shown in your screenshot:
#
# CT_Verifier.keras
# best_xray_verifier.keras
# best_exception_pneumonia_model.keras
#
# ============================================================

MODALITY_MODEL_PATH = (
    BASE_DIR /
    "CT_Verifier.keras"
)

XRAY_MODEL_PATH = (
    BASE_DIR /
    "best_xray_verifier.keras"
)

PNEUMONIA_MODEL_PATH = (
    BASE_DIR /
    "best_exception_pneumonia_model.keras"
)


# ============================================================
# IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (
    128,
    128
)

XRAY_IMAGE_SIZE = (
    128,
    128
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

# Difference between RGB channels.
#
# Smaller value = stricter grayscale detection.
#
COLOR_TOLERANCE = 5.0


# ============================================================
# MODALITY CONFIDENCE
# ============================================================
#
# The trained modality model has THREE classes:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
#
# ============================================================

MODALITY_CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# X-RAY VERIFIER CONFIDENCE
# ============================================================

XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# CLASS MAPPINGS
# ============================================================


# ------------------------------------------------------------
# MODALITY CLASS MAPPING
# ------------------------------------------------------------
#
# THIS MATCHES YOUR TRAINING CODE.
#
# Your training code created:
#
# CLASS_NAMES = [
#     "CHEST_XRAY",
#     "CT",
#     "MRI"
# ]
#
# Therefore:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
#
# There is NO class 3 / OTHER.
#
# ------------------------------------------------------------

MODALITY_CLASS_MAP = {

    0: "CHEST_XRAY",

    1: "CT",

    2: "MRI"
}


# ------------------------------------------------------------
# X-RAY VERIFIER CLASS MAPPING
# ------------------------------------------------------------
#
# IMPORTANT:
#
# This mapping must match the training of
# best_xray_verifier.keras.
#
# Based on your previous verifier setup:
#
# 0 = X-RAY
# 1 = NON-XRAY
#
# ------------------------------------------------------------

XRAY_CLASS_MAP = {

    0: "X-RAY",

    1: "NON-XRAY"
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# MODEL FILE CHECK
# ============================================================

def check_model_file(
    path,
    model_name
):

    if not path.exists():

        st.error(
            f"❌ {model_name} was not found."
        )

        st.code(
            str(path)
        )

        st.warning(
            "Make sure the model file is committed "
            "to the same GitHub repository as this "
            "Streamlit application."
        )

        st.stop()


    if path.stat().st_size == 0:

        st.error(
            f"❌ {model_name} is empty."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# CHECK MODEL FILES
# ============================================================

check_model_file(
    MODALITY_MODEL_PATH,
    "Medical modality classifier"
)

check_model_file(
    XRAY_MODEL_PATH,
    "X-ray verifier"
)

check_model_file(
    PNEUMONIA_MODEL_PATH,
    "Pneumonia model"
)


# ============================================================
# LOAD MODALITY MODEL
# ============================================================
#
# IMPORTANT:
#
# DO NOT do:
#
# build_modality_classifier(
#     num_classes=4
# )
#
# Your trained model has THREE classes.
#
# We load the complete CT_Verifier.keras model instead.
#
# ============================================================

@st.cache_resource
def load_modality_model():

    model = tf.keras.models.load_model(
        str(MODALITY_MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    model = tf.keras.models.load_model(
        str(XRAY_MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    model = tf.keras.models.load_model(
        str(PNEUMONIA_MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    modality_model = (
        load_modality_model()
    )

except Exception as e:

    st.error(
        "❌ Medical modality classifier loading failed."
    )

    st.error(
        "The file CT_Verifier.keras could not be loaded."
    )

    st.exception(e)

    st.stop()


try:

    xray_model = (
        load_xray_model()
    )

except Exception as e:

    st.error(
        "❌ X-ray verifier loading failed."
    )

    st.error(
        "The file best_xray_verifier.keras "
        "could not be loaded."
    )

    st.exception(e)

    st.stop()


try:

    pneumonia_model = (
        load_pneumonia_model()
    )

except Exception as e:

    st.error(
        "❌ Pneumonia model loading failed."
    )

    st.error(
        "The file best_exception_pneumonia_model.keras "
        "could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY MODEL OUTPUT SHAPES
# ============================================================

modality_output_shape = (
    modality_model.output_shape
)

xray_output_shape = (
    xray_model.output_shape
)

pneumonia_output_shape = (
    pneumonia_model.output_shape
)


# ============================================================
# MODALITY MODEL OUTPUT CHECK
# ============================================================

if (
    modality_output_shape[-1]
    != 3
):

    st.error(
        "❌ Medical modality classifier output mismatch."
    )

    st.write(
        "Expected: 3 classes"
    )

    st.write(
        "Expected mapping:"
    )

    st.code(
        """
0 = CHEST_XRAY
1 = CT
2 = MRI
"""
    )

    st.write(
        f"Actual model output: "
        f"{modality_output_shape}"
    )

    st.stop()


# ============================================================
# X-RAY MODEL OUTPUT CHECK
# ============================================================

if (
    xray_output_shape[-1]
    != 2
):

    st.error(
        "❌ X-ray verifier output mismatch."
    )

    st.write(
        "Expected: 2 classes"
    )

    st.code(
        """
0 = X-RAY
1 = NON-XRAY
"""
    )

    st.write(
        f"Actual model output: "
        f"{xray_output_shape}"
    )

    st.stop()


# ============================================================
# PNEUMONIA MODEL OUTPUT CHECK
# ============================================================
#
# Your repository contains a complete .keras model.
#
# We support:
#
# 1. Binary sigmoid output:
#       (None, 1)
#
# 2. Two-class softmax output:
#       (None, 2)
#
# ============================================================

if (
    pneumonia_output_shape[-1]
    not in [1, 2]
):

    st.error(
        "❌ Pneumonia model output mismatch."
    )

    st.write(
        "Expected either:"
    )

    st.code(
        """
(None, 1)  -> sigmoid
or
(None, 2)  -> softmax
"""
    )

    st.write(
        f"Actual model output: "
        f"{pneumonia_output_shape}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫁 Pneumonia Detection System"
)


st.markdown(
    """
### AI Image Verification Pipeline

Upload an image and the system will perform:

**1. Colour-image rejection**

**2. Medical modality verification**
- Chest X-ray
- CT
- MRI

**3. Chest X-ray verification**

**4. Pneumonia classification**

The pneumonia classifier is executed **only after the
uploaded image has been verified as a chest X-ray**.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Pipeline"
    )

    st.write(
        """
        Image Upload
        ↓
        Basic Validation
        ↓
        Colour Check
        ↓
        Modality Classifier
        ↓
        Chest X-ray?
        ↓
        X-ray Verifier
        ↓
        X-ray Confirmed?
        ↓
        Pneumonia Model
        ↓
        Normal / Pneumonia
        """
    )

    st.divider()

    st.header(
        "Medical Modality Classes"
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

    st.header(
        "Model Files"
    )

    st.code(
        "CT_Verifier.keras"
    )

    st.code(
        "best_xray_verifier.keras"
    )

    st.code(
        "best_exception_pneumonia_model.keras"
    )

    st.divider()

    st.header(
        "Recent Scans"
    )

    if len(
        st.session_state.history
    ) == 0:

        st.write(
            "No scans yet."
        )

    else:

        for item in reversed(
            st.session_state.history[-10:]
        ):

            st.text(
                item
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
        "**Medical Modality Classifier**"
    )

    st.code(
        "CT_Verifier.keras"
    )

    st.write(
        "Input: 128 × 128 × 3"
    )

    st.write(
        "Classes:"
    )

    st.code(
        """
0 = CHEST_XRAY
1 = CT
2 = MRI
"""
    )

    st.divider()

    st.write(
        "**X-ray Verifier**"
    )

    st.code(
        "best_xray_verifier.keras"
    )

    st.write(
        "Input: 128 × 128 × 3"
    )

    st.code(
        """
0 = X-RAY
1 = NON-XRAY
"""
    )

    st.divider()

    st.write(
        "**Pneumonia Model**"
    )

    st.code(
        "best_exception_pneumonia_model.keras"
    )

    st.write(
        "Input: 224 × 224 × 3"
    )

    st.write(
        f"Output shape: "
        f"{pneumonia_output_shape}"
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
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
# IMAGE VALIDATION FUNCTIONS
# ============================================================

def check_grayscale(
    image
):

    rgb = image.convert(
        "RGB"
    )

    array = np.asarray(
        rgb,
        dtype=np.float32
    )

    r = array[:, :, 0]

    g = array[:, :, 1]

    b = array[:, :, 2]

    rg = np.mean(
        np.abs(
            r - g
        )
    )

    gb = np.mean(
        np.abs(
            g - b
        )
    )

    rb = np.mean(
        np.abs(
            r - b
        )
    )

    channel_difference = (
        rg + gb + rb
    ) / 3.0

    is_grayscale = (
        channel_difference
        <= COLOR_TOLERANCE
    )

    return (
        is_grayscale,
        channel_difference
    )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_basic_image(
    image
):

    width, height = (
        image.size
    )

    if (
        width < 64
        or
        height < 64
    ):

        return (
            False,
            "Image resolution is too small."
        )


    grayscale_array = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Blank image
    # --------------------------------------------------------

    if (
        np.std(
            grayscale_array
        )
        < 8
    ):

        return (
            False,
            "Image appears blank or invalid."
        )


    # --------------------------------------------------------
    # Almost completely black
    # --------------------------------------------------------

    dark_ratio = np.mean(
        grayscale_array < 10
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
        grayscale_array > 245
    )

    if bright_ratio > 0.98:

        return (
            False,
            "Image is almost completely white."
        )


    return (
        True,
        "Basic image validation passed."
    )


# ============================================================
# PREPROCESSING
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

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array /= 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


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

    # Already softmax probabilities
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

    return (
        tf.nn.softmax(
            scores
        ).numpy()
    )


# ============================================================
# MODALITY PREDICTION
# ============================================================

def predict_modality(
    image
):

    input_array = preprocess_image(
        image,
        MODALITY_IMAGE_SIZE
    )

    prediction = (
        modality_model.predict(
            input_array,
            verbose=0
        )
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
            "exactly 3 classes."
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

    predicted_class = (
        MODALITY_CLASS_MAP.get(
            predicted_index,
            "UNKNOWN"
        )
    )

    return (
        predicted_class,
        predicted_index,
        confidence,
        probabilities
    )


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def verify_xray(
    image
):

    input_array = preprocess_image(
        image,
        XRAY_IMAGE_SIZE
    )

    prediction = (
        xray_model.predict(
            input_array,
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
            "X-ray verifier must output "
            "exactly 2 classes."
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

    predicted_class = (
        XRAY_CLASS_MAP.get(
            predicted_index,
            "UNKNOWN"
        )
    )

    xray_probability = float(
        probabilities[0]
    )

    non_xray_probability = float(
        probabilities[1]
    )

    is_xray = (
        predicted_class == "X-RAY"
        and
        confidence >=
        XRAY_CONFIDENCE_THRESHOLD
    )

    return {

        "predicted_index":
            predicted_index,

        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "probabilities":
            probabilities,

        "xray_probability":
            xray_probability,

        "non_xray_probability":
            non_xray_probability,

        "is_xray":
            is_xray
    }


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    image
):

    input_array = preprocess_image(
        image,
        PNEUMONIA_IMAGE_SIZE
    )

    prediction = (
        pneumonia_model.predict(
            input_array,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )


    # ========================================================
    # CASE 1
    # ========================================================
    #
    # Binary sigmoid:
    #
    # [[0.82]]
    #
    # 0 = Normal probability
    # 1 = Pneumonia probability
    #
    # ========================================================

    if (
        prediction.size == 1
    ):

        pneumonia_probability = float(
            np.squeeze(
                prediction
            )
        )

        pneumonia_probability = float(
            np.clip(
                pneumonia_probability,
                0.0,
                1.0
            )
        )

        normal_probability = (
            1.0
            -
            pneumonia_probability
        )


    # ========================================================
    # CASE 2
    # ========================================================
    #
    # Two-class softmax:
    #
    # [Normal, Pneumonia]
    #
    # ========================================================

    elif (
        prediction.ndim == 2
        and
        prediction.shape[1] == 2
    ):

        probabilities = (
            convert_to_probabilities(
                prediction[0]
            )
        )

        normal_probability = float(
            probabilities[0]
        )

        pneumonia_probability = float(
            probabilities[1]
        )


    else:

        raise ValueError(
            "Unsupported pneumonia model output shape: "
            f"{prediction.shape}"
        )


    # ========================================================
    # FINAL DIAGNOSIS
    # ========================================================

    if (
        pneumonia_probability
        >=
        normal_probability
    ):

        diagnosis = (
            "Pneumonia"
        )

        diagnosis_confidence = (
            pneumonia_probability
        )

    else:

        diagnosis = (
            "Normal"
        )

        diagnosis_confidence = (
            normal_probability
        )


    return (
        diagnosis,
        normal_probability,
        pneumonia_probability,
        diagnosis_confidence
    )


# ============================================================
# UPLOAD PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        file_bytes = (
            uploaded_file.getvalue()
        )

        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )

        image.load()

        image = image.convert(
            "RGB"
        )


        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            # ==================================================
            # STEP 1
            # BASIC VALIDATION
            # ==================================================

            st.subheader(
                "Step 1 — Basic Image Validation"
            )

            is_valid, validation_message = (
                validate_basic_image(
                    image
                )
            )

            if not is_valid:

                st.error(
                    f"❌ {validation_message}"
                )

                st.stop()


            st.success(
                "✅ Basic image validation passed."
            )


            # ==================================================
            # STEP 2
            # COLOUR CHECK
            # ==================================================

            st.subheader(
                "Step 2 — Colour Image Rejection"
            )

            is_grayscale, channel_difference = (
                check_grayscale(
                    image
                )
            )

            st.write(
                f"RGB channel difference: "
                f"{channel_difference:.4f}"
            )

            if not is_grayscale:

                st.error(
                    "❌ Colour image detected."
                )

                st.warning(
                    "This system accepts grayscale "
                    "medical images only."
                )

                history_entry = (
                    f"Rejected - Colour image - "
                    f"{uploaded_file.name}"
                )

                if (
                    history_entry
                    not in
                    st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )

                st.stop()


            st.success(
                "✅ Grayscale image confirmed."
            )


            # ==================================================
            # STEP 3
            # MEDICAL MODALITY CLASSIFICATION
            # ==================================================

            st.subheader(
                "Step 3 — Medical Image Modality Verification"
            )

            with st.spinner(
                "Determining image modality..."
            ):

                (
                    modality_class,
                    modality_index,
                    modality_confidence,
                    modality_probabilities

                ) = predict_modality(
                    image
                )


            # --------------------------------------------------
            # DISPLAY MODALITY PROBABILITIES
            # --------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Chest X-ray",
                    f"{modality_probabilities[0] * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "CT",
                    f"{modality_probabilities[1] * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "MRI",
                    f"{modality_probabilities[2] * 100:.2f}%"
                )


            st.write(
                f"**Predicted modality:** "
                f"{modality_class}"
            )

            st.write(
                f"**Modality confidence:** "
                f"{modality_confidence * 100:.2f}%"
            )


            # ==================================================
            # CT REJECTION
            # ==================================================

            if (
                modality_class == "CT"
            ):

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
                    not in
                    st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )

                st.stop()


            # ==================================================
            # MRI REJECTION
            # ==================================================

            if (
                modality_class == "MRI"
            ):

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
                    not in
                    st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )

                st.stop()


            # ==================================================
            # UNKNOWN MODALITY
            # ==================================================

            if (
                modality_class
                !=
                "CHEST_XRAY"
            ):

                st.error(
                    "❌ Unsupported medical image."
                )

                st.stop()


            # ==================================================
            # LOW CHEST X-RAY CONFIDENCE
            # ==================================================

            if (
                modality_confidence
                <
                MODALITY_CONFIDENCE_THRESHOLD
            ):

                st.error(
                    "❌ Chest X-ray confidence is too low."
                )

                st.warning(
                    f"Required confidence: "
                    f"{MODALITY_CONFIDENCE_THRESHOLD * 100:.0f}%"
                )

                st.warning(
                    "Please upload a clearer chest X-ray."
                )

                history_entry = (
                    f"Rejected - Low modality confidence - "
                    f"{uploaded_file.name}"
                )

                if (
                    history_entry
                    not in
                    st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )

                st.stop()


            # ==================================================
            # CHEST X-RAY CONFIRMED BY MODALITY MODEL
            # ==================================================

            st.success(
                "✅ Medical modality: Chest X-ray"
            )


            # ==================================================
            # STEP 4
            # X-RAY VERIFIER
            # ==================================================

            st.subheader(
                "Step 4 — Chest X-ray Verification"
            )

            with st.spinner(
                "Verifying chest X-ray..."
            ):

                xray_result = (
                    verify_xray(
                        image
                    )
                )


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

            xray_class = (
                xray_result[
                    "predicted_class"
                ]
            )

            is_xray = (
                xray_result[
                    "is_xray"
                ]
            )


            # --------------------------------------------------
            # DISPLAY X-RAY RESULTS
            # --------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "X-ray Probability",
                    f"{xray_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Non-X-ray Probability",
                    f"{non_xray_probability * 100:.2f}%"
                )


            if not is_xray:

                st.error(
                    "❌ This image failed X-ray verification."
                )

                st.warning(
                    "Pneumonia detection has been stopped."
                )

                history_entry = (
                    f"Rejected - X-ray verification - "
                    f"{uploaded_file.name}"
                )

                if (
                    history_entry
                    not in
                    st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )

                st.stop()


            st.success(
                "✅ Chest X-ray verified."
            )

            st.write(
                f"X-ray verification confidence: "
                f"**{xray_confidence * 100:.2f}%**"
            )


            # ==================================================
            # STEP 5
            # PNEUMONIA DETECTION
            # ==================================================

            st.subheader(
                "Step 5 — Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing verified chest X-ray..."
            ):

                (
                    diagnosis,
                    normal_probability,
                    pneumonia_probability,
                    diagnosis_confidence

                ) = predict_pneumonia(
                    image
                )


            # ==================================================
            # DISPLAY DIAGNOSIS
            # ==================================================

            if (
                diagnosis
                ==
                "Pneumonia"
            ):

                st.error(
                    "Diagnosis: Pneumonia"
                )

            else:

                st.success(
                    "Diagnosis: Normal"
                )


            # ==================================================
            # FINAL RESULT
            # ==================================================

            st.subheader(
                "Final Result"
            )


            st.write(
                f"### {diagnosis}"
            )


            st.metric(
                "Prediction Confidence",
                f"{diagnosis_confidence * 100:.2f}%"
            )


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


            # ==================================================
            # HISTORY
            # ==================================================

            history_entry = (
                f"{diagnosis} - "
                f"{uploaded_file.name}"
            )

            if (
                history_entry
                not in
                st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            # ==================================================
            # TECHNICAL DETAILS
            # ==================================================

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "**Medical modality:** "
                    "Chest X-ray"
                )

                st.write(
                    f"Modality confidence: "
                    f"{modality_confidence * 100:.2f}%"
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
                    f"Pneumonia probability: "
                    f"{pneumonia_probability * 100:.2f}%"
                )

                st.write(
                    f"Normal probability: "
                    f"{normal_probability * 100:.2f}%"
                )

                st.write(
                    f"Modality model output: "
                    f"{modality_output_shape}"
                )

                st.write(
                    f"X-ray model output: "
                    f"{xray_output_shape}"
                )

                st.write(
                    f"Pneumonia model output: "
                    f"{pneumonia_output_shape}"
                )


            # ==================================================
            # PDF REPORT
            # ==================================================

            st.divider()

            st.subheader(
                "Diagnostic Report"
            )


            st.write(
                f"**File:** "
                f"{uploaded_file.name}"
            )

            st.write(
                "**Modality:** Chest X-ray"
            )

            st.write(
                f"**Modality confidence:** "
                f"{modality_confidence * 100:.2f}%"
            )

            st.write(
                f"**X-ray confidence:** "
                f"{xray_confidence * 100:.2f}%"
            )

            st.write(
                f"**Diagnosis:** "
                f"{diagnosis}"
            )

            st.write(
                f"**Diagnosis confidence:** "
                f"{diagnosis_confidence * 100:.2f}%"
            )


            # ==================================================
            # CREATE PDF
            # ==================================================

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
                "Pneumonia AI Report",
                ln=True,
                align="C"
            )


            pdf.line(
                10,
                25,
                200,
                25
            )


            pdf.ln(
                10
            )


            # --------------------------------------------------
            # FILE
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "File Name:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                clean_filename,
                ln=True
            )


            # --------------------------------------------------
            # MODALITY
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Modality:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                "Chest X-ray",
                ln=True
            )


            # --------------------------------------------------
            # MODALITY CONFIDENCE
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Modality Confidence:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                f"{modality_confidence * 100:.2f}%",
                ln=True
            )


            # --------------------------------------------------
            # X-RAY CONFIDENCE
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "X-ray Confidence:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                f"{xray_confidence * 100:.2f}%",
                ln=True
            )


            # --------------------------------------------------
            # DIAGNOSIS
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Diagnosis:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                diagnosis,
                ln=True
            )


            # --------------------------------------------------
            # DIAGNOSIS CONFIDENCE
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Confidence:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                f"{diagnosis_confidence * 100:.2f}%",
                ln=True
            )


            # --------------------------------------------------
            # PROBABILITIES
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Normal Probability:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                f"{normal_probability * 100:.2f}%",
                ln=True
            )


            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Pneumonia Probability:",
                ln=False
            )

            pdf.set_font(
                "Arial",
                "",
                12
            )

            pdf.cell(
                0,
                10,
                f"{pneumonia_probability * 100:.2f}%",
                ln=True
            )


            pdf.ln(
                15
            )


            # --------------------------------------------------
            # DISCLAIMER
            # --------------------------------------------------

            pdf.set_font(
                "Arial",
                "I",
                10
            )

            pdf.multi_cell(
                0,
                7,
                "Disclaimer: "
                "This AI-generated result is intended "
                "for research and educational purposes only "
                "and does not replace professional medical "
                "diagnosis."
            )


            # ==================================================
            # PDF BYTES
            # ==================================================

            pdf_output = (
                pdf.output()
            )


            st.download_button(
                label=(
                    "Download Diagnostic Report"
                ),
                data=bytes(
                    pdf_output
                ),
                file_name=(
                    f"Report_{clean_filename}.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ An error occurred while processing "
            "the uploaded image."
        )

        st.exception(e)
