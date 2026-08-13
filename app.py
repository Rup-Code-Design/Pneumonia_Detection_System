# ============================================================
# app.py
# Pneumonia Detection System
#
# Pipeline:
#
# Uploaded Image
#       ↓
# Grayscale / Colour Check
#       ↓
# Chest X-ray / Non-X-ray Verification
#       ↓
# X-RAY VERIFIED
#       ↓
# Proposed Pneumonia Model
#
# Xception Block
# + Residual Block
# + SE Attention
# + GAP + GMP
# + GELU
#
#       ↓
# Normal / Pneumonia
# ============================================================


import os
import io

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image

from model_builder import build_model
from xray_model_builder import build_xray_classifier


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


# ============================================================
# IMAGE SIZES
# ============================================================

XRAY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# X-RAY VERIFIER CLASS MAPPING
# ============================================================
#
# IMPORTANT:
#
# This code assumes the X-ray verifier was trained with:
#
# NON_XRAY = 0
# XRAY     = 1
#
# Example:
#
# class_indices:
#
# {
#     'NON_XRAY': 0,
#     'XRAY': 1
# }
#
# If your training code produced the opposite mapping,
# CHANGE THESE VALUES.
# ============================================================

XRAY_CLASS_MAP = {
    0: "NON-XRAY",
    1: "X-RAY"
}

NON_XRAY_CLASS_INDEX = 0
XRAY_CLASS_INDEX = 1


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================
#
# Your proposed model uses:
#
# 0 = Normal
# 1 = Pneumonia
#
# ============================================================

PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}


# ============================================================
# X-RAY VERIFICATION SETTINGS
# ============================================================

# Minimum probability required to accept image as X-ray.
XRAY_ACCEPT_THRESHOLD = 0.50

# Require X-ray probability to be greater than
# Non-X-ray probability.
REQUIRE_XRAY_DOMINANCE = True


# ============================================================
# COLOUR IMAGE SETTINGS
# ============================================================

COLOR_TOLERANCE = 8.0


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {

    "X-ray verifier":
        XRAY_MODEL_PATH,

    "Pneumonia model":
        PNEUMONIA_MODEL_PATH
}


for model_name, model_path in required_files.items():

    if not os.path.isfile(model_path):

        st.error(
            f"❌ {model_name} was not found."
        )

        st.code(
            model_path
        )

        st.stop()


    if os.path.getsize(model_path) == 0:

        st.error(
            f"❌ {model_name} file is empty."
        )

        st.stop()


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    model = build_xray_classifier(
        input_shape=(128, 128, 3)
    )

    model.load_weights(
        XRAY_MODEL_PATH
    )

    return model


# ============================================================
# LOAD PROPOSED PNEUMONIA MODEL
# ============================================================
#
# Architecture supplied by you:
#
# Xception Block 64
#       ↓
# MaxPooling
#       ↓
# Xception Block 128
#       ↓
# MaxPooling
#       ↓
# Residual Block 256
#       ↓
# GAP + GMP
#       ↓
# Concatenate
#       ↓
# BatchNormalization
#       ↓
# Dense 128 + GELU
#       ↓
# Dropout 0.3
#       ↓
# Dense 2 + Softmax
#
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=2
    )

    model.load_weights(
        PNEUMONIA_MODEL_PATH
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model = load_xray_model()

except Exception as e:

    st.error(
        "❌ X-ray verifier loading failed."
    )

    st.exception(e)

    st.stop()


try:

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error(
        "❌ Proposed pneumonia model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY X-RAY MODEL
# ============================================================

try:

    xray_output_shape = (
        xray_model.output_shape
    )

    if xray_output_shape[-1] != 2:

        st.error(
            "❌ X-ray verifier configuration error."
        )

        st.stop()

except Exception as e:

    st.error(
        "❌ Unable to initialize X-ray verifier."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY PNEUMONIA MODEL
# ============================================================

try:

    pneumonia_output_shape = (
        pneumonia_model.output_shape
    )

    if pneumonia_output_shape[-1] != 2:

        st.error(
            "❌ Proposed pneumonia model configuration "
            "is invalid."
        )

        st.stop()

except Exception as e:

    st.error(
        "❌ Unable to initialize pneumonia model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# COLOUR IMAGE CHECK
# ============================================================

def check_grayscale_image(image):

    """
    Determines whether an image is effectively grayscale.

    A grayscale X-ray stored as RGB is accepted.

    A genuine colour image is rejected.

    This function does NOT determine whether a grayscale
    image is an X-ray. That is handled by the trained
    X-ray verifier.
    """

    try:

        rgb_image = image.convert(
            "RGB"
        )

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.float32
        )

        r = rgb_array[:, :, 0]

        g = rgb_array[:, :, 1]

        b = rgb_array[:, :, 2]

        rg_difference = np.mean(
            np.abs(
                r - g
            )
        )

        gb_difference = np.mean(
            np.abs(
                g - b
            )
        )

        rb_difference = np.mean(
            np.abs(
                r - b
            )
        )

        average_difference = (

            rg_difference
            +
            gb_difference
            +
            rb_difference

        ) / 3.0

        if average_difference > COLOR_TOLERANCE:

            return (
                False,
                "❌ Colour image detected. "
                "Please upload a grayscale chest X-ray."
            )

        return (
            True,
            "Grayscale image detected."
        )

    except Exception as e:

        return (
            False,
            f"❌ Unable to check image colour: {e}"
        )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(image):

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    width, height = image.size

    if width < 64 or height < 64:

        return (
            False,
            "❌ Image resolution is too small."
        )


    # --------------------------------------------------------
    # COLOUR CHECK
    # --------------------------------------------------------

    is_grayscale, message = (
        check_grayscale_image(
            image
        )
    )

    if not is_grayscale:

        return (
            False,
            message
        )


    # --------------------------------------------------------
    # GRAYSCALE STATISTICS
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )

    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # BLANK IMAGE CHECK
    # --------------------------------------------------------

    standard_deviation = np.std(
        gray_array
    )

    if standard_deviation < 8:

        return (
            False,
            "❌ Image appears blank or invalid."
        )


    # --------------------------------------------------------
    # ALMOST BLACK
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray_array < 10
    )

    if dark_ratio > 0.98:

        return (
            False,
            "❌ Image is almost completely black."
        )


    # --------------------------------------------------------
    # ALMOST WHITE
    # --------------------------------------------------------

    bright_ratio = np.mean(
        gray_array > 245
    )

    if bright_ratio > 0.98:

        return (
            False,
            "❌ Image is almost completely white."
        )


    return (
        True,
        "✅ Image passed basic validation."
    )


# ============================================================
# PREPROCESS X-RAY VERIFIER
# ============================================================

def preprocess_for_xray_verifier(image):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        XRAY_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This must match the X-ray verifier training pipeline.
    #
    # Assumed:
    #
    # rescale = 1 / 255
    # --------------------------------------------------------

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREPROCESS PROPOSED PNEUMONIA MODEL
# ============================================================

def preprocess_for_pneumonia(image):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        PNEUMONIA_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Your supplied model has no preprocessing layer.
    #
    # Assumed training preprocessing:
    #
    # image / 255.0
    # --------------------------------------------------------

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def verify_xray(image):

    image_array = (
        preprocess_for_xray_verifier(
            image
        )
    )

    predictions = (
        xray_model.predict(
            image_array,
            verbose=0
        )
    )

    predictions = np.asarray(
        predictions
    )


    # --------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------

    if (
        predictions.ndim != 2
        or
        predictions.shape[1] != 2
    ):

        raise ValueError(
            "X-ray verifier must produce exactly "
            "2 output values."
        )


    raw_scores = (
        predictions[0]
        .astype(np.float64)
    )


    # --------------------------------------------------------
    # CONVERT TO PROBABILITIES
    # --------------------------------------------------------

    if (

        np.all(
            raw_scores >= 0.0
        )

        and

        np.all(
            raw_scores <= 1.0
        )

        and

        np.isclose(
            np.sum(raw_scores),
            1.0,
            atol=1e-3
        )

    ):

        probabilities = raw_scores

    else:

        probabilities = (
            tf.nn.softmax(
                raw_scores
            ).numpy()
        )


    # --------------------------------------------------------
    # EXPLICIT CLASS PROBABILITIES
    #
    # 0 = NON-XRAY
    # 1 = X-RAY
    # --------------------------------------------------------

    non_xray_probability = float(
        probabilities[
            NON_XRAY_CLASS_INDEX
        ]
    )

    xray_probability = float(
        probabilities[
            XRAY_CLASS_INDEX
        ]
    )


    # --------------------------------------------------------
    # RAW PREDICTED CLASS
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_class = (
        XRAY_CLASS_MAP[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # X-RAY ACCEPTANCE TEST
    # --------------------------------------------------------
    #
    # Conditions:
    #
    # 1. X-ray probability >= threshold
    #
    # 2. X-ray probability > Non-X-ray probability
    #
    # --------------------------------------------------------

    xray_probability_pass = (
        xray_probability
        >= XRAY_ACCEPT_THRESHOLD
    )


    xray_dominance_pass = (
        xray_probability
        >
        non_xray_probability
    )


    if REQUIRE_XRAY_DOMINANCE:

        is_xray = (
            xray_probability_pass
            and
            xray_dominance_pass
        )

    else:

        is_xray = (
            xray_probability_pass
        )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if is_xray:

        result = "X-RAY"

        confidence = (
            xray_probability
        )

    else:

        result = "NON-XRAY"

        confidence = (
            non_xray_probability
        )


    return (
        result,
        confidence,
        probabilities,
        is_xray,
        xray_probability,
        non_xray_probability
    )


# ============================================================
# PROPOSED PNEUMONIA MODEL PREDICTION
# ============================================================

def predict_pneumonia(image):

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_array = (
        preprocess_for_pneumonia(
            image
        )
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = (
        pneumonia_model.predict(
            image_array,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------

    if (

        prediction.ndim != 2

        or

        prediction.shape[1] != 2

    ):

        raise ValueError(
            "Proposed pneumonia model must "
            "produce exactly 2 outputs."
        )


    raw_scores = (
        prediction[0]
        .astype(np.float64)
    )


    # --------------------------------------------------------
    # SOFTMAX PROBABILITIES
    # --------------------------------------------------------

    if (

        np.all(
            raw_scores >= 0.0
        )

        and

        np.all(
            raw_scores <= 1.0
        )

        and

        np.isclose(
            np.sum(raw_scores),
            1.0,
            atol=1e-3
        )

    ):

        probabilities = raw_scores

    else:

        probabilities = (
            tf.nn.softmax(
                raw_scores
            ).numpy()
        )


    # --------------------------------------------------------
    # CLASS
    # --------------------------------------------------------

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


    return (
        predicted_class,
        confidence,
        normal_probability,
        pneumonia_probability
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫁 Pneumonia Detection System"
)


st.write(
    "Chest X-ray verification followed by "
    "pneumonia classification using the proposed "
    "deep-learning model."
)


st.info(
    "Only grayscale chest X-ray images are accepted. "
    "Colour and non-X-ray images are rejected."
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
        Grayscale Check
        ↓
        X-ray Verification
        ↓
        X-RAY VERIFIED
        ↓
        Proposed Model
        ↓
        Normal / Pneumonia
        """
    )


    st.divider()


    st.write(
        "**Proposed Model**"
    )


    st.write(
        """
        Xception Block
        + SE Attention
        + Residual Block
        + GAP + GMP
        + GELU
        """
    )


    st.divider()


    st.write(
        "**Output Classes**"
    )


    st.write(
        "Normal"
    )


    st.write(
        "Pneumonia"
    )


    st.divider()


    st.write(
        "**Rejected:**"
    )


    st.write(
        "Colour images"
    )


    st.write(
        "MRI"
    )


    st.write(
        "CT"
    )


    st.write(
        "Other non-X-ray images"
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
        "X-ray verifier:"
    )

    st.code(
        "best_xray_verifier.weights.h5"
    )


    st.write(
        "Pneumonia model:"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )


    st.write(
        "X-ray verifier input:"
    )

    st.write(
        "128 × 128 × 3"
    )


    st.write(
        "Pneumonia model input:"
    )

    st.write(
        "224 × 224 × 3"
    )


    st.write(
        "Proposed architecture:"
    )

    st.write(
        "Xception Block (64) → "
        "MaxPooling → "
        "Xception Block (128) → "
        "MaxPooling → "
        "Residual Block (256) → "
        "GAP + GMP → "
        "Dense 128 + GELU → "
        "Dropout → "
        "Softmax"
    )


    st.write(
        "Attention:"
    )

    st.write(
        "Squeeze-and-Excitation (SE)"
    )


    st.write(
        "Output:"
    )

    st.write(
        "Normal / Pneumonia"
    )


    st.write(
        "X-ray acceptance threshold:"
    )

    st.write(
        f"{XRAY_ACCEPT_THRESHOLD * 100:.0f}%"
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    help="Upload a grayscale chest X-ray image."
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        )

        image.load()

    except Exception as e:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.subheader(
        "Uploaded Image"
    )


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    is_valid, validation_message = (
        validate_image(
            image
        )
    )


    if not is_valid:

        st.error(
            validation_message
        )

        st.warning(
            "Please upload a valid grayscale "
            "chest X-ray image."
        )

        st.stop()


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    ):


        # ====================================================
        # STEP 1 — X-RAY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Chest X-ray Verification"
        )


        with st.spinner(
            "Verifying chest X-ray..."
        ):

            try:

                (
                    xray_result,
                    xray_confidence,
                    xray_probabilities,
                    is_xray,
                    xray_probability,
                    non_xray_probability

                ) = verify_xray(
                    image
                )


            except Exception as e:

                st.error(
                    "❌ X-ray verification failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # X-RAY VERIFICATION DECISION
        # ====================================================

        st.markdown(
            "### X-RAY VERIFICATION DECISION"
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Chest X-ray Probability",
                f"{xray_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Non-X-ray Probability",
                f"{non_xray_probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # PROBABILITY BARS
        # ----------------------------------------------------

        st.write(
            "**Chest X-ray probability**"
        )


        st.progress(
            min(
                max(
                    xray_probability,
                    0.0
                ),
                1.0
            )
        )


        st.write(
            "**Non-X-ray probability**"
        )


        st.progress(
            min(
                max(
                    non_xray_probability,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # FINAL X-RAY DECISION
        # ====================================================

        if is_xray:

            st.success(
                "✅ X-RAY VERIFIED"
            )


            st.write(
                f"Chest X-ray confidence: "
                f"**{xray_probability * 100:.2f}%**"
            )


            st.info(
                "The uploaded image has been verified "
                "as a chest X-ray. Pneumonia detection "
                "will now be performed."
            )


        else:

            st.error(
                "❌ NON-X-RAY IMAGE DETECTED"
            )


            st.write(
                f"Non-X-ray confidence: "
                f"**{non_xray_probability * 100:.2f}%**"
            )


            st.warning(
                "Pneumonia detection has been stopped "
                "because the uploaded image was not "
                "sufficiently verified as a chest X-ray."
            )


            # ------------------------------------------------
            # EXPLAIN REJECTION
            # ------------------------------------------------

            if (
                xray_probability
                < XRAY_ACCEPT_THRESHOLD
            ):

                st.write(
                    f"X-ray probability "
                    f"({xray_probability * 100:.2f}%) "
                    f"is below the acceptance threshold "
                    f"({XRAY_ACCEPT_THRESHOLD * 100:.0f}%)."
                )


            elif (
                xray_probability
                <= non_xray_probability
            ):

                st.write(
                    "The X-ray probability is not greater "
                    "than the Non-X-ray probability."
                )


            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            history_entry = (
                f"Rejected - "
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
        # STEP 2 — PROPOSED PNEUMONIA MODEL
        # ====================================================

        st.subheader(
            "Step 2 — Pneumonia Detection"
        )


        with st.spinner(
            "Analyzing chest X-ray using "
            "the proposed model..."
        ):

            try:

                (
                    predicted_class,
                    prediction_confidence,
                    normal_probability,
                    pneumonia_probability

                ) = predict_pneumonia(
                    image
                )


            except Exception as e:

                st.error(
                    "❌ Pneumonia prediction failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # DIAGNOSIS
        # ====================================================

        if predicted_class == "Pneumonia":

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
            f"**Diagnosis:** {predicted_class}"
        )


        st.metric(
            "Prediction Confidence",
            f"{prediction_confidence * 100:.2f}%"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Normal",
                f"{normal_probability * 100:.2f}%"
            )


            st.progress(
                float(
                    normal_probability
                )
            )


        with col2:

            st.metric(
                "Pneumonia",
                f"{pneumonia_probability * 100:.2f}%"
            )


            st.progress(
                float(
                    pneumonia_probability
                )
            )


        # ====================================================
        # HISTORY
        # ====================================================

        history_entry = (
            f"{predicted_class} - "
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
            "Technical Model Details"
        ):

            st.write(
                "Verified modality: Chest X-ray"
            )


            st.write(
                f"X-ray verification confidence: "
                f"{xray_confidence * 100:.2f}%"
            )


            st.write(
                "Proposed architecture:"
            )


            st.write(
                "Xception Block (64 filters)"
            )


            st.write(
                "Xception Block (128 filters)"
            )


            st.write(
                "Residual Block (256 filters)"
            )


            st.write(
                "SE Attention"
            )


            st.write(
                "Global Average Pooling + "
                "Global Max Pooling"
            )


            st.write(
                "Batch Normalization"
            )


            st.write(
                "Dense 128 + GELU"
            )


            st.write(
                "Dropout = 0.3"
            )


            st.write(
                "Softmax = 2 classes"
            )


            st.write(
                "Classes = Normal / Pneumonia"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.info(
            "⚠️ This system is a research prototype "
            "for educational and research purposes. "
            "It is not intended to provide clinical "
            "diagnosis or replace professional medical "
            "evaluation."
        )
