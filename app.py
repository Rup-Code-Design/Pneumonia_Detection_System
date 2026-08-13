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
# Chest X-ray only
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
# This assumes your verifier training used:
#
# XRAY      = 0
# NON_XRAY  = 1
#
# If your actual training class_indices are different,
# change this mapping.
# ============================================================

XRAY_CLASS_MAP = {

    0: "X-RAY",

    1: "NON-XRAY"
}


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================
#
# This assumes:
#
# NORMAL    = 0
# PNEUMONIA = 1
#
# Verify this against the class_indices from your
# pneumonia training code.
# ============================================================

PNEUMONIA_CLASS_MAP = {

    0: "Normal",

    1: "Pneumonia"
}


# ============================================================
# THRESHOLDS
# ============================================================

XRAY_CONFIDENCE_THRESHOLD = 0.50

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
# LOAD YOUR PROPOSED PNEUMONIA MODEL
# ============================================================
#
# EXACT architecture from the user's model_builder.py:
#
# Xception Block 64
# MaxPooling
# Xception Block 128
# MaxPooling
# Residual Block 256
# GAP + GMP
# Concatenate
# BatchNormalization
# Dense 128 GELU
# Dropout 0.3
# Dense 2 Softmax
#
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Your build_model() defaults to num_classes=3.
    #
    # Pneumonia detection requires exactly 2 classes:
    #
    # 0 = Normal
    # 1 = Pneumonia
    #
    # Therefore explicitly set num_classes=2.
    # --------------------------------------------------------

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=2
    )


    # --------------------------------------------------------
    # Load trained parameters
    # --------------------------------------------------------

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
# VERIFY X-RAY MODEL OUTPUT
# ============================================================

try:

    xray_output_shape = (
        xray_model.output_shape
    )


    if xray_output_shape[-1] != 2:

        st.error(
            "❌ X-ray verifier must have "
            "2 output classes."
        )

        st.write(
            f"Actual output shape: "
            f"{xray_output_shape}"
        )

        st.stop()


except Exception as e:

    st.error(
        "❌ Unable to verify X-ray model output."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY PROPOSED PNEUMONIA MODEL OUTPUT
# ============================================================

try:

    pneumonia_output_shape = (
        pneumonia_model.output_shape
    )


    if pneumonia_output_shape[-1] != 2:

        st.error(
            "❌ Proposed pneumonia model must "
            "have exactly 2 output classes."
        )

        st.write(
            f"Actual output shape: "
            f"{pneumonia_output_shape}"
        )

        st.stop()


except Exception as e:

    st.error(
        "❌ Unable to verify pneumonia model output."
    )

    st.exception(e)

    st.stop()


# ============================================================
# COLOUR IMAGE CHECK
# ============================================================

def check_grayscale_image(image):

    """
    Checks whether the image is effectively grayscale.

    A grayscale X-ray saved as RGB will pass.

    A genuine colour image will be rejected.

    This check does NOT determine whether a grayscale image
    is an X-ray, MRI, CT, etc. That is the job of the
    trained X-ray verifier.
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
    # BLANK IMAGE
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


    # Same normalization used during training
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
    # IMPORTANT
    #
    # Your provided model does not contain an internal
    # preprocessing layer.
    #
    # Therefore this assumes the training pipeline used:
    #
    # image / 255.0
    #
    # If your actual pneumonia training generator used a
    # different preprocessing function, it MUST be changed
    # here to exactly match training.
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


    if (

        predictions.ndim != 2

        or

        predictions.shape[1] != 2

    ):

        raise ValueError(
            "X-ray verifier must produce "
            "2 output values."
        )


    raw_scores = (
        predictions[0]
        .astype(np.float64)
    )


    # --------------------------------------------------------
    # Detect whether output is already probabilities
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


    result = XRAY_CLASS_MAP.get(
        predicted_index,
        "UNKNOWN"
    )


    return (
        result,
        confidence,
        probabilities
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
    # MODEL PREDICTION
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
    # EXPECTED OUTPUT
    #
    # [Normal, Pneumonia]
    #
    # Example:
    #
    # [0.8138, 0.1862]
    #
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
    # SOFTMAX MODEL OUTPUT
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
    "Only chest X-ray images are accepted. "
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
        Chest X-ray
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
        "Input size:"
    )

    st.write(
        "224 × 224 × 3"
    )


    st.write(
        "Architecture:"
    )

    st.write(
        "Xception Block → Xception Block → "
        "Residual Block → GAP + GMP → "
        "Dense 128 + GELU → Dropout → Softmax"
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
    # DISPLAY
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
                    xray_probabilities
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
        # X-RAY PROBABILITIES
        # ====================================================

        xray_probability = float(
            xray_probabilities[0]
        )


        non_xray_probability = float(
            xray_probabilities[1]
        )


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


        # ====================================================
        # REJECT NON-X-RAY
        # ====================================================

        if (

            xray_result != "X-RAY"

            or

            xray_confidence
            < XRAY_CONFIDENCE_THRESHOLD

        ):

            st.error(
                "❌ This is not a Chest X-ray image."
            )


            st.write(
                f"Verifier confidence: "
                f"{xray_confidence * 100:.2f}%"
            )


            st.warning(
                "Please upload a valid chest X-ray image."
            )


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
        # X-RAY ACCEPTED
        # ====================================================

        st.success(
            "✅ Chest X-ray image detected."
        )


        st.write(
            f"X-ray verification confidence: "
            f"{xray_confidence * 100:.2f}%"
        )


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
        # RESULT
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
                normal_probability
            )


        with col2:

            st.metric(
                "Pneumonia",
                f"{pneumonia_probability * 100:.2f}%"
            )


            st.progress(
                pneumonia_probability
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
