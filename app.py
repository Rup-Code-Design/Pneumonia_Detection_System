# ============================================================
# app.py
# Pneumonia Detection System
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Basic Image Validation
#       ↓
# Grayscale Check
#       ↓
# X-ray Verifier
#       ↓
# X-RAY VERIFICATION DECISION
#       ↓
# If X-ray → Pneumonia Model
#       ↓
# Normal / Pneumonia
#
# Proposed Pneumonia Model:
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
# Dropout
#       ↓
# Softmax
#
# ============================================================


import os
import io

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
# IMPORTANT
#
# This MUST match the training class_indices.
#
# Expected:
#
# NON_XRAY = 0
# XRAY     = 1
#
# If your verifier was trained with the opposite mapping,
# change these indices.
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

PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}


# ============================================================
# X-RAY VERIFICATION SETTINGS
# ============================================================

# Minimum probability for X-ray acceptance.
#
# 0.50 means X-ray probability must be at least 50%.
XRAY_ACCEPT_THRESHOLD = 0.50


# Require X-ray probability to be greater than
# Non-X-ray probability.
REQUIRE_XRAY_DOMINANCE = True


# ============================================================
# IMAGE SETTINGS
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
# Architecture:
#
# Xception Block 64
# MaxPooling
# Xception Block 128
# MaxPooling
# Residual Block 256
# GAP + GMP
# Concatenate
# BatchNormalization
# Dense 128 + GELU
# Dropout 0.3
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
# VERIFY X-RAY MODEL OUTPUT
# ============================================================

try:

    xray_output_shape = xray_model.output_shape

    if xray_output_shape[-1] != 2:

        st.error(
            "❌ X-ray verifier must have exactly "
            "2 output classes."
        )

        st.write(
            f"Actual output shape: "
            f"{xray_output_shape}"
        )

        st.stop()

except Exception as e:

    st.error(
        "❌ Unable to verify X-ray model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY PNEUMONIA MODEL OUTPUT
# ============================================================

try:

    pneumonia_output_shape = (
        pneumonia_model.output_shape
    )

    if pneumonia_output_shape[-1] != 2:

        st.error(
            "❌ Pneumonia model must have exactly "
            "2 output classes."
        )

        st.write(
            f"Actual output shape: "
            f"{pneumonia_output_shape}"
        )

        st.stop()

except Exception as e:

    st.error(
        "❌ Unable to verify pneumonia model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# COLOUR / GRAYSCALE CHECK
# ============================================================

def check_grayscale_image(image):

    """
    Determines whether the image is effectively grayscale.

    A grayscale image saved as RGB is accepted.

    A genuine colour image is rejected.

    This function does NOT determine whether an image
    is an X-ray.
    """

    try:

        rgb_image = image.convert("RGB")

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.float32
        )

        r = rgb_array[:, :, 0]

        g = rgb_array[:, :, 1]

        b = rgb_array[:, :, 2]

        rg_difference = np.mean(
            np.abs(r - g)
        )

        gb_difference = np.mean(
            np.abs(g - b)
        )

        rb_difference = np.mean(
            np.abs(r - b)
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
                "❌ Colour image detected."
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
    # GRAYSCALE CHECK
    # --------------------------------------------------------

    is_grayscale, message = (
        check_grayscale_image(image)
    )

    if not is_grayscale:

        return (
            False,
            message
        )


    # --------------------------------------------------------
    # GRAYSCALE ARRAY
    # --------------------------------------------------------

    gray = image.convert("L")

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
# PREPROCESS FOR X-RAY VERIFIER
# ============================================================

def preprocess_for_xray_verifier(image):

    image = image.convert("RGB")

    image = image.resize(
        XRAY_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Must match verifier training preprocessing.
    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREPROCESS FOR PNEUMONIA MODEL
# ============================================================

def preprocess_for_pneumonia(image):

    image = image.convert("RGB")

    image = image.resize(
        PNEUMONIA_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Must match pneumonia model training preprocessing.
    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# SOFTMAX / PROBABILITY CONVERSION
# ============================================================

def convert_to_probabilities(raw_scores):

    raw_scores = np.asarray(
        raw_scores,
        dtype=np.float64
    )

    # Already probabilities
    if (

        np.all(raw_scores >= 0.0)

        and

        np.all(raw_scores <= 1.0)

        and

        np.isclose(
            np.sum(raw_scores),
            1.0,
            atol=1e-3
        )

    ):

        return raw_scores


    # Otherwise treat as logits
    return tf.nn.softmax(
        raw_scores
    ).numpy()


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def verify_xray(image):

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_array = (
        preprocess_for_xray_verifier(
            image
        )
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    predictions = xray_model.predict(
        image_array,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    )


    # --------------------------------------------------------
    # OUTPUT CHECK
    # --------------------------------------------------------

    if (

        predictions.ndim != 2

        or

        predictions.shape[1] != 2

    ):

        raise ValueError(
            "X-ray verifier must produce "
            "exactly 2 outputs."
        )


    # --------------------------------------------------------
    # CONVERT TO PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        convert_to_probabilities(
            predictions[0]
        )
    )


    # --------------------------------------------------------
    # EXPLICIT PROBABILITIES
    #
    # 0 = NON-XRAY
    # 1 = XRAY
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
    # RAW MODEL CLASS
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_class = XRAY_CLASS_MAP.get(
        predicted_index,
        "UNKNOWN"
    )


    # ========================================================
    # X-RAY DECISION LOGIC
    # ========================================================

    # Condition 1:
    # X-ray probability must reach threshold.

    threshold_pass = (
        xray_probability
        >= XRAY_ACCEPT_THRESHOLD
    )


    # Condition 2:
    # X-ray probability must beat non-X-ray probability.

    dominance_pass = (
        xray_probability
        >
        non_xray_probability
    )


    # Final decision

    if REQUIRE_XRAY_DOMINANCE:

        is_xray = (
            threshold_pass
            and
            dominance_pass
        )

    else:

        is_xray = threshold_pass


    # --------------------------------------------------------
    # FINAL CLASS
    # --------------------------------------------------------

    if is_xray:

        result = "X-RAY"

        confidence = xray_probability

    else:

        result = "NON-XRAY"

        confidence = non_xray_probability


    return {

        "result": result,

        "confidence": confidence,

        "probabilities": probabilities,

        "is_xray": is_xray,

        "xray_probability": xray_probability,

        "non_xray_probability":
            non_xray_probability,

        "threshold_pass":
            threshold_pass,

        "dominance_pass":
            dominance_pass,

        "predicted_index":
            predicted_index,

        "predicted_class":
            predicted_class
    }


# ============================================================
# PROPOSED PNEUMONIA MODEL
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

    prediction = pneumonia_model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # OUTPUT CHECK
    # --------------------------------------------------------

    if (

        prediction.ndim != 2

        or

        prediction.shape[1] != 2

    ):

        raise ValueError(
            "Pneumonia model must produce "
            "exactly 2 outputs."
        )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        convert_to_probabilities(
            prediction[0]
        )
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


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # INDIVIDUAL PROBABILITIES
    # --------------------------------------------------------

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
    "Pneumonia Detection System"
)


st.write(
    "Chest X-ray verification followed by "
    "pneumonia classification using the proposed "
    "deep-learning model."
)


st.info(
    "Only grayscale chest X-ray images are accepted. "
    "Non-X-ray images are rejected before pneumonia "
    "classification."
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
        Basic Validation
        ↓
        Grayscale Check
        ↓
        X-ray Verifier
        ↓
        X-RAY VERIFICATION DECISION
        ↓
        X-RAY VERIFIED
        ↓
        Proposed Pneumonia Model
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
        "**Rejected**"
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
        "X-ray acceptance threshold:"
    )

    st.write(
        f"{XRAY_ACCEPT_THRESHOLD * 100:.0f}%"
    )

    st.write(
        "X-ray dominance requirement:"
    )

    st.write(
        str(REQUIRE_XRAY_DOMINANCE)
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
        "BatchNormalization → "
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
        "Analyze Image",
        type="primary",
        use_container_width=True
    ):

        # ====================================================
        # STEP 1
        # X-RAY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Chest X-ray Verification"
        )


        with st.spinner(
            "Analyzing image modality..."
        ):

            try:

                xray_decision = (
                    verify_xray(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "❌ X-ray verification failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # EXTRACT RESULTS
        # ====================================================

        xray_probability = (
            xray_decision[
                "xray_probability"
            ]
        )

        non_xray_probability = (
            xray_decision[
                "non_xray_probability"
            ]
        )

        is_xray = (
            xray_decision[
                "is_xray"
            ]
        )

        threshold_pass = (
            xray_decision[
                "threshold_pass"
            ]
        )

        dominance_pass = (
            xray_decision[
                "dominance_pass"
            ]
        )


        # ====================================================
        # X-RAY VERIFICATION DECISION
        # ====================================================

        st.markdown(
            "### X-RAY VERIFICATION DECISION"
        )


        # ----------------------------------------------------
        # PROBABILITY METRICS
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
            "Chest X-ray probability"
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


        st.write(
            "Non-X-ray probability"
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
        # DECISION CONDITIONS
        # ====================================================

        st.write(
            "**Decision checks**"
        )


        col1, col2 = st.columns(2)


        with col1:

            if threshold_pass:

                st.success(
                    "X-ray confidence threshold: PASS"
                )

            else:

                st.error(
                    "X-ray confidence threshold: FAIL"
                )


        with col2:

            if dominance_pass:

                st.success(
                    "X-ray dominance: PASS"
                )

            else:

                st.error(
                    "X-ray dominance: FAIL"
                )


        # ====================================================
        # FINAL X-RAY DECISION
        # ====================================================

        if is_xray:

            # ------------------------------------------------
            # VERIFIED
            # ------------------------------------------------

            st.success(
                "X-RAY VERIFIED"
            )


            st.write(
                f"Chest X-ray confidence: "
                f"**{xray_probability * 100:.2f}%**"
            )


            st.info(
                "The image passed X-ray verification. "
                "Pneumonia classification will now "
                "be performed."
            )


        else:

            # ------------------------------------------------
            # REJECTED
            # ------------------------------------------------

            st.error(
                "NON-X-RAY IMAGE DETECTED"
            )


            st.write(
                f"X-ray probability: "
                f"**{xray_probability * 100:.2f}%**"
            )


            st.write(
                f"Non-X-ray probability: "
                f"**{non_xray_probability * 100:.2f}%**"
            )


            # ------------------------------------------------
            # EXPLAIN DECISION
            # ------------------------------------------------

            if not threshold_pass:

                st.warning(
                    f"The X-ray probability "
                    f"({xray_probability * 100:.2f}%) "
                    f"is below the required "
                    f"{XRAY_ACCEPT_THRESHOLD * 100:.0f}% "
                    f"threshold."
                )


            if (
                REQUIRE_XRAY_DOMINANCE
                and
                not dominance_pass
            ):

                st.warning(
                    "The X-ray probability is not "
                    "greater than the Non-X-ray probability."
                )


            st.warning(
                "Pneumonia detection has been stopped."
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
        # STEP 2
        # PNEUMONIA DETECTION
        # ====================================================

        st.subheader(
            "Step 2 — Pneumonia Detection"
        )


        with st.spinner(
            "Analyzing verified chest X-ray..."
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
        # PNEUMONIA PROBABILITIES
        # ====================================================

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
                f"X-ray probability: "
                f"{xray_probability * 100:.2f}%"
            )

            st.write(
                f"Non-X-ray probability: "
                f"{non_xray_probability * 100:.2f}%"
            )

            st.write(
                "X-ray verification threshold:"
            )

            st.write(
                f"{XRAY_ACCEPT_THRESHOLD * 100:.0f}%"
            )

            st.write(
                "X-ray dominance requirement:"
            )

            st.write(
                str(REQUIRE_XRAY_DOMINANCE)
            )

            st.divider()

            st.write(
                "Proposed pneumonia architecture:"
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
            "This system is a research prototype "
            "for educational and research purposes. "
            "It is not intended to provide clinical "
            "diagnosis or replace professional "
            "medical evaluation."
        )
