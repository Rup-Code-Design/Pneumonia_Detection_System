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
#
# This is important for Streamlit Cloud.
#
# All model files are searched relative to the directory
# containing this streamlit_app.py file.
#
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATHS
# ============================================================

XRAY_KERAS_PATH = os.path.join(
    BASE_DIR,
    "best_xray_verifier.keras"
)


XRAY_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "best_xray_verifier.weights.h5"
)


MODALITY_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "best_modality_classifier.weights.h5"
)


PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_exception_pneumonia_model.keras"
)


# ============================================================
# IMAGE SIZES
# ============================================================
#
# IMPORTANT:
#
# Modality classifier:
#       128 x 128
#
# X-ray verifier:
#       128 x 128
#
# Pneumonia model:
#       224 x 224
#
# ============================================================

XRAY_IMAGE_SIZE = (
    128,
    128
)

MODALITY_IMAGE_SIZE = (
    128,
    128
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# IMAGE VALIDATION SETTINGS
# ============================================================

COLOR_TOLERANCE = 5.0

MODALITY_CONFIDENCE_THRESHOLD = 0.90

XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# MODALITY CLASS MAPPING
# ============================================================
#
# IMPORTANT:
#
# This must match the class order used during modality
# classifier training.
#
# ============================================================

MODALITY_CLASS_MAP = {
    0: "CHEST_XRAY",
    1: "CT",
    2: "MRI",
    3: "OTHER"
}


# ============================================================
# X-RAY VERIFIER CLASS MAPPING
# ============================================================
#
# Based on your previous configuration:
#
# 0 = X-RAY
# 1 = NON-XRAY
#
# ============================================================

XRAY_CLASS_MAP = {
    0: "X-RAY",
    1: "NON-XRAY"
}


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================
#
# Expected:
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
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# FILE EXISTENCE CHECK
# ============================================================

def check_required_files():

    missing_files = []

    required_files = {
        "X-ray verifier Keras model": XRAY_KERAS_PATH,
        "X-ray verifier weights": XRAY_WEIGHTS_PATH,
        "Modality classifier weights": MODALITY_WEIGHTS_PATH,
        "Pneumonia model": PNEUMONIA_MODEL_PATH
    }

    for name, path in required_files.items():

        if not os.path.isfile(path):

            missing_files.append(
                f"{name}: {path}"
            )

    return missing_files


# ============================================================
# LOAD MODALITY CLASSIFIER
# ============================================================
#
# Architecture:
#
# best_xray_verifier.keras
#          ↓
# pretrained feature extractor
#          ↓
# modality classification head
#          ↓
# 4 classes
#
# Input:
#       128 x 128 x 3
#
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_WEIGHTS_PATH
    ):

        raise FileNotFoundError(
            "Medical modality classifier weights were not found.\n\n"
            f"Expected location:\n"
            f"{MODALITY_WEIGHTS_PATH}\n\n"
            "Make sure best_modality_classifier.weights.h5 "
            "is committed to the same GitHub repository."
        )


    if not os.path.isfile(
        XRAY_KERAS_PATH
    ):

        raise FileNotFoundError(
            "Pretrained X-ray verifier model was not found.\n\n"
            f"Expected location:\n"
            f"{XRAY_KERAS_PATH}\n\n"
            "The latest modality_model_builder.py uses "
            "best_xray_verifier.keras as its pretrained backbone."
        )


    try:

        # ----------------------------------------------------
        # BUILD EXACT MODALITY ARCHITECTURE
        # ----------------------------------------------------

        model = build_modality_classifier(
            input_shape=(
                128,
                128,
                3
            ),
            num_classes=4,
            xray_model_path=XRAY_KERAS_PATH,
            freeze_backbone=True
        )


        # ----------------------------------------------------
        # LOAD MODALITY WEIGHTS
        # ----------------------------------------------------

        model.load_weights(
            MODALITY_WEIGHTS_PATH
        )


    except Exception as e:

        raise RuntimeError(
            "Medical modality classifier weights could not "
            "be loaded.\n\n"
            f"File:\n{MODALITY_WEIGHTS_PATH}\n\n"
            "The modality model expects 128x128x3 input.\n"
            "The architecture generated by the current "
            "modality_model_builder.py must exactly match "
            "the architecture used when the weights were trained.\n\n"
            f"Original error:\n{e}"
        )


    # --------------------------------------------------------
    # FINAL INPUT VALIDATION
    # --------------------------------------------------------

    if tuple(
        model.input_shape[1:]
    ) != (
        128,
        128,
        3
    ):

        raise RuntimeError(
            "Modality model input shape is incorrect.\n\n"
            f"Expected: (128, 128, 3)\n"
            f"Found: {model.input_shape}"
        )


    # --------------------------------------------------------
    # FINAL OUTPUT VALIDATION
    # --------------------------------------------------------

    if (
        len(model.output_shape) != 2
        or
        model.output_shape[-1] != 4
    ):

        raise RuntimeError(
            "Modality classifier output is incorrect.\n\n"
            "Expected output: (None, 4)\n"
            f"Found: {model.output_shape}"
        )


    return model


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    if not os.path.isfile(
        XRAY_WEIGHTS_PATH
    ):

        raise FileNotFoundError(
            "X-ray verifier weights were not found.\n\n"
            f"Expected location:\n"
            f"{XRAY_WEIGHTS_PATH}\n\n"
            "Make sure best_xray_verifier.weights.h5 "
            "is committed to the repository."
        )


    try:

        model = build_xray_classifier(
            input_shape=(
                128,
                128,
                3
            )
        )


        model.load_weights(
            XRAY_WEIGHTS_PATH
        )


    except Exception as e:

        raise RuntimeError(
            "X-ray verifier weights could not be loaded.\n\n"
            f"File:\n{XRAY_WEIGHTS_PATH}\n\n"
            "The xray_model_builder.py architecture must "
            "exactly match the architecture used during training.\n\n"
            f"Original error:\n{e}"
        )


    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if tuple(
        model.input_shape[1:]
    ) != (
        128,
        128,
        3
    ):

        raise RuntimeError(
            "X-ray verifier input shape mismatch.\n\n"
            f"Expected: (128, 128, 3)\n"
            f"Found: {model.input_shape}"
        )


    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================
#
# IMPORTANT:
#
# best_exception_pneumonia_model.keras is a COMPLETE Keras
# model, therefore load_model() is used.
#
# DO NOT use:
#
#     build_model()
#     model.load_weights("best_exception_pneumonia_model.keras")
#
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(
        PNEUMONIA_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Pneumonia model was not found.\n\n"
            f"Expected location:\n"
            f"{PNEUMONIA_MODEL_PATH}\n\n"
            "Make sure best_exception_pneumonia_model.keras "
            "is committed to the same GitHub repository."
        )


    try:

        # ----------------------------------------------------
        # LOAD COMPLETE KERAS MODEL
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )


    except Exception as e:

        raise RuntimeError(
            "Pneumonia model could not be loaded.\n\n"
            f"File:\n{PNEUMONIA_MODEL_PATH}\n\n"
            "This .keras file must be a complete saved Keras "
            "model compatible with the current TensorFlow/Keras "
            "environment.\n\n"
            f"Original error:\n{e}"
        )


    # --------------------------------------------------------
    # CHECK INPUT SHAPE
    # --------------------------------------------------------

    expected_input = (
        224,
        224,
        3
    )


    actual_input = tuple(
        model.input_shape[1:]
    )


    if actual_input != expected_input:

        raise RuntimeError(
            "Pneumonia model input shape mismatch.\n\n"
            f"Expected: {expected_input}\n"
            f"Found: {actual_input}\n\n"
            "Your Streamlit preprocessing and the trained "
            "pneumonia model must use the same input size."
        )


    return model


# ============================================================
# LOAD MODELS
# ============================================================

missing_files = check_required_files()


if missing_files:

    st.error(
        "Required model files are missing."
    )

    for item in missing_files:

        st.write(
            f"❌ {item}"
        )

    st.stop()


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
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)


st.markdown(
    """
Upload an image to perform automated image verification
and pneumonia classification.

**Processing pipeline**

1. Reject color images.
2. Verify the medical image modality.
3. Accept only chest X-ray images.
4. Verify the image using the X-ray verifier.
5. Run pneumonia detection.
6. Report **Normal** or **Pneumonia**.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        "Medical Modality Classifier"
    )

    st.write(
        "Input: 128 × 128 × 3"
    )

    st.write(
        "Classes: Chest X-ray / CT / MRI / Other"
    )

    st.divider()

    st.write(
        "X-ray Verifier"
    )

    st.write(
        "Input: 128 × 128 × 3"
    )

    st.divider()

    st.write(
        "Pneumonia Classifier"
    )

    st.write(
        "Input: 224 × 224 × 3"
    )

    st.write(
        "Classes: Normal / Pneumonia"
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


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload a grayscale chest X-ray image."
    )
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # READ FILE
        # ====================================================

        file_bytes = (
            uploaded_file.getvalue()
        )


        if len(file_bytes) == 0:

            st.error(
                "The uploaded file is empty."
            )

            st.stop()


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )


        # ----------------------------------------------------
        # CHECK IMAGE MODE
        # ----------------------------------------------------

        original_mode = image.mode


        # ----------------------------------------------------
        # CONVERT TO RGB ONLY FOR MODEL PROCESSING
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )


        image_array = np.array(
            image
        )


        # ====================================================
        # BASIC VALIDATION
        # ====================================================

        if (
            image_array is None
            or
            image_array.size == 0
        ):

            st.error(
                "Could not read the uploaded image."
            )

            st.stop()


        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # ====================================================
        # IMAGE INFORMATION
        # ====================================================

        with st.expander(
            "Image Information"
        ):

            st.write(
                f"Filename: {uploaded_file.name}"
            )

            st.write(
                f"Original mode: {original_mode}"
            )

            st.write(
                f"Image size: {image.size[0]} × {image.size[1]}"
            )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                # =================================================
                # STEP 1
                # COLOR IMAGE REJECTION
                # =================================================

                st.subheader(
                    "Step 1 — Image Validation"
                )


                # -------------------------------------------------
                # RGB CHANNELS
                # -------------------------------------------------

                rgb_image = (
                    image_array.astype(
                        np.float32
                    )
                )


                red_channel = (
                    rgb_image[:, :, 0]
                )

                green_channel = (
                    rgb_image[:, :, 1]
                )

                blue_channel = (
                    rgb_image[:, :, 2]
                )


                # -------------------------------------------------
                # CHANNEL DIFFERENCES
                # -------------------------------------------------

                rg_difference = np.mean(
                    np.abs(
                        red_channel
                        -
                        green_channel
                    )
                )


                gb_difference = np.mean(
                    np.abs(
                        green_channel
                        -
                        blue_channel
                    )
                )


                rb_difference = np.mean(
                    np.abs(
                        red_channel
                        -
                        blue_channel
                    )
                )


                channel_difference = (
                    rg_difference
                    +
                    gb_difference
                    +
                    rb_difference
                ) / 3.0


                # -------------------------------------------------
                # COLOR IMAGE REJECTION
                # -------------------------------------------------

                if (
                    channel_difference
                    >
                    COLOR_TOLERANCE
                ):

                    st.error(
                        "This is not a Chest X-ray image."
                    )

                    st.warning(
                        "Color images are not accepted. "
                        "Please upload a grayscale chest X-ray."
                    )


                    history_entry = (
                        "Rejected - Color image - "
                        f"{uploaded_file.name}"
                    )


                    st.session_state.history.append(
                        history_entry
                    )


                    st.stop()


                # -------------------------------------------------
                # GRAYSCALE CONFIRMED
                # -------------------------------------------------

                st.success(
                    "Grayscale image detected."
                )


                st.write(
                    "RGB channel difference: "
                    f"{channel_difference:.4f}"
                )


                # =================================================
                # STEP 2
                # MEDICAL IMAGE MODALITY
                # =================================================

                st.subheader(
                    "Step 2 — Medical Image Modality Verification"
                )


                # -------------------------------------------------
                # RESIZE TO 128x128
                # -------------------------------------------------
                #
                # IMPORTANT:
                #
                # The modality classifier expects:
                #
                # (None, 128, 128, 3)
                #
                # -------------------------------------------------

                modality_image = cv2.resize(
                    image_array,
                    MODALITY_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )


                modality_image = (
                    modality_image.astype(
                        np.float32
                    )
                    /
                    255.0
                )


                modality_input = np.expand_dims(
                    modality_image,
                    axis=0
                )


                # -------------------------------------------------
                # SAFETY CHECK
                # -------------------------------------------------

                if modality_input.shape != (
                    1,
                    128,
                    128,
                    3
                ):

                    raise RuntimeError(
                        "Modality input shape is incorrect.\n"
                        f"Actual: {modality_input.shape}\n"
                        "Expected: (1, 128, 128, 3)"
                    )


                # -------------------------------------------------
                # PREDICTION
                # -------------------------------------------------

                modality_prediction = (
                    modality_model.predict(
                        modality_input,
                        verbose=0
                    )
                )


                modality_prediction = np.asarray(
                    modality_prediction
                )


                # -------------------------------------------------
                # OUTPUT VALIDATION
                # -------------------------------------------------

                if (
                    modality_prediction.ndim != 2
                    or
                    modality_prediction.shape[1] != 4
                ):

                    raise RuntimeError(
                        "Unexpected modality classifier output.\n"
                        f"Output shape: "
                        f"{modality_prediction.shape}\n"
                        "Expected: (1, 4)"
                    )


                # -------------------------------------------------
                # PROBABILITIES
                # -------------------------------------------------

                modality_scores = (
                    modality_prediction[0]
                    .astype(
                        np.float64
                    )
                )


                # -------------------------------------------------
                # NORMALIZE IF NECESSARY
                # -------------------------------------------------

                if not (
                    np.all(
                        modality_scores >= 0.0
                    )
                    and
                    np.all(
                        modality_scores <= 1.0
                    )
                    and
                    np.isclose(
                        np.sum(
                            modality_scores
                        ),
                        1.0,
                        atol=1e-3
                    )
                ):

                    modality_probabilities = (
                        tf.nn.softmax(
                            modality_scores
                        ).numpy()
                    )

                else:

                    modality_probabilities = (
                        modality_scores
                    )


                # -------------------------------------------------
                # CLASS
                # -------------------------------------------------

                modality_class_index = int(
                    np.argmax(
                        modality_probabilities
                    )
                )


                modality_confidence = float(
                    modality_probabilities[
                        modality_class_index
                    ]
                )


                modality_result = (
                    MODALITY_CLASS_MAP.get(
                        modality_class_index,
                        "UNKNOWN"
                    )
                )


                # -------------------------------------------------
                # DISPLAY MODALITY
                # -------------------------------------------------

                st.write(
                    f"**Detected modality:** "
                    f"{modality_result}"
                )


                st.write(
                    f"**Confidence:** "
                    f"{modality_confidence * 100:.2f}%"
                )


                # =================================================
                # MODALITY REJECTION
                # =================================================

                if modality_result != "CHEST_XRAY":

                    if modality_result == "CT":

                        st.error(
                            "CT scan detected."
                        )

                    elif modality_result == "MRI":

                        st.error(
                            "MRI image detected."
                        )

                    elif modality_result == "OTHER":

                        st.error(
                            "Unsupported image detected."
                        )

                    else:

                        st.error(
                            "Unknown image modality."
                        )


                    st.warning(
                        "This system accepts only chest X-ray images."
                    )


                    history_entry = (
                        f"Rejected - "
                        f"{modality_result} - "
                        f"{uploaded_file.name}"
                    )


                    st.session_state.history.append(
                        history_entry
                    )


                    st.stop()


                # =================================================
                # LOW CONFIDENCE REJECTION
                # =================================================

                if (
                    modality_confidence
                    <
                    MODALITY_CONFIDENCE_THRESHOLD
                ):

                    st.error(
                        "Chest X-ray confidence is too low."
                    )


                    st.warning(
                        "Please upload a clear chest X-ray image."
                    )


                    history_entry = (
                        "Rejected - Low modality confidence - "
                        f"{uploaded_file.name}"
                    )


                    st.session_state.history.append(
                        history_entry
                    )


                    st.stop()


                # =================================================
                # CHEST X-RAY CONFIRMED
                # =================================================

                st.success(
                    "Chest X-ray modality confirmed."
                )


                # =================================================
                # DISPLAY MODALITY PROBABILITIES
                # =================================================

                modality_col1, modality_col2 = (
                    st.columns(2)
                )


                with modality_col1:

                    st.metric(
                        "Chest X-ray",
                        f"{modality_probabilities[0] * 100:.2f}%"
                    )


                with modality_col2:

                    st.metric(
                        "CT",
                        f"{modality_probabilities[1] * 100:.2f}%"
                    )


                modality_col3, modality_col4 = (
                    st.columns(2)
                )


                with modality_col3:

                    st.metric(
                        "MRI",
                        f"{modality_probabilities[2] * 100:.2f}%"
                    )


                with modality_col4:

                    st.metric(
                        "Other",
                        f"{modality_probabilities[3] * 100:.2f}%"
                    )


                # =================================================
                # STEP 3
                # X-RAY VERIFICATION
                # =================================================

                st.subheader(
                    "Step 3 — Chest X-ray Verification"
                )


                # -------------------------------------------------
                # RESIZE TO 128x128
                # -------------------------------------------------

                verifier_image = cv2.resize(
                    image_array,
                    XRAY_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )


                verifier_image = (
                    verifier_image.astype(
                        np.float32
                    )
                    /
                    255.0
                )


                verifier_input = np.expand_dims(
                    verifier_image,
                    axis=0
                )


                # -------------------------------------------------
                # SAFETY CHECK
                # -------------------------------------------------

                if verifier_input.shape != (
                    1,
                    128,
                    128,
                    3
                ):

                    raise RuntimeError(
                        "X-ray verifier input shape is incorrect.\n"
                        f"Actual: {verifier_input.shape}\n"
                        "Expected: (1, 128, 128, 3)"
                    )


                # -------------------------------------------------
                # PREDICTION
                # -------------------------------------------------

                verifier_prediction = (
                    xray_model.predict(
                        verifier_input,
                        verbose=0
                    )
                )


                verifier_prediction = np.asarray(
                    verifier_prediction
                )


                # -------------------------------------------------
                # OUTPUT VALIDATION
                # -------------------------------------------------

                if (
                    verifier_prediction.ndim != 2
                    or
                    verifier_prediction.shape[1] != 2
                ):

                    raise RuntimeError(
                        "Unexpected X-ray verifier output.\n"
                        f"Output shape: "
                        f"{verifier_prediction.shape}\n"
                        "Expected: (1, 2)"
                    )


                # -------------------------------------------------
                # SCORES
                # -------------------------------------------------

                verifier_scores = (
                    verifier_prediction[0]
                    .astype(
                        np.float64
                    )
                )


                # -------------------------------------------------
                # CONVERT TO PROBABILITIES
                # -------------------------------------------------

                if not (
                    np.all(
                        verifier_scores >= 0.0
                    )
                    and
                    np.all(
                        verifier_scores <= 1.0
                    )
                    and
                    np.isclose(
                        np.sum(
                            verifier_scores
                        ),
                        1.0,
                        atol=1e-3
                    )
                ):

                    verifier_probabilities = (
                        tf.nn.softmax(
                            verifier_scores
                        ).numpy()
                    )

                else:

                    verifier_probabilities = (
                        verifier_scores
                    )


                # -------------------------------------------------
                # PREDICT CLASS
                # -------------------------------------------------

                verifier_class_index = int(
                    np.argmax(
                        verifier_probabilities
                    )
                )


                verifier_confidence = float(
                    verifier_probabilities[
                        verifier_class_index
                    ]
                )


                verifier_result = (
                    XRAY_CLASS_MAP.get(
                        verifier_class_index,
                        "UNKNOWN"
                    )
                )


                # =================================================
                # REJECT NON-X-RAY
                # =================================================

                if (
                    verifier_result != "X-RAY"
                    or
                    verifier_confidence
                    <
                    XRAY_CONFIDENCE_THRESHOLD
                ):

                    st.error(
                        "This is not a Chest X-ray image."
                    )


                    st.warning(
                        "The X-ray verifier rejected the image."
                    )


                    history_entry = (
                        "Rejected - X-ray verification - "
                        f"{uploaded_file.name}"
                    )


                    st.session_state.history.append(
                        history_entry
                    )


                    st.stop()


                # =================================================
                # X-RAY CONFIRMED
                # =================================================

                st.success(
                    "Chest X-ray verified."
                )


                st.write(
                    "X-ray verification confidence: "
                    f"{verifier_confidence * 100:.2f}%"
                )


                # -------------------------------------------------
                # X-RAY PROBABILITIES
                # -------------------------------------------------

                xray_probability = float(
                    verifier_probabilities[0]
                )


                non_xray_probability = float(
                    verifier_probabilities[1]
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Chest X-ray",
                        f"{xray_probability * 100:.2f}%"
                    )


                with col2:

                    st.metric(
                        "Non-X-ray",
                        f"{non_xray_probability * 100:.2f}%"
                    )


                # =================================================
                # STEP 4
                # PNEUMONIA DETECTION
                # =================================================

                st.subheader(
                    "Step 4 — Pneumonia Detection"
                )


                # -------------------------------------------------
                # RESIZE TO 224x224
                # -------------------------------------------------

                pneumonia_image = cv2.resize(
                    image_array,
                    PNEUMONIA_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )


                pneumonia_image = (
                    pneumonia_image.astype(
                        np.float32
                    )
                    /
                    255.0
                )


                pneumonia_input = np.expand_dims(
                    pneumonia_image,
                    axis=0
                )


                # -------------------------------------------------
                # SAFETY CHECK
                # -------------------------------------------------

                if pneumonia_input.shape != (
                    1,
                    224,
                    224,
                    3
                ):

                    raise RuntimeError(
                        "Pneumonia input shape is incorrect.\n"
                        f"Actual: {pneumonia_input.shape}\n"
                        "Expected: (1, 224, 224, 3)"
                    )


                # -------------------------------------------------
                # PREDICTION
                # -------------------------------------------------

                prediction = (
                    pneumonia_model.predict(
                        pneumonia_input,
                        verbose=0
                    )
                )


                prediction = np.asarray(
                    prediction
                )


                # =================================================
                # HANDLE PNEUMONIA OUTPUT
                # =================================================

                if (
                    prediction.ndim != 2
                ):

                    raise RuntimeError(
                        "Unexpected pneumonia model output.\n"
                        f"Output shape: "
                        f"{prediction.shape}"
                    )


                # -------------------------------------------------
                # TWO-CLASS OUTPUT
                # -------------------------------------------------

                if prediction.shape[1] == 2:

                    pneumonia_scores = (
                        prediction[0]
                        .astype(
                            np.float64
                        )
                    )


                    if not (
                        np.all(
                            pneumonia_scores >= 0.0
                        )
                        and
                        np.all(
                            pneumonia_scores <= 1.0
                        )
                        and
                        np.isclose(
                            np.sum(
                                pneumonia_scores
                            ),
                            1.0,
                            atol=1e-3
                        )
                    ):

                        pneumonia_probabilities = (
                            tf.nn.softmax(
                                pneumonia_scores
                            ).numpy()
                        )

                    else:

                        pneumonia_probabilities = (
                            pneumonia_scores
                        )


                    normal_probability = float(
                        pneumonia_probabilities[0]
                    )


                    pneumonia_probability = float(
                        pneumonia_probabilities[1]
                    )


                # -------------------------------------------------
                # SINGLE SIGMOID OUTPUT
                # -------------------------------------------------

                elif prediction.shape[1] == 1:

                    raw_probability = float(
                        prediction[0][0]
                    )


                    # If the model returned a logit,
                    # convert it using sigmoid.

                    if (
                        raw_probability < 0.0
                        or
                        raw_probability > 1.0
                    ):

                        raw_probability = float(
                            tf.sigmoid(
                                raw_probability
                            ).numpy()
                        )


                    pneumonia_probability = (
                        raw_probability
                    )


                    normal_probability = (
                        1.0
                        -
                        pneumonia_probability
                    )


                else:

                    raise RuntimeError(
                        "Unsupported pneumonia model output.\n"
                        f"Output shape: "
                        f"{prediction.shape}\n"
                        "Expected either (1, 2) or (1, 1)."
                    )


                # =================================================
                # FINAL DIAGNOSIS
                # =================================================

                if (
                    pneumonia_probability
                    >=
                    normal_probability
                ):

                    diagnosis = "Pneumonia"

                    diagnosis_confidence = (
                        pneumonia_probability
                    )

                else:

                    diagnosis = "Normal"

                    diagnosis_confidence = (
                        normal_probability
                    )


                # =================================================
                # DISPLAY DIAGNOSIS
                # =================================================

                if diagnosis == "Pneumonia":

                    st.error(
                        "Diagnosis: Pneumonia"
                    )

                else:

                    st.success(
                        "Diagnosis: Normal"
                    )


                # =================================================
                # FINAL RESULT
                # =================================================

                st.subheader(
                    "Final Result"
                )


                result_col1, result_col2 = (
                    st.columns(2)
                )


                with result_col1:

                    st.metric(
                        "Normal",
                        f"{normal_probability * 100:.2f}%"
                    )


                with result_col2:

                    st.metric(
                        "Pneumonia",
                        f"{pneumonia_probability * 100:.2f}%"
                    )


                st.write(
                    f"**Final Diagnosis:** {diagnosis}"
                )


                st.write(
                    "**Diagnosis Confidence:** "
                    f"{diagnosis_confidence * 100:.2f}%"
                )


                # =================================================
                # HISTORY
                # =================================================

                history_entry = (
                    f"{diagnosis} - "
                    f"{uploaded_file.name}"
                )


                st.session_state.history.append(
                    history_entry
                )


                # =================================================
                # DIAGNOSTIC REPORT
                # =================================================

                st.divider()

                st.subheader(
                    "Diagnostic Report"
                )


                st.write(
                    f"**File:** "
                    f"{uploaded_file.name}"
                )


                st.write(
                    "**Image Modality:** "
                    "Chest X-ray"
                )


                st.write(
                    "**Modality Confidence:** "
                    f"{modality_confidence * 100:.2f}%"
                )


                st.write(
                    "**X-ray Verification:** "
                    f"{verifier_result}"
                )


                st.write(
                    "**X-ray Confidence:** "
                    f"{verifier_confidence * 100:.2f}%"
                )


                st.write(
                    "**Diagnosis:** "
                    f"{diagnosis}"
                )


                st.write(
                    "**Diagnosis Confidence:** "
                    f"{diagnosis_confidence * 100:.2f}%"
                )


                # =================================================
                # PDF REPORT
                # =================================================

                try:

                    # ------------------------------------------------
                    # CLEAN FILENAME
                    # ------------------------------------------------

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


                    # Remove problematic characters
                    clean_filename = (
                        clean_filename
                        .replace(
                            " ",
                            "_"
                        )
                    )


                    # ------------------------------------------------
                    # CREATE PDF
                    # ------------------------------------------------

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


                    pdf.ln(
                        10
                    )


                    # ------------------------------------------------
                    # FILE
                    # ------------------------------------------------

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


                    # ------------------------------------------------
                    # MODALITY
                    # ------------------------------------------------

                    pdf.set_font(
                        "Arial",
                        "B",
                        12
                    )


                    pdf.cell(
                        55,
                        10,
                        "Image Modality:",
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


                    # ------------------------------------------------
                    # MODALITY CONFIDENCE
                    # ------------------------------------------------

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


                    # ------------------------------------------------
                    # X-RAY STATUS
                    # ------------------------------------------------

                    pdf.set_font(
                        "Arial",
                        "B",
                        12
                    )


                    pdf.cell(
                        55,
                        10,
                        "X-ray Status:",
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
                        verifier_result,
                        ln=True
                    )


                    # ------------------------------------------------
                    # X-RAY CONFIDENCE
                    # ------------------------------------------------

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
                        f"{verifier_confidence * 100:.2f}%",
                        ln=True
                    )


                    # ------------------------------------------------
                    # DIAGNOSIS
                    # ------------------------------------------------

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


                    # ------------------------------------------------
                    # DIAGNOSIS CONFIDENCE
                    # ------------------------------------------------

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


                    pdf.ln(
                        15
                    )


                    # ------------------------------------------------
                    # DISCLAIMER
                    # ------------------------------------------------

                    pdf.set_font(
                        "Arial",
                        "I",
                        10
                    )


                    pdf.multi_cell(
                        0,
                        7,
                        "Disclaimer: This AI-generated result "
                        "is intended for research purposes only "
                        "and does not replace professional "
                        "medical diagnosis."
                    )


                    # ------------------------------------------------
                    # PDF OUTPUT
                    # ------------------------------------------------

                    pdf_output = pdf.output()


                    if isinstance(
                        pdf_output,
                        str
                    ):

                        pdf_output = (
                            pdf_output.encode(
                                "latin-1"
                            )
                        )


                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    st.download_button(
                        label=(
                            "Download Diagnostic Report"
                        ),
                        data=pdf_output,
                        file_name=(
                            f"Report_{clean_filename}.pdf"
                        ),
                        mime="application/pdf"
                    )


                except Exception as pdf_error:

                    st.warning(
                        "The diagnosis was completed, "
                        "but the PDF report could not be generated."
                    )


                    st.exception(
                        pdf_error
                    )


    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(
            e
        )
