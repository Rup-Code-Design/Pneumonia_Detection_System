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
# Medical Modality Classifier
#       ↓
# ┌──────────────────────────────────────┐
# │                                      │
# │  Chest X-ray → Continue              │
# │  CT          → Reject                │
# │  MRI         → Reject                │
# │  Other       → Reject                │
# │                                      │
# └──────────────────────────────────────┘
#       ↓
# X-ray Verifier
#       ↓
# X-RAY / NON-XRAY
#       ↓
# If verified X-ray
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

MODALITY_IMAGE_SIZE = (
    224,
    224
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================

# ------------------------------------------------------------
# Color image rejection
# ------------------------------------------------------------

COLOR_TOLERANCE = 5.0


# ------------------------------------------------------------
# Modality classifier
# ------------------------------------------------------------

MODALITY_CONFIDENCE_THRESHOLD = 0.90


# ------------------------------------------------------------
# X-ray verifier
# ------------------------------------------------------------

XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# X-RAY VERIFIER CLASS MAPPING
# ============================================================
#
# MUST MATCH xray_model_builder.py / TRAINING
#
# 0 = X-RAY
# 1 = NON-XRAY
#
# ============================================================

XRAY_CLASS_MAP = {
    0: "X-RAY",
    1: "NON-XRAY"
}


XRAY_CLASS_INDEX = 0

NON_XRAY_CLASS_INDEX = 1


# ============================================================
# MODALITY CLASS MAPPING
# ============================================================
#
# MUST EXACTLY MATCH THE TRAINING CLASS INDICES
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
# 3 = OTHER
#
# ============================================================

MODALITY_CLASS_MAP = {
    0: "CHEST_XRAY",
    1: "CT",
    2: "MRI",
    3: "OTHER"
}


CHEST_XRAY_CLASS_INDEX = 0


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================
#
# MUST MATCH PNEUMONIA MODEL TRAINING
#
# 0 = Normal
# 1 = Pneumonia
#
# ============================================================

PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}


NORMAL_CLASS_INDEX = 0

PNEUMONIA_CLASS_INDEX = 1


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# CHECK MODEL FILE
# ============================================================

def check_model_file(
    model_path,
    model_name
):

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            f"{model_name} was not found.\n\n"
            f"Expected location:\n"
            f"{model_path}\n\n"
            f"Make sure the file is uploaded to "
            f"the same directory as app.py."
        )


    if os.path.getsize(model_path) == 0:

        raise ValueError(
            f"{model_name} exists but the file is empty:\n"
            f"{model_path}"
        )


# ============================================================
# LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    check_model_file(
        MODALITY_MODEL_PATH,
        "Medical modality classifier weights"
    )


    model = build_modality_classifier(
        input_shape=(
            224,
            224,
            3
        ),
        num_classes=4
    )


    model.load_weights(
        MODALITY_MODEL_PATH
    )


    return model


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    check_model_file(
        XRAY_MODEL_PATH,
        "X-ray verifier weights"
    )


    model = build_xray_classifier(
        input_shape=(
            128,
            128,
            3
        )
    )


    model.load_weights(
        XRAY_MODEL_PATH
    )


    # --------------------------------------------------------
    # Verify architecture
    # --------------------------------------------------------

    if (
        model.output_shape[-1]
        != 2
    ):

        raise ValueError(
            "X-ray verifier must have exactly "
            "2 output classes.\n\n"
            f"Received output shape: "
            f"{model.output_shape}"
        )


    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    check_model_file(
        PNEUMONIA_MODEL_PATH,
        "Pneumonia model"
    )


    model = build_model(
        input_shape=(
            224,
            224,
            3
        )
    )


    model.load_weights(
        PNEUMONIA_MODEL_PATH
    )


    # --------------------------------------------------------
    # Verify architecture
    # --------------------------------------------------------

    if (
        model.output_shape[-1]
        != 2
    ):

        raise ValueError(
            "Pneumonia model must have exactly "
            "2 output classes.\n\n"
            f"Received output shape: "
            f"{model.output_shape}"
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
# IMAGE VALIDATION
# ============================================================

def validate_basic_image(
    image
):

    if image is None:

        return (
            False,
            "Image could not be loaded."
        )


    width, height = image.size


    # --------------------------------------------------------
    # Minimum resolution
    # --------------------------------------------------------

    if (
        width < 64
        or
        height < 64
    ):

        return (
            False,
            "Image resolution is too small."
        )


    # --------------------------------------------------------
    # Convert to grayscale for basic checks
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Blank image
    # --------------------------------------------------------

    standard_deviation = np.std(
        gray_array
    )


    if standard_deviation < 8:

        return (
            False,
            "Image appears blank or invalid."
        )


    # --------------------------------------------------------
    # Almost black
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray_array < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "Image is almost completely black."
        )


    # --------------------------------------------------------
    # Almost white
    # --------------------------------------------------------

    bright_ratio = np.mean(
        gray_array > 245
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
# COLOR IMAGE CHECK
# ============================================================

def is_color_image(
    image
):

    rgb_array = np.asarray(
        image.convert("RGB"),
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


    return (
        average_difference
        > COLOR_TOLERANCE
    )


# ============================================================
# IMAGE PREPROCESSING
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
# PROBABILITY CONVERSION
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

        prediction.shape[1] != 4

    ):

        raise ValueError(
            "Modality classifier must output "
            "exactly 4 classes.\n\n"
            f"Received shape: "
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
        MODALITY_CLASS_MAP.get(
            predicted_index,
            "UNKNOWN"
        )
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    return {

        "index":
            predicted_index,

        "class":
            predicted_class,

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
            "exactly 2 classes.\n\n"
            f"Received shape: "
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

        "is_xray":
            is_xray,

        "result":
            "X-RAY"
            if is_xray
            else "NON-XRAY",

        "xray_probability":
            xray_probability,

        "non_xray_probability":
            non_xray_probability,

        "confidence":
            (
                xray_probability
                if is_xray
                else non_xray_probability
            ),

        "predicted_index":
            predicted_index,

        "predicted_class":
            predicted_class,

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
            "exactly 2 classes.\n\n"
            f"Received shape: "
            f"{prediction.shape}"
        )


    probabilities = (
        convert_to_probabilities(
            prediction[0]
        )
    )


    normal_probability = float(
        probabilities[
            NORMAL_CLASS_INDEX
        ]
    )


    pneumonia_probability = float(
        probabilities[
            PNEUMONIA_CLASS_INDEX
        ]
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    diagnosis = (
        PNEUMONIA_CLASS_MAP[
            predicted_index
        ]
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    return {

        "diagnosis":
            diagnosis,

        "confidence":
            confidence,

        "normal_probability":
            normal_probability,

        "pneumonia_probability":
            pneumonia_probability
    }


# ============================================================
# ADD HISTORY
# ============================================================

def add_history(
    text
):

    if text not in st.session_state.history:

        st.session_state.history.append(
            text
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)


st.write(
    """
This system follows a sequential verification pipeline:

**Image → Color Check → Modality Classification →
X-ray Verification → Pneumonia Detection**
"""
)


st.info(
    "Only grayscale chest X-ray images are accepted. "
    "CT, MRI, color images, non-X-ray images, and "
    "unsupported images are rejected."
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
1. Upload image
2. Basic validation
3. Color-image rejection
4. Medical modality classification
5. Chest X-ray verification
6. Pneumonia classification
7. Normal / Pneumonia result
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

    st.write(
        "3 — Other"
    )


    st.divider()


    st.write(
        "**X-ray Verifier**"
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
        "**Modality input:** 224 × 224 × 3"
    )

    st.write(
        "**X-ray verifier input:** 128 × 128 × 3"
    )

    st.write(
        "**Pneumonia input:** 224 × 224 × 3"
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
3 = OTHER
"""
    )


    st.write(
        "**X-ray mapping:**"
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


    st.divider()


    st.write(
        "Modality confidence threshold:"
    )

    st.write(
        f"{MODALITY_CONFIDENCE_THRESHOLD * 100:.0f}%"
    )


    st.write(
        "X-ray verification threshold:"
    )

    st.write(
        f"{XRAY_CONFIDENCE_THRESHOLD * 100:.0f}%"
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
# PROCESS UPLOADED IMAGE
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


        # Keep original mode available
        original_mode = image.mode


        # Convert only after reading
        image_rgb = image.convert(
            "RGB"
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        st.subheader(
            "Uploaded Image"
        )


        st.image(
            image_rgb,
            caption=uploaded_file.name,
            use_container_width=True
        )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        analyze = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        )


        if analyze:

            # =================================================
            # STEP 1 — BASIC VALIDATION
            # =================================================

            st.subheader(
                "Step 1 — Basic Image Validation"
            )


            valid, message = (
                validate_basic_image(
                    image_rgb
                )
            )


            if not valid:

                st.error(
                    message
                )


                add_history(
                    f"Rejected - Invalid image - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            st.success(
                "Basic image validation passed."
            )


            # =================================================
            # STEP 2 — COLOR IMAGE REJECTION
            # =================================================

            st.subheader(
                "Step 2 — Color Image Verification"
            )


            color_detected = (
                is_color_image(
                    image_rgb
                )
            )


            if color_detected:

                st.error(
                    "This is not a Chest X-ray image."
                )


                st.warning(
                    "Color images are not accepted. "
                    "Please upload a grayscale chest X-ray."
                )


                add_history(
                    f"Rejected - Color image - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            st.success(
                "Grayscale image confirmed."
            )


            # =================================================
            # STEP 3 — MEDICAL MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 3 — Medical Image Modality Verification"
            )


            with st.spinner(
                "Determining image modality..."
            ):

                modality_result = (
                    predict_modality(
                        image_rgb
                    )
                )


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


            # =================================================
            # DISPLAY MODALITY RESULT
            # =================================================

            if modality_class == "CHEST_XRAY":

                if (
                    modality_confidence
                    >= MODALITY_CONFIDENCE_THRESHOLD
                ):

                    st.success(
                        "Chest X-ray modality detected."
                    )


                    st.metric(
                        "Chest X-ray Confidence",
                        f"{modality_confidence * 100:.2f}%"
                    )

                else:

                    st.error(
                        "Chest X-ray confidence is too low."
                    )


                    st.write(
                        f"Confidence: "
                        f"{modality_confidence * 100:.2f}%"
                    )


                    st.warning(
                        "The image cannot be reliably "
                        "verified as a chest X-ray."
                    )


                    add_history(
                        f"Rejected - Low modality confidence - "
                        f"{uploaded_file.name}"
                    )


                    st.stop()


            elif modality_class == "CT":

                st.error(
                    "CT scan detected."
                )


                st.write(
                    f"CT confidence: "
                    f"{modality_confidence * 100:.2f}%"
                )


                st.warning(
                    "This system accepts only "
                    "chest X-ray images."
                )


                add_history(
                    f"Rejected - CT - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            elif modality_class == "MRI":

                st.error(
                    "MRI image detected."
                )


                st.write(
                    f"MRI confidence: "
                    f"{modality_confidence * 100:.2f}%"
                )


                st.warning(
                    "This system accepts only "
                    "chest X-ray images."
                )


                add_history(
                    f"Rejected - MRI - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            elif modality_class == "OTHER":

                st.error(
                    "Unsupported medical image detected."
                )


                st.write(
                    f"Confidence: "
                    f"{modality_confidence * 100:.2f}%"
                )


                st.warning(
                    "Please upload a chest X-ray image."
                )


                add_history(
                    f"Rejected - Other modality - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            else:

                st.error(
                    "Unknown image modality."
                )


                add_history(
                    f"Rejected - Unknown modality - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            # =================================================
            # MODALITY PROBABILITY DISPLAY
            # =================================================

            with st.expander(
                "Modality Class Probabilities"
            ):

                modality_table = {

                    "Chest X-ray":
                        float(
                            modality_probabilities[0]
                        ),

                    "CT":
                        float(
                            modality_probabilities[1]
                        ),

                    "MRI":
                        float(
                            modality_probabilities[2]
                        ),

                    "Other":
                        float(
                            modality_probabilities[3]
                        )
                }


                for name, probability in (
                    modality_table.items()
                ):

                    st.write(
                        f"**{name}:** "
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


            # =================================================
            # STEP 4 — X-RAY VERIFIER
            # =================================================

            st.subheader(
                "Step 4 — Chest X-ray Verification"
            )


            with st.spinner(
                "Verifying chest X-ray..."
            ):

                xray_result = (
                    verify_xray(
                        image_rgb
                    )
                )


            # =================================================
            # EXTRACT RESULTS
            # =================================================

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


            xray_is_verified = (
                xray_result[
                    "is_xray"
                ]
            )


            # =================================================
            # X-RAY RESULT
            # =================================================

            if not xray_is_verified:

                st.error(
                    "This is not a Chest X-ray image."
                )


                st.write(
                    f"X-ray probability: "
                    f"{xray_probability * 100:.2f}%"
                )


                st.write(
                    f"Non-X-ray probability: "
                    f"{non_xray_probability * 100:.2f}%"
                )


                st.warning(
                    "Pneumonia detection has been stopped."
                )


                add_history(
                    f"Rejected - X-ray verification failed - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            # =================================================
            # X-RAY VERIFIED
            # =================================================

            st.success(
                "Chest X-ray verified successfully."
            )


            st.metric(
                "X-ray Verification Confidence",
                f"{xray_confidence * 100:.2f}%"
            )


            # =================================================
            # X-RAY PROBABILITIES
            # =================================================

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Chest X-ray",
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
                    "Non-X-ray",
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


            # =================================================
            # STEP 5 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 5 — Pneumonia Detection"
            )


            with st.spinner(
                "Analyzing verified chest X-ray..."
            ):

                pneumonia_result = (
                    predict_pneumonia(
                        image_rgb
                    )
                )


            diagnosis = (
                pneumonia_result[
                    "diagnosis"
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


            # =================================================
            # DIAGNOSIS
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


            st.write(
                f"**Diagnosis:** {diagnosis}"
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


            # =================================================
            # HISTORY
            # =================================================

            add_history(
                f"{diagnosis} - "
                f"{uploaded_file.name}"
            )


            # =================================================
            # TECHNICAL DETAILS
            # =================================================

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "**Original image mode:** "
                    f"{original_mode}"
                )


                st.write(
                    "**Verified modality:** "
                    "Chest X-ray"
                )


                st.write(
                    "**Modality confidence:** "
                    f"{modality_confidence * 100:.2f}%"
                )


                st.write(
                    "**X-ray confidence:** "
                    f"{xray_confidence * 100:.2f}%"
                )


                st.write(
                    "**X-ray probability:** "
                    f"{xray_probability * 100:.2f}%"
                )


                st.write(
                    "**Non-X-ray probability:** "
                    f"{non_xray_probability * 100:.2f}%"
                )


                st.write(
                    "**Diagnosis:** "
                    f"{diagnosis}"
                )


                st.write(
                    "**Diagnosis confidence:** "
                    f"{diagnosis_confidence * 100:.2f}%"
                )


                st.write(
                    "**Modality classes:** "
                    "Chest X-ray / CT / MRI / Other"
                )


                st.write(
                    "**X-ray classes:** "
                    "X-ray / Non-X-ray"
                )


                st.write(
                    "**Pneumonia classes:** "
                    "Normal / Pneumonia"
                )


            # =================================================
            # PDF REPORT
            # =================================================

            st.divider()


            st.subheader(
                "Diagnostic Report"
            )


            # -------------------------------------------------
            # Safe filename
            # -------------------------------------------------

            clean_filename = (
                os.path.splitext(
                    uploaded_file.name
                )[0]
            )


            clean_filename = "".join(
                c
                for c in clean_filename
                if c.isalnum()
                or c in (
                    "_",
                    "-",
                    " "
                )
            )


            if not clean_filename:

                clean_filename = "image"


            # -------------------------------------------------
            # Create PDF
            # -------------------------------------------------

            pdf = FPDF()

            pdf.set_auto_page_break(
                auto=True,
                margin=15
            )

            pdf.add_page()


            # -------------------------------------------------
            # Title
            # -------------------------------------------------

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
                27,
                200,
                27
            )


            pdf.ln(
                10
            )


            # -------------------------------------------------
            # Report information
            # -------------------------------------------------

            report_rows = [

                (
                    "File Name",
                    uploaded_file.name
                ),

                (
                    "Image Modality",
                    "Chest X-ray"
                ),

                (
                    "Modality Confidence",
                    f"{modality_confidence * 100:.2f}%"
                ),

                (
                    "X-ray Verification",
                    "X-RAY"
                ),

                (
                    "X-ray Confidence",
                    f"{xray_confidence * 100:.2f}%"
                ),

                (
                    "Diagnosis",
                    diagnosis
                ),

                (
                    "Diagnosis Confidence",
                    f"{diagnosis_confidence * 100:.2f}%"
                )
            ]


            for label, value in report_rows:

                pdf.set_font(
                    "Arial",
                    "B",
                    11
                )


                pdf.cell(
                    55,
                    9,
                    f"{label}:",
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


            # -------------------------------------------------
            # Disclaimer
            # -------------------------------------------------

            pdf.ln(
                15
            )


            pdf.set_font(
                "Arial",
                "I",
                9
            )


            pdf.multi_cell(
                0,
                6,
                (
                    "Disclaimer: This AI-generated result "
                    "is intended for research and educational "
                    "purposes only. It is not a clinical "
                    "diagnosis and does not replace evaluation "
                    "by a qualified medical professional."
                )
            )


            # -------------------------------------------------
            # Generate PDF bytes
            # -------------------------------------------------

            pdf_output = pdf.output(
                dest="S"
            )


            if isinstance(
                pdf_output,
                str
            ):

                pdf_bytes = (
                    pdf_output.encode(
                        "latin-1"
                    )
                )

            else:

                pdf_bytes = bytes(
                    pdf_output
                )


            # -------------------------------------------------
            # Download
            # -------------------------------------------------

            st.download_button(
                label=(
                    "Download Diagnostic Report"
                ),
                data=pdf_bytes,
                file_name=(
                    f"Report_{clean_filename}.pdf"
                ),
                mime="application/pdf"
            )


    except Exception as e:

        st.error(
            "An error occurred while "
            "processing the image."
        )


        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Research prototype for educational and research "
    "purposes. This system is not intended for clinical "
    "diagnosis or treatment decisions."
)
