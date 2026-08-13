import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL PATHS
# ============================================================

XRAY_MODEL_PATH = BASE_DIR / "best_xray_verifier.weights.h5"

# IMPORTANT:
# This is the exact filename present in your GitHub repository.
PNEUMONIA_MODEL_PATH = BASE_DIR / "best_xception_pneumonia_model.keras"


# ============================================================
# IMAGE SIZES
# ============================================================

XRAY_IMAGE_SIZE = (128, 128)
PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# X-RAY CLASS MAPPING
# ============================================================

# Based on the verifier configuration used previously:
#
# Class 0 = NON-XRAY
# Class 1 = XRAY
#
# If your training code used the opposite mapping, these two
# values must be reversed.

NON_XRAY_CLASS_INDEX = 0
XRAY_CLASS_INDEX = 1

XRAY_ACCEPT_THRESHOLD = 0.50


# ============================================================
# IMPORT X-RAY MODEL BUILDER
# ============================================================

try:
    from xray_model_builder import build_xray_classifier
except Exception as e:
    build_xray_classifier = None
    XRAY_IMPORT_ERROR = str(e)
else:
    XRAY_IMPORT_ERROR = None


# ============================================================
# HELPER: CHECK FILE
# ============================================================

def check_model_files():

    missing_files = []

    if not XRAY_MODEL_PATH.exists():
        missing_files.append(
            f"X-ray verifier: {XRAY_MODEL_PATH.name}"
        )

    if not PNEUMONIA_MODEL_PATH.exists():
        missing_files.append(
            f"Pneumonia model: {PNEUMONIA_MODEL_PATH.name}"
        )

    return missing_files


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    if build_xray_classifier is None:
        raise ImportError(
            "Could not import build_xray_classifier from "
            "xray_model_builder.py.\n\n"
            f"Original error:\n{XRAY_IMPORT_ERROR}"
        )

    if not XRAY_MODEL_PATH.exists():
        raise FileNotFoundError(
            "X-ray verifier model was not found.\n\n"
            f"Expected location:\n{XRAY_MODEL_PATH}"
        )

    # Build the same architecture used during training.
    model = build_xray_classifier()

    # Load trained weights.
    model.load_weights(str(XRAY_MODEL_PATH))

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    # IMPORTANT:
    # The repository contains:
    #
    # best_xception_pneumonia_model.keras
    #
    # NOT:
    #
    # best_exception_pneumonia_model.keras

    if not PNEUMONIA_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pneumonia model was not found.\n\n"
            f"Expected location:\n{PNEUMONIA_MODEL_PATH}\n\n"
            "Make sure "
            "'best_xception_pneumonia_model.keras' "
            "is committed to the same GitHub repository "
            "as streamlit_app.py."
        )

    # Load the complete saved model directly.
    #
    # compile=False prevents unnecessary optimizer/loss
    # reconstruction during deployment.
    model = tf.keras.models.load_model(
        str(PNEUMONIA_MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# IMAGE PREPROCESSING FOR X-RAY VERIFIER
# ============================================================

def preprocess_for_xray(image):

    # Convert to RGB because most CNN models expect 3 channels.
    image = image.convert("RGB")

    image = image.resize(XRAY_IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# IMAGE PREPROCESSING FOR PNEUMONIA MODEL
# ============================================================

def preprocess_for_pneumonia(image):

    image = image.convert("RGB")

    image = image.resize(PNEUMONIA_IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def verify_xray(image):

    model = load_xray_model()

    processed_image = preprocess_for_xray(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    prediction = np.asarray(prediction)

    # --------------------------------------------------------
    # Handle common binary model output formats
    # --------------------------------------------------------

    if prediction.ndim == 2 and prediction.shape[1] == 2:

        probabilities = prediction[0]

        non_xray_probability = float(
            probabilities[NON_XRAY_CLASS_INDEX]
        )

        xray_probability = float(
            probabilities[XRAY_CLASS_INDEX]
        )

    elif prediction.ndim == 2 and prediction.shape[1] == 1:

        # For sigmoid output:
        #
        # 0 = NON-XRAY
        # 1 = XRAY

        xray_probability = float(prediction[0][0])

        non_xray_probability = (
            1.0 - xray_probability
        )

    elif prediction.ndim == 1 and prediction.size == 2:

        probabilities = prediction

        non_xray_probability = float(
            probabilities[NON_XRAY_CLASS_INDEX]
        )

        xray_probability = float(
            probabilities[XRAY_CLASS_INDEX]
        )

    elif prediction.ndim == 1 and prediction.size == 1:

        xray_probability = float(
            prediction[0]
        )

        non_xray_probability = (
            1.0 - xray_probability
        )

    else:

        raise ValueError(
            "Unexpected X-ray verifier output shape: "
            f"{prediction.shape}"
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    is_xray = (
        xray_probability >= XRAY_ACCEPT_THRESHOLD
        and xray_probability > non_xray_probability
    )

    return {
        "is_xray": is_xray,
        "xray_probability": xray_probability,
        "non_xray_probability": non_xray_probability
    }


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(image):

    model = load_pneumonia_model()

    processed_image = preprocess_for_pneumonia(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    prediction = np.asarray(prediction)

    # --------------------------------------------------------
    # Normalize prediction shape
    # --------------------------------------------------------

    # Case 1:
    # Softmax output:
    # [Normal probability, Pneumonia probability]
    #
    # Case 2:
    # Sigmoid output:
    # [Pneumonia probability]
    #
    # The function supports both.

    if prediction.ndim == 2 and prediction.shape[1] == 2:

        probabilities = prediction[0]

        normal_probability = float(
            probabilities[0]
        )

        pneumonia_probability = float(
            probabilities[1]
        )

    elif prediction.ndim == 2 and prediction.shape[1] == 1:

        pneumonia_probability = float(
            prediction[0][0]
        )

        normal_probability = (
            1.0 - pneumonia_probability
        )

    elif prediction.ndim == 1 and prediction.size == 2:

        normal_probability = float(
            prediction[0]
        )

        pneumonia_probability = float(
            prediction[1]
        )

    elif prediction.ndim == 1 and prediction.size == 1:

        pneumonia_probability = float(
            prediction[0]
        )

        normal_probability = (
            1.0 - pneumonia_probability
        )

    else:

        raise ValueError(
            "Unexpected pneumonia model output shape: "
            f"{prediction.shape}"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if pneumonia_probability >= normal_probability:

        predicted_class = "Pneumonia"

    else:

        predicted_class = "Normal"

    return {
        "class": predicted_class,
        "pneumonia_probability": pneumonia_probability,
        "normal_probability": normal_probability
    }


# ============================================================
# HEADER
# ============================================================

st.title("Chest X-ray Pneumonia Detection System")

st.markdown(
    """
This system performs pneumonia detection using a two-stage
deep-learning pipeline.

**Stage 1:** Verify that the uploaded image is a chest X-ray.

**Stage 2:** If the image is a valid chest X-ray, classify it
as **Normal** or **Pneumonia**.
"""
)


# ============================================================
# MODEL FILE CHECK
# ============================================================

missing_files = check_model_files()

if missing_files:

    st.error("Required model file(s) are missing:")

    for file_name in missing_files:
        st.write(f"- `{file_name}`")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

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

    st.write(
        "**X-ray input size:** "
        "128 × 128"
    )

    st.write(
        "**Pneumonia input size:** "
        "224 × 224"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        )

        # Force image loading before model inference.
        image.load()

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Unable to read the uploaded image: {e}"
        )

        st.stop()

    # ========================================================
    # STAGE 1 — X-RAY VERIFICATION
    # ========================================================

    st.divider()

    st.subheader(
        "Step 1 — Chest X-ray Verification"
    )

    with st.spinner(
        "Checking whether the image is a chest X-ray..."
    ):

        try:

            xray_result = verify_xray(
                image
            )

        except Exception as e:

            st.error(
                "X-ray verification failed."
            )

            st.exception(e)

            st.stop()

    xray_probability = (
        xray_result["xray_probability"]
    )

    non_xray_probability = (
        xray_result["non_xray_probability"]
    )

    is_xray = xray_result["is_xray"]

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

    # ========================================================
    # X-RAY ACCEPTED
    # ========================================================

    if is_xray:

        st.success(
            "This image is identified as a Chest X-ray."
        )

        # ====================================================
        # STAGE 2 — PNEUMONIA DETECTION
        # ====================================================

        st.divider()

        st.subheader(
            "Step 2 — Pneumonia Detection"
        )

        with st.spinner(
            "Analyzing the chest X-ray for pneumonia..."
        ):

            try:

                pneumonia_result = predict_pneumonia(
                    image
                )

            except Exception as e:

                st.error(
                    "Pneumonia detection failed."
                )

                st.exception(e)

                st.stop()

        predicted_class = (
            pneumonia_result["class"]
        )

        pneumonia_probability = (
            pneumonia_result[
                "pneumonia_probability"
            ]
        )

        normal_probability = (
            pneumonia_result[
                "normal_probability"
            ]
        )

        # ====================================================
        # RESULT
        # ====================================================

        st.subheader(
            "Pneumonia Detection Result"
        )

        if predicted_class == "Pneumonia":

            st.error(
                "Prediction: Pneumonia"
            )

        else:

            st.success(
                "Prediction: Normal"
            )

        # ====================================================
        # PROBABILITIES
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Pneumonia Probability",
                f"{pneumonia_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Normal Probability",
                f"{normal_probability * 100:.2f}%"
            )

        st.progress(
            float(
                min(
                    max(
                        pneumonia_probability,
                        0.0
                    ),
                    1.0
                )
            )
        )

    # ========================================================
    # NOT AN X-RAY
    # ========================================================

    else:

        st.error(
            "This is not identified as a Chest X-ray image."
        )

        st.warning(
            "Pneumonia detection has been stopped because "
            "the uploaded image did not pass the chest "
            "X-ray verification stage."
        )
