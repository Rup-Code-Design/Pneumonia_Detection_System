# ============================================================
# app.py
# X-ray Verification + Pneumonia Detection
# ============================================================

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# ============================================================
# IMPORT MODEL BUILDERS
# ============================================================

from xray_model_builder import build_xray_classifier
from pneumonia_model_builder import build_model


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩻",
    layout="centered"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL WEIGHT FILES
# ============================================================

XRAY_WEIGHTS = (
    BASE_DIR / "best_xray_verifier.weights.h5"
)

PNEUMONIA_WEIGHTS = (
    BASE_DIR / "best_pneumonia_model.weights.h5"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

XRAY_INPUT_SIZE = (128, 128)
PNEUMONIA_INPUT_SIZE = (224, 224)


# ============================================================
# MODEL FILE CHECK
# ============================================================

def check_model_file(path, model_name):

    if not path.exists():

        st.error(
            f"{model_name} weights were not found."
        )

        st.code(
            str(path)
        )

        st.stop()

    if path.stat().st_size == 0:

        st.error(
            f"{model_name} weights file is empty."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# CHECK REQUIRED WEIGHT FILES
# ============================================================

check_model_file(
    XRAY_WEIGHTS,
    "X-ray verifier"
)

check_model_file(
    PNEUMONIA_WEIGHTS,
    "Pneumonia model"
)


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_verifier():

    model = build_xray_classifier(
        input_shape=(128, 128, 3)
    )

    model.load_weights(
        str(XRAY_WEIGHTS)
    )

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    # IMPORTANT:
    #
    # The corrected pneumonia_model_builder.py contains:
    #
    # def build_model(input_shape=(224, 224, 3)):
    #
    # Therefore DO NOT pass num_classes=1 here.
    #
    # The output is fixed as:
    #
    # Dense(
    #     1,
    #     activation="sigmoid",
    #     name="pneumonia_probability"
    # )

    model = build_model(
        input_shape=(224, 224, 3)
    )

    model.load_weights(
        str(PNEUMONIA_WEIGHTS)
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model = load_xray_verifier()

except Exception as e:

    st.error(
        "X-ray verifier model loading failed."
    )

    st.error(
        "Check that best_xray_verifier.weights.h5 "
        "was created using the same xray_model_builder.py."
    )

    st.exception(e)

    st.stop()


try:

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error(
        "Pneumonia model loading failed."
    )

    st.error(
        "The architecture in pneumonia_model_builder.py "
        "must exactly match the architecture used to train "
        "best_pneumonia_model.weights.h5."
    )

    st.error(
        "Expected pneumonia output: "
        "Dense(1, activation='sigmoid')."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY MODEL OUTPUT SHAPES
# ============================================================

xray_output_shape = xray_model.output_shape
pneumonia_output_shape = pneumonia_model.output_shape


if xray_output_shape[-1] != 2:

    st.error(
        "X-ray verifier output mismatch."
    )

    st.write(
        f"Expected output shape: (None, 2)"
    )

    st.write(
        f"Actual output shape: {xray_output_shape}"
    )

    st.stop()


if pneumonia_output_shape[-1] != 1:

    st.error(
        "Pneumonia model output mismatch."
    )

    st.write(
        "Expected output shape: (None, 1)"
    )

    st.write(
        f"Actual output shape: {pneumonia_output_shape}"
    )

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    target_size
):

    image = image.convert("RGB")

    image = image.resize(
        target_size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # 0-255 -> 0-1
    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# X-RAY PREPROCESSING
# ============================================================

def preprocess_for_xray_verifier(image):

    return preprocess_image(
        image,
        XRAY_INPUT_SIZE
    )


# ============================================================
# PNEUMONIA PREPROCESSING
# ============================================================

def preprocess_for_pneumonia(image):

    return preprocess_image(
        image,
        PNEUMONIA_INPUT_SIZE
    )


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def verify_xray(image):

    image_array = preprocess_for_xray_verifier(
        image
    )

    predictions = xray_model.predict(
        image_array,
        verbose=0
    )

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32
    )

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(image):

    image_array = preprocess_for_pneumonia(
        image
    )

    prediction = pneumonia_model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------------
    # Expected output:
    #
    # [[pneumonia_probability]]
    #
    # 0 = Normal
    # 1 = Pneumonia
    # --------------------------------------------------------

    pneumonia_probability = float(
        prediction[0][0]
    )

    # Safety check
    pneumonia_probability = np.clip(
        pneumonia_probability,
        0.0,
        1.0
    )

    normal_probability = (
        1.0 - pneumonia_probability
    )

    if pneumonia_probability >= 0.5:

        predicted_class = "Pneumonia"

    else:

        predicted_class = "Normal"

    return (
        predicted_class,
        pneumonia_probability,
        normal_probability
    )


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "Chest X-ray Pneumonia Detection System"
)

st.write(
    "Upload an image. The system first verifies whether "
    "the image is a Chest X-ray. Pneumonia detection is "
    "performed only for verified Chest X-ray images."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "X-ray verifier input: "
        "128 × 128 × 3"
    )

    st.write(
        "Pneumonia model input: "
        "224 × 224 × 3"
    )

    st.write(
        "X-ray verifier output: "
        "2-class softmax"
    )

    st.write(
        "Pneumonia model output: "
        "1-unit sigmoid"
    )

    st.write(
        "Pneumonia threshold: 0.50"
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
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        )

        # Force image loading before closing/upload lifecycle
        image.load()

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )


        # ====================================================
        # STEP 1 — X-RAY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Chest X-ray Verification"
        )

        with st.spinner(
            "Checking whether the image is a Chest X-ray..."
        ):

            (
                xray_class,
                xray_confidence,
                xray_probabilities
            ) = verify_xray(image)


        # ----------------------------------------------------
        # X-RAY CLASS DEFINITIONS
        #
        # 0 = Chest X-ray
        # 1 = Non-X-ray
        # ----------------------------------------------------

        if xray_class == 0:

            st.success(
                f"Chest X-ray detected "
                f"(confidence: "
                f"{xray_confidence * 100:.2f}%)"
            )

            st.write(
                f"Chest X-ray probability: "
                f"{xray_probabilities[0] * 100:.2f}%"
            )

            st.write(
                f"Non-X-ray probability: "
                f"{xray_probabilities[1] * 100:.2f}%"
            )


            # ================================================
            # STEP 2 — PNEUMONIA DETECTION
            # ================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing the Chest X-ray..."
            ):

                (
                    result,
                    pneumonia_probability,
                    normal_probability
                ) = predict_pneumonia(image)


            # ================================================
            # PNEUMONIA RESULT
            # ================================================

            if result == "Pneumonia":

                st.error(
                    "Pneumonia Detected"
                )

                st.write(
                    f"Pneumonia probability: "
                    f"{pneumonia_probability * 100:.2f}%"
                )

                st.write(
                    f"Normal probability: "
                    f"{normal_probability * 100:.2f}%"
                )

            else:

                st.success(
                    "Normal — No Pneumonia Detected"
                )

                st.write(
                    f"Normal probability: "
                    f"{normal_probability * 100:.2f}%"
                )

                st.write(
                    f"Pneumonia probability: "
                    f"{pneumonia_probability * 100:.2f}%"
                )


            # ================================================
            # MEDICAL DISCLAIMER
            # ================================================

            st.info(
                "This system is intended for research and "
                "educational purposes and is not a substitute "
                "for professional medical diagnosis."
            )


        else:

            # =================================================
            # NON-X-RAY
            # =================================================

            st.error(
                "This is not a Chest X-ray image."
            )

            st.write(
                f"Non-X-ray confidence: "
                f"{xray_confidence * 100:.2f}%"
            )

            st.warning(
                "Pneumonia detection was not performed "
                "because the uploaded image was not "
                "classified as a Chest X-ray."
            )


    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)
