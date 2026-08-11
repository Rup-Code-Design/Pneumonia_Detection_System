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

XRAY_WEIGHTS = BASE_DIR / "best_xray_verifier.weights.h5"

PNEUMONIA_WEIGHTS = BASE_DIR / "best_pneumonia_model.weights.h5"


# ============================================================
# IMAGE SETTINGS
# ============================================================

XRAY_INPUT_SIZE = (128, 128)
PNEUMONIA_INPUT_SIZE = (224, 224)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not XRAY_WEIGHTS.exists():

    st.error(
        "X-ray verifier weights were not found:\n\n"
        f"{XRAY_WEIGHTS}"
    )

    st.stop()


if not PNEUMONIA_WEIGHTS.exists():

    st.error(
        "Pneumonia model weights were not found:\n\n"
        f"{PNEUMONIA_WEIGHTS}"
    )

    st.stop()


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
    # num_classes=1 is required because the trained
    # pneumonia model has:
    #
    # Dense(1, activation="sigmoid",
    #       name="pneumonia_probability")

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=1
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

    st.exception(e)

    st.stop()


try:

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error(
        "Pneumonia model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING FOR X-RAY VERIFIER
# ============================================================

def preprocess_for_xray_verifier(image):

    image = image.convert("RGB")

    image = image.resize(
        XRAY_INPUT_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Convert 0-255 → 0-1
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

    image = image.resize(
        PNEUMONIA_INPUT_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Convert 0-255 → 0-1
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

    image_array = preprocess_for_xray_verifier(
        image
    )

    predictions = xray_model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

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

    # Binary sigmoid output
    pneumonia_probability = float(
        prediction[0][0]
    )

    if pneumonia_probability >= 0.5:

        predicted_class = "Pneumonia"

    else:

        predicted_class = "Normal"

    return (
        predicted_class,
        pneumonia_probability
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

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )


        # ====================================================
        # STEP 1: X-RAY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Chest X-ray Verification"
        )

        with st.spinner(
            "Checking whether the image is a Chest X-ray..."
        ):

            xray_class, xray_confidence, xray_probabilities = (
                verify_xray(image)
            )


        # ----------------------------------------------------
        # CLASS DEFINITIONS
        #
        # XRay_Verifier:
        # 0 = Chest X-ray
        # 1 = Non-X-ray
        # ----------------------------------------------------

        if xray_class == 0:

            st.success(
                f"Chest X-ray detected "
                f"(confidence: {xray_confidence * 100:.2f}%)"
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
            # STEP 2: PNEUMONIA DETECTION
            # ================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing the Chest X-ray..."
            ):

                result, pneumonia_probability = (
                    predict_pneumonia(image)
                )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

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
                    f"{(1 - pneumonia_probability) * 100:.2f}%"
                )

            else:

                st.success(
                    "Normal — No Pneumonia Detected"
                )

                st.write(
                    f"Normal probability: "
                    f"{(1 - pneumonia_probability) * 100:.2f}%"
                )

                st.write(
                    f"Pneumonia probability: "
                    f"{pneumonia_probability * 100:.2f}%"
                )


            # ================================================
            # DISCLAIMER
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
```
