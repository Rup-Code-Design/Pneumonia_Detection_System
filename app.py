# ============================================================
# app.py
# Medical Image Modality Verification
# +
# Chest X-ray Pneumonia Detection
# ============================================================

import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from modality_model_builder import (
    build_modality_classifier
)

from pneumonia_model_builder import (
    build_model
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL FILES
# ============================================================

MODALITY_WEIGHTS = (
    BASE_DIR /
    "best_modality_classifier.weights.h5"
)

PNEUMONIA_WEIGHTS = (
    BASE_DIR /
    "best_pneumonia_model.weights.h5"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

MODALITY_INPUT_SIZE = (128, 128)

PNEUMONIA_INPUT_SIZE = (224, 224)


# ============================================================
# CHEST X-RAY THRESHOLD
# ============================================================
#
# An image will be accepted as a chest X-ray when the
# CHEST_XRAY probability is >= 40%.
#
# ============================================================

CHEST_XRAY_THRESHOLD = 0.50


# ============================================================
# MODALITY CLASS MAPPING
# ============================================================

MODALITY_CLASS_NAMES = {
    0: "CHEST_XRAY",
    1: "CT",
    2: "MRI",
    3: "OTHER"
}


# ============================================================
# CHECK MODEL FILE
# ============================================================

def check_model_file(path, model_name):

    if not path.exists():

        st.error(
            f"{model_name} weights were not found."
        )

        st.code(str(path))

        st.stop()

    if path.stat().st_size == 0:

        st.error(
            f"{model_name} weights file is empty."
        )

        st.code(str(path))

        st.stop()


# ============================================================
# CHECK REQUIRED MODEL FILES
# ============================================================

check_model_file(
    MODALITY_WEIGHTS,
    "Medical modality classifier"
)

check_model_file(
    PNEUMONIA_WEIGHTS,
    "Pneumonia model"
)


# ============================================================
# LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    model = build_modality_classifier(
        input_shape=(128, 128, 3),
        num_classes=4
    )

    model.load_weights(
        str(MODALITY_WEIGHTS)
    )

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

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

    modality_model = load_modality_model()

except Exception as e:

    st.error(
        "Medical modality classifier loading failed."
    )

    st.error(
        "Make sure that modality_model_builder.py "
        "and best_modality_classifier.weights.h5 "
        "were created from the same architecture."
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
        "Make sure that pneumonia_model_builder.py "
        "and best_pneumonia_model.weights.h5 "
        "use exactly the same architecture."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY MODEL OUTPUTS
# ============================================================

modality_output_shape = (
    modality_model.output_shape
)

pneumonia_output_shape = (
    pneumonia_model.output_shape
)


# ============================================================
# CHECK MODALITY OUTPUT
# ============================================================

if modality_output_shape[-1] != 4:

    st.error(
        "Medical modality classifier output mismatch."
    )

    st.write(
        "Expected: 4 classes"
    )

    st.write(
        f"Actual: {modality_output_shape}"
    )

    st.stop()


# ============================================================
# CHECK PNEUMONIA OUTPUT
# ============================================================

if pneumonia_output_shape[-1] != 1:

    st.error(
        "Pneumonia model output mismatch."
    )

    st.write(
        "Expected: 1 sigmoid output"
    )

    st.write(
        f"Actual: {pneumonia_output_shape}"
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

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# MODALITY PREPROCESSING
# ============================================================

def preprocess_for_modality(image):

    return preprocess_image(
        image,
        MODALITY_INPUT_SIZE
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
# MODALITY PREDICTION
# ============================================================

def predict_modality(image):

    image_array = (
        preprocess_for_modality(image)
    )

    predictions = (
        modality_model.predict(
            image_array,
            verbose=0
        )
    )

    predictions = np.asarray(
        predictions
    )

    if (
        predictions.ndim != 2
        or predictions.shape[1] != 4
    ):

        raise ValueError(
            "Invalid modality classifier output."
        )

    probabilities = (
        predictions[0]
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # HANDLE LOGITS IF NECESSARY
    # --------------------------------------------------------

    if not (
        np.all(probabilities >= 0.0)
        and
        np.all(probabilities <= 1.0)
        and
        np.isclose(
            np.sum(probabilities),
            1.0,
            atol=1e-3
        )
    ):

        probabilities = (
            tf.nn.softmax(
                probabilities
            ).numpy()
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = (
        MODALITY_CLASS_NAMES.get(
            predicted_index,
            "UNKNOWN"
        )
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        predicted_index,
        confidence,
        probabilities
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(image):

    image_array = (
        preprocess_for_pneumonia(image)
    )

    prediction = (
        pneumonia_model.predict(
            image_array,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )

    if prediction.size != 1:

        raise ValueError(
            "Pneumonia model must produce "
            "one sigmoid probability."
        )

    pneumonia_probability = float(
        np.squeeze(prediction)
    )

    pneumonia_probability = float(
        np.clip(
            pneumonia_probability,
            0.0,
            1.0
        )
    )

    normal_probability = (
        1.0 -
        pneumonia_probability
    )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    if pneumonia_probability >= 0.50:

        diagnosis = "Pneumonia"

        diagnosis_confidence = (
            pneumonia_probability
        )

    else:

        diagnosis = "Normal"

        diagnosis_confidence = (
            normal_probability
        )

    return (
        diagnosis,
        pneumonia_probability,
        normal_probability,
        diagnosis_confidence
    )


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "Chest X-ray Pneumonia Detection System"
)

st.write(
    "Upload an image. The system first verifies "
    "whether it is a chest X-ray. Pneumonia "
    "detection is performed only for verified "
    "chest X-ray images."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "Medical modality classifier: "
        "CHEST X-RAY / CT / MRI / OTHER"
    )

    st.write(
        "Modality input: 128 × 128 × 3"
    )

    st.write(
        "Pneumonia model input: 224 × 224 × 3"
    )

    st.write(
        "Chest X-ray acceptance threshold: 40%"
    )

    st.write(
        "Pneumonia decision threshold: 50%"
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
    help="Upload a chest X-ray image."
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        )

        image.load()

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        analyze = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        )

        if analyze:

            # =================================================
            # STEP 1
            # =================================================

            st.subheader(
                "Step 1 — Image Verification"
            )

            with st.spinner(
                "Analyzing image modality..."
            ):

                (
                    modality_class,
                    modality_index,
                    modality_confidence,
                    modality_probabilities
                ) = predict_modality(
                    image
                )

            # =================================================
            # GET PROBABILITIES
            # =================================================

            chest_xray_probability = float(
                modality_probabilities[0]
            )

            ct_probability = float(
                modality_probabilities[1]
            )

            mri_probability = float(
                modality_probabilities[2]
            )

            other_probability = float(
                modality_probabilities[3]
            )

            # =================================================
            # DISPLAY DETECTED MODALITY
            # =================================================

            st.subheader(
                "Medical Image Modality"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Detected Type",
                    modality_class
                )

            with col2:

                st.metric(
                    "Chest X-ray Probability",
                    (
                        f"{chest_xray_probability * 100:.2f}%"
                    )
                )

            # =================================================
            # DISPLAY PROBABILITIES
            # =================================================

            probability_data = {

                "Chest X-ray":
                    f"{chest_xray_probability * 100:.2f}%",

                "CT":
                    f"{ct_probability * 100:.2f}%",

                "MRI":
                    f"{mri_probability * 100:.2f}%",

                "Other":
                    f"{other_probability * 100:.2f}%"
            }

            with st.expander(
                "Modality probabilities"
            ):

                st.table(
                    probability_data
                )

            # =================================================
            # CHEST X-RAY THRESHOLD DECISION
            # =================================================
            #
            # IMPORTANT:
            #
            # Only this threshold has been changed.
            #
            # 0.40 = 40%
            #
            # If Chest X-ray probability >= 40%,
            # continue to pneumonia detection.
            #
            # =================================================

            if (
                chest_xray_probability
                >= CHEST_XRAY_THRESHOLD
            ):

                # =============================================
                # CHEST X-RAY ACCEPTED
                # =============================================

                st.success(
                    "✅ Chest X-ray image detected."
                )

                st.write(
                    "Chest X-ray probability: "
                    f"{chest_xray_probability * 100:.2f}%"
                )

            else:

                # =============================================
                # IMAGE REJECTED
                # =============================================

                if modality_class == "CT":

                    st.error(
                        "❌ This image appears to be "
                        "a CT image, not a chest X-ray."
                    )

                elif modality_class == "MRI":

                    st.error(
                        "❌ This image appears to be "
                        "an MRI image, not a chest X-ray."
                    )

                elif modality_class == "OTHER":

                    st.error(
                        "❌ This image was not identified "
                        "as a chest X-ray."
                    )

                else:

                    st.error(
                        "❌ Chest X-ray probability "
                        "is below the required threshold."
                    )

                st.write(
                    "Chest X-ray probability: "
                    f"{chest_xray_probability * 100:.2f}%"
                )

                st.write(
                    "Required threshold: 40.00%"
                )

                st.warning(
                    "Please upload a valid chest X-ray image."
                )

                st.info(
                    "Pneumonia detection was not performed."
                )

                st.stop()

            # =================================================
            # STEP 2 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing the chest X-ray..."
            ):

                (
                    diagnosis,
                    pneumonia_probability,
                    normal_probability,
                    diagnosis_confidence
                ) = predict_pneumonia(
                    image
                )

            # =================================================
            # FINAL DIAGNOSIS
            # =================================================

            if diagnosis == "Pneumonia":

                st.error(
                    "Pneumonia"
                )

            else:

                st.success(
                    "Normal"
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
                    "Diagnosis",
                    diagnosis
                )

            with result_col2:

                st.metric(
                    "Diagnosis Confidence",
                    (
                        f"{diagnosis_confidence * 100:.2f}%"
                    )
                )

            # =================================================
            # PNEUMONIA PROBABILITIES
            # =================================================

            st.write(
                "Pneumonia probability: "
                f"{pneumonia_probability * 100:.2f}%"
            )

            st.write(
                "Normal probability: "
                f"{normal_probability * 100:.2f}%"
            )

            # =================================================
            # DISCLAIMER
            # =================================================

            st.info(
                "This system is intended for research "
                "and educational purposes only and is "
                "not a substitute for professional "
                "medical diagnosis."
            )

    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)
