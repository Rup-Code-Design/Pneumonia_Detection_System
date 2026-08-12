# ============================================================
# app.py
# Chest X-ray Verification + Pneumonia Detection
# ============================================================

import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# MODEL BUILDER
# ============================================================

from modality_model_builder import (
    build_modality_classifier
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

BASE_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# MODEL PATHS
# ============================================================

MODALITY_MODEL_PATH = (
    BASE_DIR /
    "best_modality_classifier.weights.h5"
)

PNEUMONIA_MODEL_PATH = (
    BASE_DIR /
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# IMAGE SIZE
# ============================================================

MODALITY_IMAGE_SIZE = (
    128,
    128
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# THRESHOLDS
# ============================================================

# Chest X-ray acceptance threshold
CHEST_XRAY_THRESHOLD = 0.40

# Pneumonia classification threshold
PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# MODALITY CLASS MAPPING
# ============================================================
#
# This MUST match the class_indices generated during
# modality-model training.
#
# Expected:
#
# CHEST_XRAY = 0
# CT         = 1
# MRI        = 2
# OTHER      = 3
#
# ============================================================

MODALITY_CLASS_NAMES = {

    0: "CHEST_XRAY",

    1: "CT",

    2: "MRI",

    3: "OTHER"
}


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🫁 Chest X-ray Pneumonia Detection System"
)

st.markdown(
    """
Upload a medical image. The system first verifies whether
the image is a **chest X-ray**. Pneumonia detection is
performed only when the image passes the chest X-ray
verification stage.
"""
)


# ============================================================
# MODEL FILE CHECK
# ============================================================

def check_model_file(
    model_path,
    model_name
):

    if not model_path.exists():

        st.error(
            f"{model_name} was not found."
        )

        st.code(
            str(model_path)
        )

        st.stop()

    if model_path.stat().st_size == 0:

        st.error(
            f"{model_name} file is empty."
        )

        st.stop()


# ============================================================
# CHECK MODEL FILES
# ============================================================

check_model_file(
    MODALITY_MODEL_PATH,
    "Modality classifier"
)

check_model_file(
    PNEUMONIA_MODEL_PATH,
    "Pneumonia model"
)


# ============================================================
# LOAD MODALITY CLASSIFIER
# ============================================================

@st.cache_resource
def load_modality_model():

    model = build_modality_classifier(

        input_shape=(
            128,
            128,
            3
        ),

        num_classes=4
    )

    model.load_weights(
        str(MODALITY_MODEL_PATH)
    )

    return model


# ============================================================
# LOAD COMPLETE PNEUMONIA MODEL
# ============================================================
#
# IMPORTANT:
#
# best_xception_pneumonia_model.keras contains the complete
# trained model.
#
# We therefore DO NOT rebuild it using build_model().
#
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    model = tf.keras.models.load_model(

        str(
            PNEUMONIA_MODEL_PATH
        ),

        compile=False
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    modality_model = (
        load_modality_model()
    )

except Exception as e:

    st.error(
        "Failed to load the modality classifier."
    )

    st.error(
        "Make sure that "
        "modality_model_builder.py and "
        "best_modality_classifier.weights.h5 "
        "use exactly the same architecture."
    )

    st.exception(e)

    st.stop()


try:

    pneumonia_model = (
        load_pneumonia_model()
    )

except Exception as e:

    st.error(
        "Failed to load the pneumonia model."
    )

    st.error(
        "Make sure that "
        "best_xception_pneumonia_model.keras "
        "is the complete trained pneumonia model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(
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
# MODALITY PREDICTION
# ============================================================

def predict_modality(
    image
):

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_image = prepare_image(

        image,

        MODALITY_IMAGE_SIZE
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = (
        modality_model.predict(
            input_image,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )

    # --------------------------------------------------------
    # GET RAW SCORES
    # --------------------------------------------------------

    raw_scores = (
        prediction[0]
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # CONVERT TO PROBABILITY
    # --------------------------------------------------------

    # Model uses softmax, but this also safely handles
    # logits if encountered.

    if (
        np.all(
            raw_scores >= 0
        )
        and
        np.all(
            raw_scores <= 1
        )
        and
        np.isclose(
            np.sum(raw_scores),
            1.0,
            atol=1e-3
        )
    ):

        probabilities = (
            raw_scores
        )

    else:

        probabilities = (
            tf.nn.softmax(
                raw_scores
            ).numpy()
        )

    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_class = (
        MODALITY_CLASS_NAMES.get(
            predicted_index,
            "UNKNOWN"
        )
    )

    predicted_confidence = float(
        probabilities[
            predicted_index
        ]
    )

    return (
        probabilities,
        predicted_index,
        predicted_class,
        predicted_confidence
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    image
):

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_image = prepare_image(

        image,

        PNEUMONIA_IMAGE_SIZE
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = (
        pneumonia_model.predict(
            input_image,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )

    # --------------------------------------------------------
    # GET SINGLE PROBABILITY
    # --------------------------------------------------------

    if prediction.size != 1:

        raise ValueError(
            "The loaded pneumonia model does not "
            "produce a single probability."
        )

    pneumonia_probability = float(
        np.squeeze(
            prediction
        )
    )

    # --------------------------------------------------------
    # CLAMP
    # --------------------------------------------------------

    pneumonia_probability = float(
        np.clip(
            pneumonia_probability,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # NORMAL PROBABILITY
    # --------------------------------------------------------

    normal_probability = (
        1.0 -
        pneumonia_probability
    )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    if (
        pneumonia_probability
        >= PNEUMONIA_THRESHOLD
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

    return (
        diagnosis,
        pneumonia_probability,
        normal_probability,
        diagnosis_confidence
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "System Information"
):

    st.write(
        "Modality classifier:"
    )

    st.write(
        "CHEST_XRAY / CT / MRI / OTHER"
    )

    st.write(
        "Chest X-ray acceptance threshold: 40%"
    )

    st.write(
        "Pneumonia decision threshold: 50%"
    )

    st.write(
        "Modality input size: 128 × 128"
    )

    st.write(
        "Pneumonia input size: 224 × 224"
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
        "Upload a chest X-ray image."
    )
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
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

        analyze_button = st.button(

            "Analyze Image",

            type="primary",

            use_container_width=True
        )

        if analyze_button:

            # =================================================
            # STEP 1
            # =================================================

            st.subheader(
                "Step 1 — Chest X-ray Verification"
            )

            with st.spinner(
                "Verifying image..."
            ):

                (
                    modality_probabilities,
                    modality_index,
                    modality_class,
                    modality_confidence
                ) = predict_modality(
                    image
                )

            # =================================================
            # GET CHEST X-RAY PROBABILITY
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
            # SHOW CHEST X-RAY PROBABILITY
            # =================================================

            st.metric(

                "Chest X-ray Probability",

                (
                    f"{chest_xray_probability * 100:.2f}%"
                )
            )

            # =================================================
            # CHEST X-RAY THRESHOLD
            # =================================================
            #
            # 40% threshold
            #
            # If probability >= 40%:
            #       Continue to pneumonia detection.
            #
            # Otherwise:
            #       Reject.
            #
            # =================================================

            if (
                chest_xray_probability
                >= CHEST_XRAY_THRESHOLD
            ):

                # =================================================
                # CHEST X-RAY ACCEPTED
                # =================================================

                st.success(
                    "✅ Chest X-ray image detected."
                )

                st.write(
                    "Chest X-ray confidence: "
                    f"{chest_xray_probability * 100:.2f}%"
                )

                # =================================================
                # STEP 2
                # =================================================

                st.subheader(
                    "Step 2 — Pneumonia Detection"
                )

                with st.spinner(
                    "Analyzing chest X-ray..."
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
                # PROBABILITIES
                # =================================================

                probability_col1, probability_col2 = (
                    st.columns(2)
                )

                with probability_col1:

                    st.metric(
                        "Normal Probability",
                        (
                            f"{normal_probability * 100:.2f}%"
                        )
                    )

                with probability_col2:

                    st.metric(
                        "Pneumonia Probability",
                        (
                            f"{pneumonia_probability * 100:.2f}%"
                        )
                    )

                # =================================================
                # DISCLAIMER
                # =================================================

                st.info(
                    "This AI-generated result is intended "
                    "for research and educational purposes "
                    "only and does not replace professional "
                    "medical diagnosis."
                )

            else:

                # =================================================
                # CHEST X-RAY REJECTED
                # =================================================

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

    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the image."
        )

        st.exception(e)
