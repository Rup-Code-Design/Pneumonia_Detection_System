# ============================================================
# app.py
# 3-Class Medical Image Modality Verification
# + Pneumonia Detection
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Image Validation
#       ↓
# Colour Image Rejection
#       ↓
# 3-Class Modality Verifier
#       ↓
# ┌───────────────┬───────────────┬───────────────┐
# │     CT        │     MRI       │    X-RAY      │
# │    Reject     │    Reject     │    Accept     │
# └───────────────┴───────────────┴───────────────┘
#                                      ↓
#                             Pneumonia Model
#                                      ↓
#                              Normal / Pneumonia
#
# ============================================================

import os
import io

import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image

from xray_model_builder import build_xray_classifier
from pneumonia_model_builder import build_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
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
# 3-CLASS MODALITY MAPPING
# ============================================================
#
# IMPORTANT:
#
# This MUST match the class_indices from your 3-class
# X-ray verifier training.
#
# Current assumption:
#
# CT    = 0
# MRI   = 1
# X-RAY = 2
#
# ============================================================

CT_CLASS_INDEX = 0
MRI_CLASS_INDEX = 1
XRAY_CLASS_INDEX = 2


MODALITY_CLASS_MAP = {
    0: "CT",
    1: "MRI",
    2: "X-RAY"
}


# ============================================================
# PNEUMONIA CLASS MAPPING
# ============================================================
#
# This MUST match the class_indices from pneumonia training.
#
# Current assumption:
#
# Normal    = 0
# Pneumonia = 1
#
# ============================================================

NORMAL_CLASS_INDEX = 0
PNEUMONIA_CLASS_INDEX = 1


PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}


# ============================================================
# COLOUR IMAGE SETTINGS
# ============================================================
#
# The verifier should NOT be responsible for rejecting colour
# images.
#
# We explicitly reject colour images BEFORE modality
# classification.
#
# ============================================================

COLOR_TOLERANCE = 8.0


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = {
    "3-class modality verifier": XRAY_MODEL_PATH,
    "Pneumonia model": PNEUMONIA_MODEL_PATH
}


for model_name, model_path in required_files.items():

    if not os.path.isfile(model_path):

        st.error(
            f"❌ {model_name} was not found."
        )

        st.code(model_path)

        st.stop()


    if os.path.getsize(model_path) == 0:

        st.error(
            f"❌ {model_name} file is empty."
        )

        st.stop()


# ============================================================
# LOAD 3-CLASS MODALITY VERIFIER
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
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    # IMPORTANT:
    # Your pneumonia_model_builder.py has:
    #
    # build_model(input_shape=(224,224,3))
    #
    # It does NOT accept num_classes.

    model = build_model(
        input_shape=(224, 224, 3)
    )

    # If this is a .keras model containing the complete
    # architecture + weights, load it with load_model instead.
    #
    # This code assumes best_xception_pneumonia_model.keras
    # is a complete Keras model.

    model = tf.keras.models.load_model(
        PNEUMONIA_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model = load_xray_model()

except Exception as e:

    st.error(
        "❌ 3-class modality verifier loading failed."
    )

    st.exception(e)

    st.stop()


try:

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error(
        "❌ Pneumonia model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY MODALITY MODEL OUTPUT
# ============================================================

if xray_model.output_shape[-1] != 3:

    st.error(
        "❌ The modality verifier must have exactly "
        "3 output classes: CT, MRI and X-RAY."
    )

    st.write(
        f"Actual output shape: "
        f"{xray_model.output_shape}"
    )

    st.stop()


# ============================================================
# VERIFY PNEUMONIA MODEL OUTPUT
# ============================================================

if pneumonia_model.output_shape[-1] != 2:

    st.error(
        "❌ The pneumonia model must have exactly "
        "2 output classes: Normal and Pneumonia."
    )

    st.write(
        f"Actual output shape: "
        f"{pneumonia_model.output_shape}"
    )

    st.stop()


# ============================================================
# COLOUR / GRAYSCALE CHECK
# ============================================================

def check_grayscale_image(image):

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

        # ----------------------------------------------------
        # COLOUR IMAGE
        # ----------------------------------------------------

        if average_difference > COLOR_TOLERANCE:

            return (
                False,
                "COLOUR"
            )

        # ----------------------------------------------------
        # GRAYSCALE IMAGE
        # ----------------------------------------------------

        return (
            True,
            "GRAYSCALE"
        )

    except Exception as e:

        return (
            False,
            f"ERROR: {e}"
        )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(image):

    width, height = image.size

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    if width < 64 or height < 64:

        return (
            False,
            "❌ Image resolution is too small."
        )


    # --------------------------------------------------------
    # COLOUR CHECK
    # --------------------------------------------------------

    is_grayscale, image_type = (
        check_grayscale_image(image)
    )

    if image_type == "COLOUR":

        return (
            False,
            "❌ Colour image detected. "
            "Only grayscale medical images are accepted."
        )


    if image_type == "ERROR":

        return (
            False,
            "❌ Unable to determine image colour."
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
        "✅ Grayscale image passed basic validation."
    )


# ============================================================
# PREPROCESS MODALITY IMAGE
# ============================================================

def preprocess_for_modality(image):

    image = image.convert("RGB")

    image = image.resize(
        XRAY_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # This MUST match your 3-class verifier training.
    #
    # If training used rescale=1./255, this is correct.
    # --------------------------------------------------------

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREPROCESS PNEUMONIA IMAGE
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

    # Must match pneumonia training preprocessing.

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# CONVERT OUTPUT TO PROBABILITIES
# ============================================================

def convert_to_probabilities(raw_scores):

    raw_scores = np.asarray(
        raw_scores,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Already probabilities
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Logits
    # --------------------------------------------------------

    return tf.nn.softmax(
        raw_scores
    ).numpy()


# ============================================================
# 3-CLASS MODALITY PREDICTION
# ============================================================

def predict_modality(image):

    image_array = (
        preprocess_for_modality(
            image
        )
    )


    predictions = xray_model.predict(
        image_array,
        verbose=0
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
        predictions.shape[1] != 3
    ):

        raise ValueError(
            "3-class modality verifier must produce "
            "exactly 3 outputs."
        )


    raw_scores = (
        predictions[0]
        .astype(np.float64)
    )


    probabilities = (
        convert_to_probabilities(
            raw_scores
        )
    )


    # --------------------------------------------------------
    # INDIVIDUAL PROBABILITIES
    # --------------------------------------------------------

    ct_probability = float(
        probabilities[CT_CLASS_INDEX]
    )

    mri_probability = float(
        probabilities[MRI_CLASS_INDEX]
    )

    xray_probability = float(
        probabilities[XRAY_CLASS_INDEX]
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
        MODALITY_CLASS_MAP[
            predicted_index
        ]
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # X-RAY DECISION
    # --------------------------------------------------------
    #
    # X-ray is accepted ONLY when X-RAY is the model's
    # highest-probability class.
    #
    # Therefore:
    #
    # CT  → reject
    # MRI → reject
    # X-RAY → continue
    #
    # --------------------------------------------------------

    is_xray = (
        predicted_index
        ==
        XRAY_CLASS_INDEX
    )


    return {

        "predicted_index":
            predicted_index,

        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "ct_probability":
            ct_probability,

        "mri_probability":
            mri_probability,

        "xray_probability":
            xray_probability,

        "probabilities":
            probabilities,

        "raw_scores":
            raw_scores,

        "is_xray":
            is_xray
    }


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(image):

    image_array = (
        preprocess_for_pneumonia(
            image
        )
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


    # --------------------------------------------------------
    # OUTPUT VALIDATION
    # --------------------------------------------------------

    if (
        prediction.ndim != 2
        or
        prediction.shape[1] != 2
    ):

        raise ValueError(
            "Pneumonia model must produce exactly "
            "2 output values."
        )


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
    "3-class medical image verification followed by "
    "pneumonia classification."
)


st.info(
    "Only grayscale chest X-ray images are accepted. "
    "Colour images, CT scans and MRI scans are rejected "
    "before pneumonia detection."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "AI Pipeline"
    )


    st.write(
        """
        Image Upload
        ↓
        Colour Check
        ↓
        Basic Validation
        ↓
        CT / MRI / X-RAY Verification
        ↓
        X-RAY?
        ↓
        Pneumonia Detection
        ↓
        Normal / Pneumonia
        """
    )


    st.divider()


    st.write(
        "**Modality Classes**"
    )

    st.write(
        f"CT = Class {CT_CLASS_INDEX}"
    )

    st.write(
        f"MRI = Class {MRI_CLASS_INDEX}"
    )

    st.write(
        f"X-RAY = Class {XRAY_CLASS_INDEX}"
    )


    st.divider()


    st.write(
        "**Pneumonia Classes**"
    )

    st.write(
        f"Normal = Class {NORMAL_CLASS_INDEX}"
    )

    st.write(
        f"Pneumonia = Class {PNEUMONIA_CLASS_INDEX}"
    )


    st.divider()


    st.write(
        "**Rejected**"
    )

    st.write(
        "Colour images"
    )

    st.write(
        "CT scans"
    )

    st.write(
        "MRI scans"
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
        "**3-Class Modality Verifier**"
    )

    st.code(
        "best_xray_verifier.weights.h5"
    )


    st.write(
        "**Modality Input Size**"
    )

    st.write(
        "128 × 128 × 3"
    )


    st.write(
        "**Modality Mapping**"
    )

    st.code(
        f"""
Class {CT_CLASS_INDEX} = CT
Class {MRI_CLASS_INDEX} = MRI
Class {XRAY_CLASS_INDEX} = X-RAY
"""
    )


    st.divider()


    st.write(
        "**Pneumonia Model**"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )


    st.write(
        "**Pneumonia Input Size**"
    )

    st.write(
        "224 × 224 × 3"
    )


    st.write(
        "**Pneumonia Mapping**"
    )

    st.code(
        f"""
Class {NORMAL_CLASS_INDEX} = Normal
Class {PNEUMONIA_CLASS_INDEX} = Pneumonia
"""
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    help=(
        "Upload a grayscale chest X-ray image. "
        "Colour images, CT and MRI images will be rejected."
    )
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
        caption=uploaded_file.name,
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


    st.success(
        validation_message
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True
    ):

        # ====================================================
        # STEP 1 — MODALITY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Medical Image Verification"
        )


        with st.spinner(
            "Identifying CT, MRI or X-ray..."
        ):

            try:

                modality_result = (
                    predict_modality(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "❌ Modality verification failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # EXTRACT RESULTS
        # ====================================================

        predicted_class = (
            modality_result[
                "predicted_class"
            ]
        )

        predicted_index = (
            modality_result[
                "predicted_index"
            ]
        )

        modality_confidence = (
            modality_result[
                "confidence"
            ]
        )

        ct_probability = (
            modality_result[
                "ct_probability"
            ]
        )

        mri_probability = (
            modality_result[
                "mri_probability"
            ]
        )

        xray_probability = (
            modality_result[
                "xray_probability"
            ]
        )

        is_xray = (
            modality_result[
                "is_xray"
            ]
        )


        # ====================================================
        # MODALITY PROBABILITIES
        # ====================================================

        st.write(
            "**Modality probabilities**"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "CT",
                f"{ct_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "MRI",
                f"{mri_probability * 100:.2f}%"
            )


        with col3:

            st.metric(
                "X-RAY",
                f"{xray_probability * 100:.2f}%"
            )


        # ====================================================
        # PROBABILITY BARS
        # ====================================================

        st.write(
            "CT probability"
        )

        st.progress(
            float(
                np.clip(
                    ct_probability,
                    0.0,
                    1.0
                )
            )
        )


        st.write(
            "MRI probability"
        )

        st.progress(
            float(
                np.clip(
                    mri_probability,
                    0.0,
                    1.0
                )
            )
        )


        st.write(
            "X-ray probability"
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


        # ====================================================
        # MODALITY RESULT
        # ====================================================

        if predicted_class == "X-RAY":

            st.success(
                "X-RAY IMAGE DETECTED"
            )

            st.write(
                f"X-ray confidence: "
                f"**{modality_confidence * 100:.2f}%**"
            )

            st.info(
                "The image has been verified as a "
                "chest X-ray. Pneumonia detection will "
                "now be performed."
            )


        elif predicted_class == "CT":

            st.error(
                "CT SCAN DETECTED"
            )

            st.write(
                f"CT confidence: "
                f"**{modality_confidence * 100:.2f}%**"
            )

            st.warning(
                "This system accepts only chest X-ray "
                "images. Pneumonia detection has been stopped."
            )


            history_entry = (
                f"Rejected CT - "
                f"{uploaded_file.name}"
            )


            if history_entry not in st.session_state.history:

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        elif predicted_class == "MRI":

            st.error(
                "MRI SCAN DETECTED"
            )

            st.write(
                f"MRI confidence: "
                f"**{modality_confidence * 100:.2f}%**"
            )

            st.warning(
                "This system accepts only chest X-ray "
                "images. Pneumonia detection has been stopped."
            )


            history_entry = (
                f"Rejected MRI - "
                f"{uploaded_file.name}"
            )


            if history_entry not in st.session_state.history:

                st.session_state.history.append(
                    history_entry
                )


            st.stop()


        else:

            st.error(
                "❌ Unknown image modality detected."
            )

            st.stop()


        # ====================================================
        # STEP 2 — PNEUMONIA DETECTION
        # ====================================================

        st.subheader(
            "Step 2 — Pneumonia Detection"
        )


        with st.spinner(
            "Analyzing verified chest X-ray..."
        ):

            try:

                (
                    pneumonia_class,
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

        if pneumonia_class == "Pneumonia":

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
            f"**Diagnosis:** {pneumonia_class}"
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
            f"{pneumonia_class} - "
            f"{uploaded_file.name}"
        )


        if history_entry not in st.session_state.history:

            st.session_state.history.append(
                history_entry
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "Technical Details"
        ):

            st.write(
                "**Verified modality:** Chest X-ray"
            )

            st.write(
                f"CT probability: "
                f"{ct_probability * 100:.2f}%"
            )

            st.write(
                f"MRI probability: "
                f"{mri_probability * 100:.2f}%"
            )

            st.write(
                f"X-ray probability: "
                f"{xray_probability * 100:.2f}%"
            )

            st.write(
                f"Modality confidence: "
                f"{modality_confidence * 100:.2f}%"
            )

            st.divider()

            st.write(
                "**Pneumonia model:**"
            )

            st.write(
                "Xception-style blocks + "
                "Residual blocks + SE Attention + "
                "GELU"
            )

            st.write(
                "Input: 224 × 224 × 3"
            )

            st.write(
                "Output: Normal / Pneumonia"
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
