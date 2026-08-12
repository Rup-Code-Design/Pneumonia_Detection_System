# ============================================================
# PNEUMONIA DETECTION SYSTEM
# ============================================================
#
# PIPELINE:
#
# Uploaded Image
#       |
#       v
# CT_Verifier.keras
#       |
#       +---- CT  ----------> Reject
#       |
#       +---- MRI ---------> Reject
#       |
#       +---- Chest X-ray -> Pneumonia Model
#                              |
#                              +--> Normal
#                              |
#                              +--> Pneumonia
#
# ============================================================

import os
import io
from datetime import datetime

import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from fpdf import FPDF


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia AI",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

# IMPORTANT:
# This is the complete 3-class modality model that you
# trained and uploaded.
#
# Classes:
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI

MODALITY_MODEL_PATH = "CT_Verifier.keras"


# Complete pneumonia model
PNEUMONIA_MODEL_PATH = "best_xception_pneumonia_model.keras"


# ============================================================
# IMAGE SIZES
# ============================================================

# Your CT_Verifier.keras was trained at 128 x 128

MODALITY_IMAGE_SIZE = (
    128,
    128
)


# Your pneumonia model uses 224 x 224

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# MODALITY CLASS NAMES
# ============================================================

# MUST MATCH YOUR TRAINING CODE

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# CHEST X-RAY THRESHOLD
# ============================================================

# Important:
#
# The model must predict CHEST_XRAY as the highest class.
#
# The probability must also exceed this threshold.
#
# Do NOT make this extremely low.
#
# 0.35 means:
#
# Chest X-ray probability >= 35%
# AND
# Chest X-ray must be the highest probability.
#
# If your trained model performs poorly, changing this
# threshold alone will NOT fix the model.

CHEST_XRAY_THRESHOLD = 0.35


# ============================================================
# PNEUMONIA THRESHOLD
# ============================================================

PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_MODEL_PATH
    ):

        raise FileNotFoundError(
            "CT_Verifier.keras was not found.\n\n"
            "Place CT_Verifier.keras in the same "
            "folder as app.py."
        )

    # --------------------------------------------------------
    # Load complete Keras model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODALITY_MODEL_PATH,
        compile=False
    )

    # --------------------------------------------------------
    # Verify input shape
    # --------------------------------------------------------

    expected_input = (
        128,
        128,
        3
    )

    if model.input_shape[1:] != expected_input:

        raise ValueError(
            "Unexpected modality model input shape.\n"
            f"Expected: {expected_input}\n"
            f"Found: {model.input_shape[1:]}"
        )

    # --------------------------------------------------------
    # Verify output shape
    # --------------------------------------------------------

    if model.output_shape[-1] != 3:

        raise ValueError(
            "CT_Verifier.keras must have exactly "
            "3 output classes.\n"
            f"Found output shape: {model.output_shape}"
        )

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(
        PNEUMONIA_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Pneumonia model was not found.\n\n"
            f"Required file:\n"
            f"{PNEUMONIA_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load complete .keras model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        PNEUMONIA_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD BOTH MODELS
# ============================================================

try:

    modality_model = load_modality_model()

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
This system performs medical image modality verification
before pneumonia detection.

### Processing pipeline

**Chest X-ray → Pneumonia Detection**

**CT → Rejected**

**MRI → Rejected**

**Colour / unsupported image → Rejected**
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
        "Modality Model:"
    )

    st.code(
        "CT_Verifier.keras"
    )

    st.write(
        "Modality Classes:"
    )

    st.write(
        "0 → Chest X-ray"
    )

    st.write(
        "1 → CT"
    )

    st.write(
        "2 → MRI"
    )

    st.divider()

    st.write(
        "Pneumonia Model:"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )

    st.divider()

    st.write(
        f"Chest X-ray threshold: "
        f"{CHEST_XRAY_THRESHOLD:.2f}"
    )

    st.write(
        f"Pneumonia threshold: "
        f"{PNEUMONIA_THRESHOLD:.2f}"
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

    "Upload a medical image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],

    help=(
        "Upload a Chest X-ray image. "
        "CT and MRI images will be identified "
        "and rejected."
    )
)


# ============================================================
# MAIN PROCESSING
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
            io.BytesIO(file_bytes)
        )

        # ----------------------------------------------------
        # Original image information
        # ----------------------------------------------------

        original_mode = image.mode

        original_width, original_height = (
            image.size
        )

        # ----------------------------------------------------
        # Convert only for model processing
        # ----------------------------------------------------

        rgb_image = image.convert(
            "RGB"
        )

        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            rgb_image,
            caption=(
                f"{uploaded_file.name} | "
                f"{original_width} × {original_height}"
            ),
            use_container_width=True
        )


        # ====================================================
        # BASIC IMAGE VALIDATION
        # ====================================================

        # Reject completely invalid images

        if (
            original_width < 32
            or
            original_height < 32
        ):

            st.error(
                "Image resolution is too small."
            )

            st.warning(
                "Please upload a proper medical image."
            )

            st.stop()


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            # =================================================
            # STEP 0 — COLOUR IMAGE CHECK
            # =================================================

            st.subheader(
                "Step 0 — Image Quality Verification"
            )

            # -------------------------------------------------
            # Determine whether image contains meaningful
            # colour information.
            #
            # This is NOT the final modality classifier.
            # It is an additional safeguard against ordinary
            # colour photographs/images.
            # -------------------------------------------------

            rgb_array = np.asarray(
                rgb_image,
                dtype=np.float32
            )

            red = rgb_array[:, :, 0]
            green = rgb_array[:, :, 1]
            blue = rgb_array[:, :, 2]

            channel_difference = (
                np.maximum.reduce(
                    [
                        np.abs(red - green),
                        np.abs(red - blue),
                        np.abs(green - blue)
                    ]
                )
            )

            colour_fraction = float(
                np.mean(
                    channel_difference > 12.0
                )
            )

            mean_channel_difference = float(
                np.mean(
                    channel_difference
                )
            )

            # -------------------------------------------------
            # Colour image rejection
            #
            # A grayscale X-ray converted to RGB will have
            # nearly identical RGB channels.
            #
            # Normal colour photographs generally have much
            # larger channel differences.
            # -------------------------------------------------

            IS_COLOUR_IMAGE = (
                colour_fraction > 0.10
                and
                mean_channel_difference > 8.0
            )

            if IS_COLOUR_IMAGE:

                st.error(
                    "Colour image detected."
                )

                st.warning(
                    "Please upload a Chest X-ray image. "
                    "Colour photographs and other colour "
                    "images are not accepted."
                )

                st.write(
                    f"Colour-pixel fraction: "
                    f"{colour_fraction * 100:.2f}%"
                )

                st.write(
                    f"Mean RGB channel difference: "
                    f"{mean_channel_difference:.2f}"
                )

                history_entry = (
                    f"Rejected - Colour image - "
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


            st.success(
                "Image passed the colour-image check."
            )


            # =================================================
            # STEP 1 — MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 1 — Medical Image Modality"
            )

            # -------------------------------------------------
            # Resize to 128 x 128
            # -------------------------------------------------

            modality_image = rgb_image.resize(
                MODALITY_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            modality_array = np.asarray(
                modality_image,
                dtype=np.float32
            )


            # -------------------------------------------------
            # IMPORTANT
            #
            # Your training code used:
            #
            # rescale=1.0 / 255.0
            #
            # Therefore DO NOT use MobileNetV2
            # preprocess_input here.
            # -------------------------------------------------

            modality_array = (
                modality_array / 255.0
            )


            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )


            # =================================================
            # PREDICT MODALITY
            # =================================================

            with st.spinner(
                "Identifying image modality..."
            ):

                modality_prediction = (
                    modality_model.predict(
                        modality_input,
                        verbose=0
                    )
                )


            modality_prediction = np.asarray(
                modality_prediction
            )


            # =================================================
            # CHECK OUTPUT
            # =================================================

            if (
                modality_prediction.ndim != 2
                or
                modality_prediction.shape[1] != 3
            ):

                st.error(
                    "Invalid modality model output."
                )

                st.write(
                    "Expected output: (1, 3)"
                )

                st.write(
                    "Actual output:",
                    modality_prediction.shape
                )

                st.stop()


            # =================================================
            # GET PROBABILITIES
            # =================================================

            modality_probabilities = (
                modality_prediction[0]
                .astype(np.float64)
            )


            # -------------------------------------------------
            # Safety normalization
            # -------------------------------------------------

            if (
                np.any(
                    modality_probabilities < 0
                )
                or
                np.any(
                    modality_probabilities > 1
                )
                or
                not np.isclose(
                    np.sum(
                        modality_probabilities
                    ),
                    1.0,
                    atol=1e-3
                )
            ):

                modality_probabilities = (
                    tf.nn.softmax(
                        modality_probabilities
                    ).numpy()
                )


            # =================================================
            # EXTRACT PROBABILITIES
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


            # =================================================
            # PREDICTED MODALITY
            # =================================================

            predicted_index = int(
                np.argmax(
                    modality_probabilities
                )
            )

            predicted_modality = (
                MODALITY_CLASS_NAMES[
                    predicted_index
                ]
            )

            predicted_confidence = float(
                modality_probabilities[
                    predicted_index
                ]
            )


            # =================================================
            # DISPLAY MODALITY PROBABILITIES
            # =================================================

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Chest X-ray",
                    f"{chest_xray_probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "CT",
                    f"{ct_probability * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "MRI",
                    f"{mri_probability * 100:.2f}%"
                )


            # =================================================
            # CHEST X-RAY DECISION
            # =================================================

            is_chest_xray = (

                predicted_index == 0

                and

                chest_xray_probability
                >= CHEST_XRAY_THRESHOLD
            )


            # =================================================
            # CT DETECTED
            # =================================================

            if (
                predicted_modality
                == "CT"
            ):

                st.error(
                    "CT Scan detected."
                )

                st.warning(
                    "This application accepts only "
                    "Chest X-ray images for pneumonia "
                    "detection."
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )


                history_entry = (
                    f"Rejected - CT - "
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


            # =================================================
            # MRI DETECTED
            # =================================================

            if (
                predicted_modality
                == "MRI"
            ):

                st.error(
                    "MRI image detected."
                )

                st.warning(
                    "This application accepts only "
                    "Chest X-ray images for pneumonia "
                    "detection."
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )


                history_entry = (
                    f"Rejected - MRI - "
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


            # =================================================
            # CHEST X-RAY NOT CONFIDENT ENOUGH
            # =================================================

            if not is_chest_xray:

                st.error(
                    "The image could not be confidently "
                    "identified as a Chest X-ray."
                )

                st.warning(
                    "Please upload a clear Chest X-ray image."
                )

                st.write(
                    f"Predicted modality: "
                    f"**{predicted_modality}**"
                )

                st.write(
                    f"Predicted confidence: "
                    f"{predicted_confidence * 100:.2f}%"
                )

                history_entry = (
                    f"Rejected - Low X-ray confidence - "
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


            # =================================================
            # CHEST X-RAY CONFIRMED
            # =================================================

            st.success(
                "Chest X-ray detected."
            )

            st.write(
                f"Chest X-ray confidence: "
                f"{chest_xray_probability * 100:.2f}%"
            )


            # =================================================
            # STEP 2 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )


            # =================================================
            # PREPARE PNEUMONIA IMAGE
            # =================================================

            pneumonia_image = rgb_image.resize(
                PNEUMONIA_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            pneumonia_array = np.asarray(
                pneumonia_image,
                dtype=np.float32
            )


            # -------------------------------------------------
            # Your previous pneumonia app used /255.0
            # -------------------------------------------------

            pneumonia_array = (
                pneumonia_array / 255.0
            )


            pneumonia_input = np.expand_dims(
                pneumonia_array,
                axis=0
            )


            # =================================================
            # PNEUMONIA PREDICTION
            # =================================================

            with st.spinner(
                "Analyzing Chest X-ray for pneumonia..."
            ):

                prediction = (
                    pneumonia_model.predict(
                        pneumonia_input,
                        verbose=0
                    )
                )


            prediction = np.asarray(
                prediction
            )


            prediction_values = np.squeeze(
                prediction
            )


            # =================================================
            # HANDLE PNEUMONIA MODEL OUTPUT
            # =================================================
            #
            # Supported:
            #
            # 1. Sigmoid:
            #    [Pneumonia probability]
            #
            # 2. Two-class softmax:
            #    [Normal, Pneumonia]
            #
            # =================================================

            normal_probability = None
            pneumonia_probability = None


            # -------------------------------------------------
            # Single sigmoid output
            # -------------------------------------------------

            if prediction_values.size == 1:

                pneumonia_probability = float(
                    prediction_values
                )

                pneumonia_probability = float(
                    np.clip(
                        pneumonia_probability,
                        0.0,
                        1.0
                    )
                )

                normal_probability = (
                    1.0
                    -
                    pneumonia_probability
                )


            # -------------------------------------------------
            # Two-class output
            # -------------------------------------------------

            elif prediction_values.size == 2:

                probabilities = (
                    prediction_values
                    .astype(np.float64)
                )


                # Already probabilities?

                if (

                    np.all(
                        probabilities >= 0
                    )

                    and

                    np.all(
                        probabilities <= 1
                    )

                    and

                    np.isclose(
                        np.sum(
                            probabilities
                        ),
                        1.0,
                        atol=1e-3
                    )

                ):

                    pneumonia_probabilities = (
                        probabilities
                    )

                else:

                    pneumonia_probabilities = (
                        tf.nn.softmax(
                            probabilities
                        ).numpy()
                    )


                # ------------------------------------------------
                # Expected mapping from your previous model:
                #
                # 0 = Normal
                # 1 = Pneumonia
                # ------------------------------------------------

                normal_probability = float(
                    pneumonia_probabilities[0]
                )

                pneumonia_probability = float(
                    pneumonia_probabilities[1]
                )


            else:

                st.error(
                    "Unsupported pneumonia model output."
                )

                st.write(
                    "Model output shape:",
                    prediction.shape
                )

                st.stop()


            # =================================================
            # FINAL SAFETY
            # =================================================

            normal_probability = float(
                np.clip(
                    normal_probability,
                    0.0,
                    1.0
                )
            )

            pneumonia_probability = float(
                np.clip(
                    pneumonia_probability,
                    0.0,
                    1.0
                )
            )


            # =================================================
            # DIAGNOSIS
            # =================================================

            if (
                pneumonia_probability
                >= PNEUMONIA_THRESHOLD
            ):

                diagnosis = (
                    "Pneumonia"
                )

                diagnosis_confidence = (
                    pneumonia_probability
                )

            else:

                diagnosis = (
                    "Normal"
                )

                diagnosis_confidence = (
                    normal_probability
                )


            # =================================================
            # DISPLAY DIAGNOSIS
            # =================================================

            st.subheader(
                "Final Result"
            )


            if diagnosis == "Pneumonia":

                st.error(
                    "Diagnosis: Pneumonia"
                )

            else:

                st.success(
                    "Diagnosis: Normal"
                )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Chest X-ray",
                    f"{chest_xray_probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Normal",
                    f"{normal_probability * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "Pneumonia",
                    f"{pneumonia_probability * 100:.2f}%"
                )


            st.write(
                f"**Final Diagnosis:** "
                f"{diagnosis}"
            )

            st.write(
                f"**Diagnosis Confidence:** "
                f"{diagnosis_confidence * 100:.2f}%"
            )


            # =================================================
            # HISTORY
            # =================================================

            history_entry = (
                f"{diagnosis} - "
                f"{uploaded_file.name}"
            )

            if (
                history_entry
                not in st.session_state.history
            ):

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

            report_date = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            st.write(
                f"**File:** "
                f"{uploaded_file.name}"
            )

            st.write(
                f"**Date:** "
                f"{report_date}"
            )

            st.write(
                "**Image Modality:** Chest X-ray"
            )

            st.write(
                f"**X-ray Confidence:** "
                f"{chest_xray_probability * 100:.2f}%"
            )

            st.write(
                f"**Normal Probability:** "
                f"{normal_probability * 100:.2f}%"
            )

            st.write(
                f"**Pneumonia Probability:** "
                f"{pneumonia_probability * 100:.2f}%"
            )

            st.write(
                f"**Diagnosis:** "
                f"{diagnosis}"
            )

            st.write(
                f"**Diagnosis Confidence:** "
                f"{diagnosis_confidence * 100:.2f}%"
            )


            # =================================================
            # CREATE PDF REPORT
            # =================================================

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

            if not clean_filename:

                clean_filename = "uploaded_image"


            # -------------------------------------------------
            # Remove extension for report filename
            # -------------------------------------------------

            report_base_name = os.path.splitext(
                clean_filename
            )[0]


            pdf = FPDF()

            pdf.add_page()


            # =================================================
            # TITLE
            # =================================================

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


            pdf.ln(10)


            # =================================================
            # REPORT DATE
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Report Date:",
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
                report_date,
                ln=True
            )


            # =================================================
            # FILE NAME
            # =================================================

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


            # =================================================
            # MODALITY
            # =================================================

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


            # =================================================
            # X-RAY CONFIDENCE
            # =================================================

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
                f"{chest_xray_probability * 100:.2f}%",
                ln=True
            )


            # =================================================
            # NORMAL PROBABILITY
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Normal Probability:",
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
                f"{normal_probability * 100:.2f}%",
                ln=True
            )


            # =================================================
            # PNEUMONIA PROBABILITY
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Pneumonia Probability:",
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
                f"{pneumonia_probability * 100:.2f}%",
                ln=True
            )


            # =================================================
            # DIAGNOSIS
            # =================================================

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


            # =================================================
            # DIAGNOSIS CONFIDENCE
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
                10,
                "Diagnosis Confidence:",
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


            # =================================================
            # DISCLAIMER
            # =================================================

            pdf.ln(15)

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


            # =================================================
            # PDF OUTPUT
            # =================================================

            pdf_output = pdf.output()


            # =================================================
            # DOWNLOAD BUTTON
            # =================================================

            st.download_button(

                label=(
                    "Download Diagnostic Report"
                ),

                data=bytes(
                    pdf_output
                ),

                file_name=(
                    f"Report_"
                    f"{report_base_name}.pdf"
                ),

                mime="application/pdf",

                use_container_width=True
            )


    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the uploaded image."
        )

        st.exception(e)
