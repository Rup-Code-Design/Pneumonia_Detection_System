import os
import io
from datetime import datetime

import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from fpdf import FPDF

from modality_model_builder import (
    build_modality_classifier
)

from model_builder import (
    build_model
)


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

MODALITY_MODEL_PATH = (
    "best_modality_classifier.weights.h5"
)

PNEUMONIA_MODEL_PATH = (
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# IMAGE SIZES
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
# MODALITY CLASS MAPPING
# ============================================================
#
# IMPORTANT:
#
# Your current modality training code has ONLY 3 classes:
#
# CHEST_XRAY = 0
# CT         = 1
# MRI        = 2
#
# Do NOT add OTHER here.
#
# ============================================================

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# CHEST X-RAY THRESHOLD
# ============================================================

CHEST_XRAY_THRESHOLD = 0.50


# ============================================================
# PNEUMONIA THRESHOLD
# ============================================================

PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# COLOR IMAGE DETECTION SETTINGS
# ============================================================
#
# A normal X-ray may technically be stored as RGB.
# Therefore, we should NOT simply check:
#
# image.mode == "RGB"
#
# because that would incorrectly reject RGB-encoded X-rays.
#
# Instead, we measure actual color/saturation.
#
# ============================================================

COLOR_SATURATION_THRESHOLD = 0.12

COLOR_PIXEL_RATIO_THRESHOLD = 0.20


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# COLOR IMAGE DETECTION FUNCTION
# ============================================================

def is_color_image(image):
    """
    Detect whether an image contains significant actual color.

    RGB-encoded grayscale X-rays are allowed.

    Returns:
        is_color : bool
        color_ratio : float
        mean_saturation : float
    """

    rgb_image = image.convert("RGB")

    rgb_array = np.asarray(
        rgb_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Convert RGB to HSV using TensorFlow
    # --------------------------------------------------------

    rgb_tensor = tf.convert_to_tensor(
        rgb_array,
        dtype=tf.float32
    )

    hsv_tensor = tf.image.rgb_to_hsv(
        rgb_tensor
    )

    saturation = hsv_tensor[:, :, 1].numpy()

    # --------------------------------------------------------
    # Calculate percentage of significantly colored pixels
    # --------------------------------------------------------

    colored_pixels = (
        saturation >
        COLOR_SATURATION_THRESHOLD
    )

    color_ratio = float(
        np.mean(colored_pixels)
    )

    mean_saturation = float(
        np.mean(saturation)
    )

    is_color = (
        color_ratio >
        COLOR_PIXEL_RATIO_THRESHOLD
    )

    return (
        is_color,
        color_ratio,
        mean_saturation
    )


# ============================================================
# LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Modality model weights not found:\n"
            f"{MODALITY_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Current model has 3 classes.
    # --------------------------------------------------------

    model = build_modality_classifier(

        input_shape=(
            128,
            128,
            3
        ),

        num_classes=3
    )

    model.load_weights(
        MODALITY_MODEL_PATH
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
            "Pneumonia model not found:\n"
            f"{PNEUMONIA_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # First try loading complete .keras model
    # --------------------------------------------------------

    try:

        loaded_model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

        return loaded_model

    except Exception:

        pass

    # --------------------------------------------------------
    # Fallback: build architecture and load weights
    # --------------------------------------------------------

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
### AI Medical Image Analysis

The system follows this pipeline:

**Uploaded Image → Color Check → Modality Classification**

- Chest X-ray → Pneumonia / Normal detection
- CT → Identified as CT and rejected for pneumonia analysis
- MRI → Identified as MRI and rejected for pneumonia analysis
- Color image → Rejected
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

    st.write(
        "Chest X-ray / CT / MRI"
    )

    st.write(
        "Pneumonia Model:"
    )

    st.write(
        "Xception-based classifier"
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
        "png",
        "bmp",
        "webp"
    ],

    help=(
        "Upload a medical image. "
        "Color images are rejected. "
        "Chest X-rays continue to pneumonia detection."
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
        ).convert("RGB")


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

        st.write(
            f"**Image Size:** "
            f"{image.width} × {image.height}"
        )

        st.write(
            f"**File:** "
            f"{uploaded_file.name}"
        )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            # =================================================
            # STEP 1 — COLOR IMAGE CHECK
            # =================================================

            st.subheader(
                "Step 1 — Image Quality Check"
            )

            (
                color_detected,
                color_ratio,
                mean_saturation
            ) = is_color_image(
                image
            )


            # =================================================
            # COLOR IMAGE REJECTION
            # =================================================

            if color_detected:

                st.error(
                    "Color image detected."
                )

                st.warning(
                    "Please upload a grayscale "
                    "Chest X-ray image."
                )

                st.write(
                    f"Detected colored pixel ratio: "
                    f"{color_ratio * 100:.2f}%"
                )

                st.write(
                    f"Mean saturation: "
                    f"{mean_saturation:.4f}"
                )


                # ---------------------------------------------
                # HISTORY
                # ---------------------------------------------

                history_entry = (
                    f"Rejected - Color Image - "
                    f"{uploaded_file.name}"
                )

                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                # ---------------------------------------------
                # PDF REPORT
                # ---------------------------------------------

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
                    "Pneumonia AI Image Report",
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

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
                    10,
                    "File Name:",
                    ln=False
                )

                pdf.set_font(
                    "Arial",
                    "",
                    12
                )

                safe_filename = (
                    uploaded_file.name
                    .encode(
                        "ascii",
                        "ignore"
                    )
                    .decode(
                        "ascii"
                    )
                )

                pdf.cell(
                    0,
                    10,
                    safe_filename,
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
                    10,
                    "Result:",
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
                    "Rejected - Color Image",
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
                    10,
                    "Recommendation:",
                    ln=False
                )

                pdf.set_font(
                    "Arial",
                    "",
                    12
                )

                pdf.multi_cell(
                    0,
                    10,
                    "Please upload a grayscale "
                    "chest X-ray image."
                )

                pdf.ln(10)

                pdf.set_font(
                    "Arial",
                    "I",
                    10
                )

                pdf.multi_cell(
                    0,
                    7,
                    "Disclaimer: This AI-generated "
                    "result is intended for research "
                    "purposes only and does not replace "
                    "professional medical diagnosis."
                )

                pdf_output = bytes(
                    pdf.output()
                )

                st.download_button(

                    label=(
                        "Download Report"
                    ),

                    data=pdf_output,

                    file_name=(
                        "Color_Image_Rejection_Report.pdf"
                    ),

                    mime="application/pdf"
                )

                st.stop()


            # =================================================
            # STEP 2 — MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 2 — Medical Image Modality"
            )


            # =================================================
            # PREPARE MODALITY IMAGE
            # =================================================

            modality_image = image.resize(
                MODALITY_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            modality_array = np.asarray(
                modality_image,
                dtype=np.float32
            )


            # =================================================
            # IMPORTANT PREPROCESSING
            # =================================================
            #
            # Your current training code uses:
            #
            # rescale=1.0 / 255.0
            #
            # Therefore the application MUST use
            # the same preprocessing.
            #
            # DO NOT use MobileNetV2 preprocess_input()
            # here for this trained model.
            #
            # =================================================

            modality_array = (
                modality_array / 255.0
            )


            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )


            # =================================================
            # MODALITY PREDICTION
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
            # OUTPUT VALIDATION
            # =================================================

            if (
                modality_prediction.ndim != 2
            ):

                st.error(
                    "Invalid modality model output."
                )

                st.stop()


            if (
                modality_prediction.shape[1]
                != 3
            ):

                st.error(
                    "The loaded modality model "
                    "must produce exactly 3 outputs."
                )

                st.write(
                    "Actual output shape:",
                    modality_prediction.shape
                )

                st.write(
                    "Expected:"
                )

                st.code(
                    "(None, 3)"
                )

                st.stop()


            # =================================================
            # GET PROBABILITIES
            # =================================================

            modality_probabilities = (
                modality_prediction[0]
                .astype(np.float64)
            )


            # =================================================
            # SAFETY CHECK
            # =================================================

            probability_sum = np.sum(
                modality_probabilities
            )


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
                    probability_sum,
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
            # MODALITY RESULTS
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
            # CT DETECTED
            # =================================================

            if predicted_modality == "CT":

                st.error(
                    "CT Scan detected."
                )

                st.warning(
                    "This system accepts Chest X-ray "
                    "images for pneumonia detection."
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )


                # ---------------------------------------------
                # HISTORY
                # ---------------------------------------------

                history_entry = (
                    f"CT Scan - "
                    f"{uploaded_file.name}"
                )

                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                # ---------------------------------------------
                # PDF
                # ---------------------------------------------

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
                    "Pneumonia AI Image Report",
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

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
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
                    safe_filename,
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
                    10,
                    "Detected Modality:",
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
                    "CT Scan",
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
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
                    f"{ct_probability * 100:.2f}%",
                    ln=True
                )

                pdf.ln(15)

                pdf.set_font(
                    "Arial",
                    "I",
                    10
                )

                pdf.multi_cell(
                    0,
                    7,
                    "Pneumonia detection was not "
                    "performed because the uploaded "
                    "image was classified as a CT scan."
                )

                pdf.ln(5)

                pdf.multi_cell(
                    0,
                    7,
                    "Disclaimer: This AI-generated "
                    "result is intended for research "
                    "purposes only and does not replace "
                    "professional medical diagnosis."
                )

                pdf_output = bytes(
                    pdf.output()
                )

                st.download_button(

                    label="Download CT Report",

                    data=pdf_output,

                    file_name=(
                        "CT_Scan_Report.pdf"
                    ),

                    mime="application/pdf"
                )

                st.stop()


            # =================================================
            # MRI DETECTED
            # =================================================

            if predicted_modality == "MRI":

                st.error(
                    "MRI image detected."
                )

                st.warning(
                    "This system accepts Chest X-ray "
                    "images for pneumonia detection."
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )


                # ---------------------------------------------
                # HISTORY
                # ---------------------------------------------

                history_entry = (
                    f"MRI - "
                    f"{uploaded_file.name}"
                )

                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                # ---------------------------------------------
                # PDF
                # ---------------------------------------------

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
                    "Pneumonia AI Image Report",
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

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
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
                    safe_filename,
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
                    10,
                    "Detected Modality:",
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
                    "MRI",
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    50,
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
                    f"{mri_probability * 100:.2f}%",
                    ln=True
                )

                pdf.ln(15)

                pdf.set_font(
                    "Arial",
                    "I",
                    10
                )

                pdf.multi_cell(
                    0,
                    7,
                    "Pneumonia detection was not "
                    "performed because the uploaded "
                    "image was classified as an MRI image."
                )

                pdf.ln(5)

                pdf.multi_cell(
                    0,
                    7,
                    "Disclaimer: This AI-generated "
                    "result is intended for research "
                    "purposes only and does not replace "
                    "professional medical diagnosis."
                )

                pdf_output = bytes(
                    pdf.output()
                )

                st.download_button(

                    label="Download MRI Report",

                    data=pdf_output,

                    file_name=(
                        "MRI_Report.pdf"
                    ),

                    mime="application/pdf"
                )

                st.stop()


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
            # X-RAY NOT CONFIRMED
            # =================================================

            if not is_chest_xray:

                st.error(
                    "Chest X-ray could not be confirmed."
                )

                st.warning(
                    "Please upload a clear Chest X-ray image."
                )

                st.write(
                    f"Chest X-ray confidence: "
                    f"{chest_xray_probability * 100:.2f}%"
                )

                st.write(
                    f"Predicted modality: "
                    f"**{predicted_modality}**"
                )


                history_entry = (
                    f"Rejected - "
                    f"{predicted_modality} - "
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
            # STEP 3 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 3 — Pneumonia Detection"
            )


            # =================================================
            # PREPARE PNEUMONIA IMAGE
            # =================================================

            pneumonia_image = image.resize(
                PNEUMONIA_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            pneumonia_array = np.asarray(
                pneumonia_image,
                dtype=np.float32
            )


            # =================================================
            # PNEUMONIA PREPROCESSING
            # =================================================
            #
            # Your previous pneumonia application code
            # uses /255.0.
            #
            # Keep this consistent with training.
            #
            # =================================================

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
                "Analyzing chest X-ray for pneumonia..."
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
            # PNEUMONIA OUTPUT HANDLING
            # =================================================

            normal_probability = None

            pneumonia_probability = None


            # -------------------------------------------------
            # SINGLE SIGMOID OUTPUT
            # -------------------------------------------------

            if prediction_values.size == 1:

                pneumonia_probability = float(
                    prediction_values
                )

                normal_probability = (
                    1.0
                    -
                    pneumonia_probability
                )


            # -------------------------------------------------
            # TWO-CLASS SOFTMAX OUTPUT
            # -------------------------------------------------

            elif prediction_values.size == 2:

                probabilities = (
                    prediction_values
                    .astype(np.float64)
                )


                # ---------------------------------------------
                # Already probabilities?
                # ---------------------------------------------

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
                        np.sum(probabilities),
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
                # IMPORTANT:
                #
                # Expected:
                #
                # 0 = Normal
                # 1 = Pneumonia
                #
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
                    "Output shape:",
                    prediction.shape
                )

                st.stop()


            # =================================================
            # CLAMP PROBABILITIES
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
            # FINAL DIAGNOSIS
            # =================================================

            if (
                pneumonia_probability
                >= PNEUMONIA_THRESHOLD
            ):

                diagnosis = "Pneumonia"

                diagnosis_confidence = (
                    pneumonia_probability
                )

                st.error(
                    "Diagnosis: Pneumonia"
                )

            else:

                diagnosis = "Normal"

                diagnosis_confidence = (
                    normal_probability
                )

                st.success(
                    "Diagnosis: Normal"
                )


            # =================================================
            # FINAL RESULT
            # =================================================

            st.subheader(
                "Final Result"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "X-ray Confidence",
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
                f"**Final Diagnosis:** {diagnosis}"
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

            st.write(
                f"**File:** "
                f"{uploaded_file.name}"
            )

            st.write(
                "**Image Modality:** "
                "Chest X-ray"
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
            # CREATE PDF
            # =================================================

            pdf = FPDF()

            pdf.add_page()


            # =================================================
            # PDF TITLE
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
            # PDF DATE
            # =================================================

            pdf.set_font(
                "Arial",
                "",
                10
            )

            pdf.cell(
                0,
                8,
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ln=True
            )

            pdf.ln(5)


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
                safe_filename,
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
                "Disclaimer: This AI-generated "
                "result is intended for research "
                "purposes only and does not replace "
                "professional medical diagnosis."
            )


            # =================================================
            # PDF OUTPUT
            # =================================================

            pdf_output = bytes(
                pdf.output()
            )


            st.download_button(

                label=(
                    "Download Diagnostic Report"
                ),

                data=pdf_output,

                file_name=(
                    "Pneumonia_Diagnostic_Report.pdf"
                ),

                mime="application/pdf"
            )


    except Exception as e:

        st.error(
            "An error occurred while "
            "processing the image."
        )

        st.exception(e)
