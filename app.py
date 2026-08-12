# ============================================================
# app.py
# ============================================================
#
# MEDICAL IMAGE MODALITY + PNEUMONIA DETECTION SYSTEM
#
# PIPELINE:
#
# 1. Reject obvious colour images
#
# 2. CT_Verifier.keras:
#       0 = CHEST_XRAY
#       1 = CT
#       2 = MRI
#
# 3. If CHEST_XRAY:
#       Run pneumonia model
#       -> Normal / Pneumonia
#
# 4. If CT:
#       Show CT Scan detected
#
# 5. If MRI:
#       Show MRI detected
#
# 6. Generate downloadable PDF report
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
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia AI",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# 2. MODEL PATHS
# ============================================================

MODALITY_MODEL_PATH = "CT_Verifier.keras"

PNEUMONIA_MODEL_PATH = (
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# 3. IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# 4. MODALITY CLASS MAPPING
# ============================================================
#
# THIS MUST MATCH YOUR TRAINING CODE:
#
# CHEST_XRAY = 0
# CT         = 1
# MRI        = 2
#
# ============================================================

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# 5. THRESHOLDS
# ============================================================

# The predicted class must be CHEST_XRAY
# AND its probability must reach this threshold.

CHEST_XRAY_THRESHOLD = 0.50


# Pneumonia decision threshold.
#
# 0 = Normal
# 1 = Pneumonia
#
PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# 6. SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# 7. LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_MODEL_PATH
    ):

        raise FileNotFoundError(
            "CT_Verifier.keras was not found.\n\n"
            f"Expected location:\n"
            f"{os.path.abspath(MODALITY_MODEL_PATH)}"
        )

    model = tf.keras.models.load_model(
        MODALITY_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# 8. LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(
        PNEUMONIA_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Pneumonia model was not found.\n\n"
            f"Expected location:\n"
            f"{os.path.abspath(PNEUMONIA_MODEL_PATH)}"
        )

    model = tf.keras.models.load_model(
        PNEUMONIA_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# 9. LOAD BOTH MODELS
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
# 10. VERIFY MODALITY MODEL
# ============================================================

try:

    modality_input_shape = (
        modality_model.input_shape
    )

    modality_output_shape = (
        modality_model.output_shape
    )

except Exception as e:

    st.error(
        "Could not read modality model "
        "input/output shapes."
    )

    st.exception(e)

    st.stop()


# ------------------------------------------------------------
# Check input shape
# ------------------------------------------------------------

if modality_input_shape != (
    None,
    128,
    128,
    3
):

    st.error(
        "Unexpected CT_Verifier input shape."
    )

    st.write(
        "Expected:",
        "(None, 128, 128, 3)"
    )

    st.write(
        "Actual:",
        modality_input_shape
    )

    st.stop()


# ------------------------------------------------------------
# Check output shape
# ------------------------------------------------------------

if modality_output_shape != (
    None,
    3
):

    st.error(
        "Unexpected CT_Verifier output shape."
    )

    st.write(
        "Expected:",
        "(None, 3)"
    )

    st.write(
        "Actual:",
        modality_output_shape
    )

    st.stop()


# ============================================================
# 11. COLOUR IMAGE DETECTOR
# ============================================================

def is_obvious_colour_image(
    image,
    saturation_threshold=18.0,
    colour_pixel_ratio=0.08
):

    """
    Detect strongly coloured images.

    Important:
    An RGB file is NOT automatically a colour image.

    Many chest X-rays are stored as RGB files but are
    visually grayscale.

    Therefore this function examines actual differences
    between R, G and B channels.
    """

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )

    r = rgb[:, :, 0]

    g = rgb[:, :, 1]

    b = rgb[:, :, 2]

    max_channel = np.maximum(
        np.maximum(r, g),
        b
    )

    min_channel = np.minimum(
        np.minimum(r, g),
        b
    )

    channel_difference = (
        max_channel - min_channel
    )

    mean_difference = float(
        np.mean(
            channel_difference
        )
    )

    coloured_pixels = (
        channel_difference
        > saturation_threshold
    )

    coloured_ratio = float(
        np.mean(
            coloured_pixels
        )
    )

    is_colour = (

        mean_difference
        > saturation_threshold

        and

        coloured_ratio
        > colour_pixel_ratio
    )

    return (
        is_colour,
        mean_difference,
        coloured_ratio
    )


# ============================================================
# 12. CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    filename,
    modality,
    modality_confidence,
    diagnosis=None,
    diagnosis_confidence=None,
    xray_confidence=None
):

    pdf = FPDF()

    pdf.add_page()

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        18
    )

    pdf.cell(
        0,
        15,
        "Medical Image AI Report",
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

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        11
    )

    pdf.cell(
        55,
        8,
        "Date:",
        ln=False
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.cell(
        0,
        8,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        ln=True
    )

    # --------------------------------------------------------
    # FILE NAME
    # --------------------------------------------------------

    clean_filename = (
        filename
        .encode(
            "ascii",
            "ignore"
        )
        .decode(
            "ascii"
        )
    )

    pdf.set_font(
        "Arial",
        "B",
        11
    )

    pdf.cell(
        55,
        8,
        "File Name:",
        ln=False
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.cell(
        0,
        8,
        clean_filename,
        ln=True
    )

    # --------------------------------------------------------
    # MODALITY
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        11
    )

    pdf.cell(
        55,
        8,
        "Image Modality:",
        ln=False
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.cell(
        0,
        8,
        modality,
        ln=True
    )

    # --------------------------------------------------------
    # MODALITY CONFIDENCE
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        11
    )

    pdf.cell(
        55,
        8,
        "Modality Confidence:",
        ln=False
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.cell(
        0,
        8,
        f"{modality_confidence * 100:.2f}%",
        ln=True
    )

    # --------------------------------------------------------
    # X-RAY CONFIDENCE
    # --------------------------------------------------------

    if xray_confidence is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            55,
            8,
            "X-ray Confidence:",
            ln=False
        )

        pdf.set_font(
            "Arial",
            "",
            11
        )

        pdf.cell(
            0,
            8,
            f"{xray_confidence * 100:.2f}%",
            ln=True
        )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    if diagnosis is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            55,
            8,
            "Diagnosis:",
            ln=False
        )

        pdf.set_font(
            "Arial",
            "",
            11
        )

        pdf.cell(
            0,
            8,
            diagnosis,
            ln=True
        )

    # --------------------------------------------------------
    # DIAGNOSIS CONFIDENCE
    # --------------------------------------------------------

    if diagnosis_confidence is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            55,
            8,
            "Diagnosis Confidence:",
            ln=False
        )

        pdf.set_font(
            "Arial",
            "",
            11
        )

        pdf.cell(
            0,
            8,
            f"{diagnosis_confidence * 100:.2f}%",
            ln=True
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    pdf.ln(15)

    pdf.set_font(
        "Arial",
        "I",
        9
    )

    pdf.multi_cell(
        0,
        6,
        "Disclaimer: This AI-generated result is intended "
        "for research purposes only and does not replace "
        "professional medical diagnosis."
    )

    return bytes(
        pdf.output()
    )


# ============================================================
# 13. HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
### Medical Image Analysis Pipeline

**Step 1:** Reject obvious colour images.

**Step 2:** Identify the medical image modality.

- Chest X-ray
- CT Scan
- MRI

**Step 3:** If the image is a Chest X-ray, detect:

- Normal
- Pneumonia

**Step 4:** Generate a downloadable PDF report.
"""
)


# ============================================================
# 14. SIDEBAR
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
        "Modality Input:"
    )

    st.code(
        "(128, 128, 3)"
    )

    st.write(
        "Modality Classes:"
    )

    st.code(
        "0 = CHEST_XRAY\n"
        "1 = CT\n"
        "2 = MRI"
    )

    st.write(
        "Pneumonia Model:"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )

    st.divider()

    st.write(
        "Chest X-ray threshold:"
    )

    st.write(
        f"{CHEST_XRAY_THRESHOLD * 100:.0f}%"
    )

    st.write(
        "Pneumonia threshold:"
    )

    st.write(
        f"{PNEUMONIA_THRESHOLD * 100:.0f}%"
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
# 15. FILE UPLOADER
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
        "Upload a Chest X-ray, CT scan, "
        "or MRI image."
    )
)


# ============================================================
# 16. PROCESS IMAGE
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

        image_rgb = image.convert(
            "RGB"
        )

        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        st.image(
            image_rgb,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ====================================================
        # IMAGE INFORMATION
        # ====================================================

        st.write(
            f"**Image size:** "
            f"{image.width} × {image.height}"
        )

        st.write(
            f"**Original image mode:** "
            f"{image.mode}"
        )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            # =================================================
            # STEP 0 — COLOUR IMAGE CHECK
            # =================================================

            st.subheader(
                "Step 0 — Image Quality Check"
            )

            (
                colour_detected,
                mean_colour_difference,
                coloured_ratio
            ) = is_obvious_colour_image(
                image_rgb
            )

            st.write(
                f"Mean RGB channel difference: "
                f"{mean_colour_difference:.2f}"
            )

            st.write(
                f"Coloured pixel ratio: "
                f"{coloured_ratio * 100:.2f}%"
            )

            # -------------------------------------------------
            # REJECT COLOUR IMAGE
            # -------------------------------------------------

            if colour_detected:

                st.error(
                    "Colour image detected."
                )

                st.warning(
                    "Please upload a grayscale "
                    "Chest X-ray image."
                )

                history_entry = (
                    "Rejected - Colour image - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(

                    filename=uploaded_file.name,

                    modality=(
                        "Rejected - Colour Image"
                    ),

                    modality_confidence=0.0
                )

                st.download_button(

                    label=(
                        "Download Rejection Report"
                    ),

                    data=pdf_data,

                    file_name=(
                        "Colour_Image_Rejection_Report.pdf"
                    ),

                    mime="application/pdf"
                )

                st.stop()

            else:

                st.success(
                    "Image appears grayscale."
                )

            # =================================================
            # STEP 1 — MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 1 — Medical Image Modality"
            )

            # =================================================
            # RESIZE TO 128 × 128
            # =================================================

            modality_image = image_rgb.resize(

                MODALITY_IMAGE_SIZE,

                Image.Resampling.LANCZOS
            )

            modality_array = np.asarray(

                modality_image,

                dtype=np.float32
            )

            # =================================================
            # IMPORTANT
            #
            # TRAINING:
            #
            # ImageDataGenerator(
            #     rescale=1.0 / 255.0
            # )
            #
            # Therefore:
            #
            # /255.0
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
                modality_prediction,
                dtype=np.float64
            )

            # =================================================
            # VALIDATE OUTPUT
            # =================================================

            if (
                modality_prediction.ndim
                != 2
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
                    "CT_Verifier.keras must "
                    "produce exactly 3 outputs."
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
            )

            # -------------------------------------------------
            # SAFETY CHECK
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

            xray_probability = float(
                modality_probabilities[0]
            )

            ct_probability = float(
                modality_probabilities[1]
            )

            mri_probability = float(
                modality_probabilities[2]
            )

            # =================================================
            # PREDICTED CLASS
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
                    f"{xray_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "CT Scan",
                    f"{ct_probability * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "MRI",
                    f"{mri_probability * 100:.2f}%"
                )

            # =================================================
            # DISPLAY PREDICTED MODALITY
            # =================================================

            st.write(
                f"**Predicted Modality:** "
                f"{predicted_modality}"
            )

            st.write(
                f"**Confidence:** "
                f"{predicted_confidence * 100:.2f}%"
            )

            # =================================================
            # DECISION
            # =================================================

            is_chest_xray = (

                predicted_index == 0

                and

                xray_probability
                >= CHEST_XRAY_THRESHOLD
            )

            # =================================================
            # CT DETECTED
            # =================================================

            if predicted_index == 1:

                st.error(
                    "CT Scan detected."
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )

                st.info(
                    "CT images are not passed to "
                    "the pneumonia detection model."
                )

                history_entry = (
                    "CT Scan - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(

                    filename=uploaded_file.name,

                    modality="CT Scan",

                    modality_confidence=(
                        ct_probability
                    )
                )

                st.download_button(

                    label=(
                        "Download CT Report (PDF)"
                    ),

                    data=pdf_data,

                    file_name=(
                        "CT_Scan_Report.pdf"
                    ),

                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # MRI DETECTED
            # =================================================

            if predicted_index == 2:

                st.error(
                    "MRI image detected."
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )

                st.info(
                    "MRI images are not passed to "
                    "the pneumonia detection model."
                )

                history_entry = (
                    "MRI - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(

                    filename=uploaded_file.name,

                    modality="MRI",

                    modality_confidence=(
                        mri_probability
                    )
                )

                st.download_button(

                    label=(
                        "Download MRI Report (PDF)"
                    ),

                    data=pdf_data,

                    file_name=(
                        "MRI_Report.pdf"
                    ),

                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # LOW CONFIDENCE X-RAY
            # =================================================

            if not is_chest_xray:

                st.error(
                    "Chest X-ray could not be confirmed."
                )

                st.warning(
                    "Please upload a clear Chest "
                    "X-ray image."
                )

                st.write(
                    f"Chest X-ray probability: "
                    f"{xray_probability * 100:.2f}%"
                )

                history_entry = (
                    "Rejected - Low X-ray confidence - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(

                    filename=uploaded_file.name,

                    modality=(
                        "Unconfirmed Medical Image"
                    ),

                    modality_confidence=(
                        predicted_confidence
                    )
                )

                st.download_button(

                    label=(
                        "Download Rejection Report (PDF)"
                    ),

                    data=pdf_data,

                    file_name=(
                        "Modality_Rejection_Report.pdf"
                    ),

                    mime="application/pdf"
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
                f"{xray_probability * 100:.2f}%"
            )

            # =================================================
            # STEP 2 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            # =================================================
            # PREPARE PNEUMONIA INPUT
            # =================================================

            pneumonia_image = image_rgb.resize(

                PNEUMONIA_IMAGE_SIZE,

                Image.Resampling.LANCZOS
            )

            pneumonia_array = np.asarray(

                pneumonia_image,

                dtype=np.float32
            )

            # =================================================
            # PNEUMONIA PREPROCESSING
            #
            # Keep /255.0 because your previous pneumonia
            # training pipeline used this preprocessing.
            # =================================================

            pneumonia_array = (
                pneumonia_array / 255.0
            )

            pneumonia_input = np.expand_dims(

                pneumonia_array,

                axis=0
            )

            # =================================================
            # PREDICT PNEUMONIA
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

            prediction_values = np.squeeze(
                prediction
            )

            # =================================================
            # INITIALIZE
            # =================================================

            normal_probability = None

            pneumonia_probability = None

            # =================================================
            # SINGLE OUTPUT
            # =================================================

            if prediction_values.size == 1:

                pneumonia_probability = float(
                    prediction_values
                )

                normal_probability = (
                    1.0
                    -
                    pneumonia_probability
                )

            # =================================================
            # TWO OUTPUTS
            # =================================================

            elif prediction_values.size == 2:

                probabilities = (
                    prediction_values
                    .astype(np.float64)
                )

                # ---------------------------------------------
                # Check if already probabilities
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
                        np.sum(
                            probabilities
                        ),
                        1.0,
                        atol=1e-3
                    )
                ):

                    pass

                else:

                    probabilities = (
                        tf.nn.softmax(
                            probabilities
                        ).numpy()
                    )

                # ---------------------------------------------
                # Expected:
                #
                # 0 = Normal
                # 1 = Pneumonia
                # ---------------------------------------------

                normal_probability = float(
                    probabilities[0]
                )

                pneumonia_probability = float(
                    probabilities[1]
                )

            else:

                st.error(
                    "Unsupported pneumonia model output."
                )

                st.write(
                    "Model output:",
                    prediction_values
                )

                st.stop()

            # =================================================
            # CLAMP PROBABILITIES
            # =================================================

            pneumonia_probability = float(
                np.clip(
                    pneumonia_probability,
                    0.0,
                    1.0
                )
            )

            normal_probability = float(
                np.clip(
                    normal_probability,
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

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "X-ray Confidence",
                    f"{xray_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Normal Probability",
                    f"{normal_probability * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Pneumonia Probability",
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
                "**Image Modality:** Chest X-ray"
            )

            st.write(
                f"**X-ray Confidence:** "
                f"{xray_probability * 100:.2f}%"
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

            pdf_data = create_pdf_report(

                filename=uploaded_file.name,

                modality="Chest X-ray",

                modality_confidence=(
                    xray_probability
                ),

                diagnosis=diagnosis,

                diagnosis_confidence=(
                    diagnosis_confidence
                ),

                xray_confidence=(
                    xray_probability
                )
            )

            # =================================================
            # DOWNLOAD PDF
            # =================================================

            clean_filename = (

                os.path.splitext(
                    uploaded_file.name
                )[0]

                .replace(
                    " ",
                    "_"
                )
            )

            st.download_button(

                label=(
                    "Download Diagnostic Report (PDF)"
                ),

                data=pdf_data,

                file_name=(
                    f"{clean_filename}_"
                    f"Pneumonia_Report.pdf"
                ),

                mime="application/pdf"
            )


    except Exception as e:

        st.error(
            "An error occurred while "
            "processing the image."
        )

        st.exception(e)
