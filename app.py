# ============================================================
# app.py
# ============================================================
#
# SYSTEM:
#
# 1. Reject obvious colour images
# 2. Classify medical image:
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
# IMPORTANT:
# CT_Verifier.keras was trained with:
#
#   ImageDataGenerator(rescale=1/255.0)
#
# Therefore the app ALSO uses /255.0.
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
# MODEL PATHS
# ============================================================

MODALITY_MODEL_PATH = "CT_Verifier.keras"

PNEUMONIA_MODEL_PATH = "best_xception_pneumonia_model.keras"


# ============================================================
# IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# MODALITY CLASS MAPPING
# ============================================================
#
# MUST MATCH YOUR TRAINING CODE:
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
# THRESHOLD
# ============================================================
#
# The predicted class must be CHEST_XRAY AND its probability
# must reach this threshold.
#
# Do NOT lower this excessively.
#
# ============================================================

CHEST_XRAY_THRESHOLD = 0.50


# ============================================================
# PNEUMONIA THRESHOLD
# ============================================================

PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia AI",
    page_icon="🫁",
    layout="wide"
)


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

    if not os.path.isfile(MODALITY_MODEL_PATH):

        raise FileNotFoundError(
            f"Modality model not found:\n"
            f"{MODALITY_MODEL_PATH}\n\n"
            f"Upload CT_Verifier.keras to the same "
            f"directory as app.py."
        )

    model = tf.keras.models.load_model(
        MODALITY_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(PNEUMONIA_MODEL_PATH):

        raise FileNotFoundError(
            f"Pneumonia model not found:\n"
            f"{PNEUMONIA_MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        PNEUMONIA_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    modality_model = load_modality_model()

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY MODALITY MODEL OUTPUT
# ============================================================

try:

    modality_output_shape = (
        modality_model.output_shape
    )

except Exception:

    modality_output_shape = None


if modality_output_shape is not None:

    try:

        number_of_outputs = (
            modality_output_shape[-1]
        )

        if number_of_outputs != 3:

            st.error(
                "CT_Verifier.keras does not appear "
                "to be a 3-class modality classifier."
            )

            st.write(
                "Expected output:",
                "(None, 3)"
            )

            st.write(
                "Actual output:",
                modality_output_shape
            )

            st.stop()

    except Exception:
        pass


# ============================================================
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
This system first identifies the image modality.

**Supported modalities**

- Chest X-ray
- CT Scan
- MRI

Only a confirmed **Chest X-ray** is passed to the
pneumonia detection model.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.write(
        "Modality Model:"
    )

    st.code(
        "CT_Verifier.keras"
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

    st.divider()

    st.header("Recent Scans")

    if len(st.session_state.history) == 0:

        st.write(
            "No scans yet."
        )

    else:

        for item in reversed(
            st.session_state.history[-10:]
        ):

            st.text(item)


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
        "Upload a Chest X-ray, CT scan, or MRI image."
    )
)


# ============================================================
# HELPER:
# CHECK WHETHER IMAGE IS OBVIOUSLY COLOUR
# ============================================================

def is_obvious_colour_image(
    image,
    saturation_threshold=18.0,
    colour_pixel_ratio=0.08
):
    """
    Detect strongly coloured images.

    Important:
    Many X-rays are saved as RGB files even though they
    visually contain grayscale information.

    Therefore we DO NOT simply check image.mode == RGB.

    Instead we examine actual colour differences.
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
        np.mean(channel_difference)
    )

    coloured_pixels = (
        channel_difference > saturation_threshold
    )

    coloured_ratio = float(
        np.mean(coloured_pixels)
    )

    is_colour = (
        mean_difference > saturation_threshold
        and
        coloured_ratio > colour_pixel_ratio
    )

    return (
        is_colour,
        mean_difference,
        coloured_ratio
    )


# ============================================================
# HELPER:
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    filename,
    modality,
    modality_confidence,
    diagnosis=None,
    diagnosis_confidence=None
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
        50,
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
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    pdf.set_font(
        "Arial",
        "B",
        11
    )

    pdf.cell(
        50,
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
        50,
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
        50,
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
    # DIAGNOSIS
    # --------------------------------------------------------

    if diagnosis is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            50,
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

        # ----------------------------------------------------
        # DIAGNOSIS CONFIDENCE
        # ----------------------------------------------------

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            50,
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
# PROCESS IMAGE
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

        image.load()

        image_rgb = image.convert(
            "RGB"
        )

        # ====================================================
        # DISPLAY
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
            f"**Image mode:** "
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
                f"Colour difference: "
                f"{mean_colour_difference:.2f}"
            )

            st.write(
                f"Coloured pixel ratio: "
                f"{coloured_ratio * 100:.2f}%"
            )

            if colour_detected:

                st.error(
                    "Colour image detected."
                )

                st.warning(
                    "Please upload a grayscale "
                    "Chest X-ray image."
                )

                history_entry = (
                    f"Rejected - Colour image - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                # ---------------------------------------------
                # PDF REPORT
                # ---------------------------------------------

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="Rejected — Colour Image",
                    modality_confidence=0.0
                )

                st.download_button(
                    label="Download Rejection Report",
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
                "Step 1 — Modality Classification"
            )

            # =================================================
            # RESIZE
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
            # Training used:
            #
            # rescale=1.0 / 255.0
            #
            # Therefore use exactly:
            #
            # /255.0
            #
            # NOT MobileNetV2 preprocess_input()
            # =================================================

            modality_array = (
                modality_array / 255.0
            )

            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )

            # =================================================
            # PREDICT
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

            if modality_prediction.ndim != 2:

                st.error(
                    "Invalid modality model output."
                )

                st.stop()

            if modality_prediction.shape[1] != 3:

                st.error(
                    "CT_Verifier.keras must output "
                    "exactly 3 classes."
                )

                st.write(
                    "Actual output shape:",
                    modality_prediction.shape
                )

                st.stop()

            # =================================================
            # PROBABILITIES
            # =================================================

            modality_probabilities = (
                modality_prediction[0]
            )

            # Safety: normalize if necessary
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
            # DISPLAY PROBABILITIES
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
            # DECISION
            # =================================================

            # Chest X-ray is accepted ONLY when:
            #
            # 1. X-ray is the highest class
            # 2. X-ray probability >= threshold

            is_chest_xray = (

                predicted_index == 0

                and

                xray_probability
                >= CHEST_XRAY_THRESHOLD
            )

            # =================================================
            # CT
            # =================================================

            if predicted_modality == "CT":

                st.error(
                    "CT Scan detected"
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )

                st.info(
                    "This application accepts only "
                    "Chest X-ray images for pneumonia "
                    "detection."
                )

                history_entry = (
                    f"CT Scan - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="CT Scan",
                    modality_confidence=ct_probability
                )

                st.download_button(
                    label="Download CT Report",
                    data=pdf_data,
                    file_name="CT_Scan_Report.pdf",
                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # MRI
            # =================================================

            if predicted_modality == "MRI":

                st.error(
                    "MRI image detected"
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )

                st.info(
                    "This application accepts only "
                    "Chest X-ray images for pneumonia "
                    "detection."
                )

                history_entry = (
                    f"MRI - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="MRI",
                    modality_confidence=mri_probability
                )

                st.download_button(
                    label="Download MRI Report",
                    data=pdf_data,
                    file_name="MRI_Report.pdf",
                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # LOW-CONFIDENCE X-RAY
            # =================================================

            if not is_chest_xray:

                st.error(
                    "Chest X-ray could not be confirmed."
                )

                st.write(
                    f"Predicted modality: "
                    f"**{predicted_modality}**"
                )

                st.write(
                    f"X-ray probability: "
                    f"{xray_probability * 100:.2f}%"
                )

                st.warning(
                    "Please upload a clear Chest X-ray image."
                )

                history_entry = (
                    f"Rejected - "
                    f"{predicted_modality} - "
                    f"{uploaded_file.name}"
                )

                st.session_state.history.append(
                    history_entry
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality=(
                        "Unconfirmed / Unsupported"
                    ),
                    modality_confidence=(
                        predicted_confidence
                    )
                )

                st.download_button(
                    label="Download Rejection Report",
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
            # PNEUMONIA MODEL PREPROCESSING
            #
            # Your previous app used /255.0.
            #
            # Keep this if your pneumonia model was trained
            # using /255.0.
            # =================================================

            pneumonia_array = (
                pneumonia_array / 255.0
            )

            pneumonia_input = np.expand_dims(
                pneumonia_array,
                axis=0
            )

            # =================================================
            # PREDICTION
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
            # HANDLE MODEL OUTPUT
            # =================================================

            normal_probability = None

            pneumonia_probability = None

            # -------------------------------------------------
            # SINGLE OUTPUT
            # -------------------------------------------------

            if prediction_values.size == 1:

                pneumonia_probability = float(
                    prediction_values
                )

                normal_probability = (
                    1.0 -
                    pneumonia_probability
                )

            # -------------------------------------------------
            # TWO OUTPUTS
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
                        np.sum(probabilities),
                        1.0,
                        atol=1e-3
                    )
                ):

                    probabilities = (
                        probabilities
                    )

                else:

                    probabilities = (
                        tf.nn.softmax(
                            probabilities
                        ).numpy()
                    )

                # Expected:
                #
                # 0 = Normal
                # 1 = Pneumonia

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
                    "Output:",
                    prediction_values
                )

                st.stop()

            # =================================================
            # CLAMP
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
            # FINAL RESULTS
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

            st.session_state.history.append(
                history_entry
            )

            # =================================================
            # REPORT
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
                )
            )

            # =================================================
            # DOWNLOAD
            # =================================================

            clean_filename = (
                os.path.splitext(
                    uploaded_file.name
                )[0]
                .replace(" ", "_")
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
