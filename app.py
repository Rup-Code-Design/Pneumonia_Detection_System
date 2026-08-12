# ============================================================
# app.py
# ============================================================
#
# MEDICAL IMAGE MODALITY + PNEUMONIA DETECTION SYSTEM
#
# Pipeline:
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

from PIL import Image, ImageOps
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
# 4. CLASS MAPPING
# ============================================================
#
# MUST MATCH:
#
# train_generator.class_indices
#
# Your training code is expected to produce:
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
# 5. PNEUMONIA THRESHOLD
# ============================================================

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
            f"\nModality model not found:\n"
            f"{MODALITY_MODEL_PATH}\n\n"
            f"Make sure CT_Verifier.keras is "
            f"in the same folder as app.py."
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
            f"\nPneumonia model not found:\n"
            f"{PNEUMONIA_MODEL_PATH}"
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
        "Could not read modality model structure."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    st.write(
        "Modality model:"
    )

    st.code(
        "CT_Verifier.keras"
    )

    st.write(
        "Modality input:"
    )

    st.code(
        str(modality_input_shape)
    )

    st.write(
        "Modality output:"
    )

    st.code(
        str(modality_output_shape)
    )

    st.divider()

    st.write(
        "Pneumonia model:"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
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
# 12. VERIFY MODALITY OUTPUT
# ============================================================

if (
    modality_output_shape is not None
    and
    modality_output_shape[-1] != 3
):

    st.error(
        "CT_Verifier.keras is not a 3-class "
        "modality classifier."
    )

    st.write(
        "Expected output:"
    )

    st.code(
        "(None, 3)"
    )

    st.write(
        "Actual output:"
    )

    st.code(
        str(modality_output_shape)
    )

    st.stop()


# ============================================================
# 13. HEADER
# ============================================================

st.title(
    "Medical Image Modality & Pneumonia Detection"
)

st.markdown(
    """
### Processing Pipeline

**Step 1:** Reject obvious colour images.

**Step 2:** Identify the medical image modality:

- Chest X-ray
- CT Scan
- MRI

**Step 3:** Only Chest X-ray images proceed to pneumonia detection.

**Step 4:** Classify the Chest X-ray as:

- Normal
- Pneumonia

**Step 5:** Generate a downloadable PDF report.
"""
)


# ============================================================
# 14. IMAGE UPLOADER
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
# 15. COLOUR IMAGE DETECTOR
# ============================================================

def is_obvious_colour_image(
    image,
    saturation_threshold=18.0,
    colour_pixel_ratio=0.08
):
    """
    Detect strongly coloured images.

    Important:
    An X-ray may be stored as RGB even though
    it is visually grayscale.

    Therefore we inspect actual channel
    differences rather than checking image.mode.
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
        channel_difference >
        saturation_threshold
    )

    coloured_ratio = float(
        np.mean(coloured_pixels)
    )

    is_colour = (
        mean_difference >
        saturation_threshold
        and
        coloured_ratio >
        colour_pixel_ratio
    )

    return (
        is_colour,
        mean_difference,
        coloured_ratio
    )


# ============================================================
# 16. PDF REPORT FUNCTION
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
    # FILE
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
        "Disclaimer: This AI-generated result is "
        "intended for research purposes only and "
        "does not replace professional medical "
        "diagnosis."
    )

    return bytes(
        pdf.output()
    )


# ============================================================
# 17. PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        file_bytes = (
            uploaded_file.getvalue()
        )

        original_image = Image.open(
            io.BytesIO(file_bytes)
        )

        original_image.load()

        # ----------------------------------------------------
        # Correct orientation using EXIF
        # ----------------------------------------------------

        original_image = (
            ImageOps.exif_transpose(
                original_image
            )
        )

        image_rgb = original_image.convert(
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

        st.write(
            f"**Image size:** "
            f"{image_rgb.width} × "
            f"{image_rgb.height}"
        )

        st.write(
            f"**Original image mode:** "
            f"{original_image.mode}"
        )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            # =================================================
            # STEP 0 — COLOUR CHECK
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
                    "Chest X-ray, CT, or MRI image."
                )

                st.session_state.history.append(
                    "Rejected - Colour image - "
                    + uploaded_file.name
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="Rejected - Colour Image",
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
            # PREPROCESSING
            # =================================================
            #
            # TRAINING:
            #
            # ImageDataGenerator(
            #     rescale=1.0 / 255.0
            # )
            #
            # Therefore:
            #
            # resize -> float32 -> /255
            #
            # Do NOT use MobileNetV2 preprocess_input().
            #
            # =================================================

            modality_image = image_rgb.resize(
                MODALITY_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            modality_array = np.asarray(
                modality_image,
                dtype=np.float32
            )

            modality_array /= 255.0

            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )

            # =================================================
            # DEBUG INFORMATION
            # =================================================

            with st.expander(
                "Modality model diagnostic information"
            ):

                st.write(
                    "**Model input shape:**",
                    modality_model.input_shape
                )

                st.write(
                    "**Model output shape:**",
                    modality_model.output_shape
                )

                st.write(
                    "**Application input shape:**",
                    modality_input.shape
                )

                st.write(
                    "**Application input minimum:**",
                    float(
                        modality_input.min()
                    )
                )

                st.write(
                    "**Application input maximum:**",
                    float(
                        modality_input.max()
                    )
                )

            # =================================================
            # MODEL PREDICTION
            # =================================================

            with st.spinner(
                "Identifying image modality..."
            ):

                raw_prediction = (
                    modality_model.predict(
                        modality_input,
                        verbose=0
                    )
                )

            raw_prediction = np.asarray(
                raw_prediction,
                dtype=np.float64
            )

            # =================================================
            # VALIDATE OUTPUT
            # =================================================

            if raw_prediction.ndim != 2:

                st.error(
                    "Invalid modality model output."
                )

                st.write(
                    "Output:",
                    raw_prediction
                )

                st.stop()

            if raw_prediction.shape[1] != 3:

                st.error(
                    "CT_Verifier.keras must "
                    "produce exactly 3 outputs."
                )

                st.write(
                    "Actual output shape:",
                    raw_prediction.shape
                )

                st.stop()

            # =================================================
            # EXTRACT MODEL OUTPUT
            # =================================================

            modality_probabilities = (
                raw_prediction[0]
            )

            # =================================================
            # IMPORTANT:
            #
            # CT_Verifier.keras already has:
            #
            # Dense(3, activation='softmax')
            #
            # Therefore DO NOT apply another softmax
            # unless the output is clearly not probabilities.
            # =================================================

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
            # CLASS PROBABILITIES
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
            # PREDICT CLASS
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
            # DISPLAY RAW PREDICTION
            # =================================================

            st.write(
                "### Modality Prediction"
            )

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
            # SHOW EXACT MODEL DECISION
            # =================================================

            st.write(
                f"**Predicted modality:** "
                f"{predicted_modality}"
            )

            st.write(
                f"**Confidence:** "
                f"{predicted_confidence * 100:.2f}%"
            )

            # =================================================
            # IMPORTANT FIX
            # =================================================
            #
            # DO NOT use:
            #
            #     xray_probability >= 0.50
            #
            # to decide whether it is X-ray.
            #
            # The model is a 3-class classifier.
            #
            # We use its highest probability:
            #
            # argmax([Xray, CT, MRI])
            #
            # If X-ray is the highest class,
            # classify as Chest X-ray.
            #
            # =================================================

            if predicted_index == 0:

                # =================================================
                # CHEST X-RAY
                # =================================================

                st.success(
                    "Chest X-ray detected."
                )

                st.write(
                    f"Chest X-ray model confidence: "
                    f"{xray_probability * 100:.2f}%"
                )

            elif predicted_index == 1:

                # =================================================
                # CT
                # =================================================

                st.error(
                    "CT Scan detected."
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )

                st.info(
                    "This application does not run "
                    "pneumonia detection on CT images."
                )

                st.session_state.history.append(
                    "CT Scan - "
                    + uploaded_file.name
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

            else:

                # =================================================
                # MRI
                # =================================================

                st.error(
                    "MRI image detected."
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )

                st.info(
                    "This application does not run "
                    "pneumonia detection on MRI images."
                )

                st.session_state.history.append(
                    "MRI - "
                    + uploaded_file.name
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
            # STEP 2 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            # =================================================
            # PNEUMONIA PREPROCESSING
            # =================================================

            pneumonia_image = image_rgb.resize(
                PNEUMONIA_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            pneumonia_array = np.asarray(
                pneumonia_image,
                dtype=np.float32
            )

            # -------------------------------------------------
            # Your previous pneumonia app uses /255
            # -------------------------------------------------

            pneumonia_array /= 255.0

            pneumonia_input = np.expand_dims(
                pneumonia_array,
                axis=0
            )

            # =================================================
            # PREDICT PNEUMONIA
            # =================================================

            with st.spinner(
                "Analyzing Chest X-ray..."
            ):

                pneumonia_prediction = (
                    pneumonia_model.predict(
                        pneumonia_input,
                        verbose=0
                    )
                )

            prediction_values = np.squeeze(
                pneumonia_prediction
            )

            # =================================================
            # OUTPUT HANDLING
            # =================================================

            if prediction_values.size == 1:

                pneumonia_probability = float(
                    prediction_values
                )

                normal_probability = (
                    1.0 -
                    pneumonia_probability
                )

            elif prediction_values.size == 2:

                probabilities = (
                    prediction_values
                    .astype(np.float64)
                )

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

                # Expected:
                #
                # 0 = Normal
                # 1 = Pneumonia

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

            st.session_state.history.append(
                f"{diagnosis} - "
                f"{uploaded_file.name}"
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
                modality_confidence=xray_probability,
                diagnosis=diagnosis,
                diagnosis_confidence=diagnosis_confidence
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
            "An error occurred while processing "
            "the image."
        )

        st.exception(e)
