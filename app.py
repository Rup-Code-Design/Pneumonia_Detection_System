# ============================================================
# app.py
# ============================================================
#
# SYSTEM:
#
# 1. Reject genuinely coloured images
#
# 2. Classify:
#       0 = CHEST_XRAY
#       1 = CT
#       2 = MRI
#
# 3. If CHEST_XRAY:
#       Run pneumonia classifier
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Image AI",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODALITY_MODEL_PATH = "CT_Verifier.keras"

PNEUMONIA_MODEL_PATH = (
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# CLASS MAPPING
# ============================================================
#
# This MUST match:
#
# {'CHEST_XRAY': 0, 'CT': 1, 'MRI': 2}
#
# from your training generator.
# ============================================================

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


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
            f"Modality model not found:\n"
            f"{MODALITY_MODEL_PATH}"
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

    if not os.path.isfile(
        PNEUMONIA_MODEL_PATH
    ):

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
# VERIFY MODALITY MODEL
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
        "Could not determine modality model shape."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CHECK INPUT SHAPE
# ============================================================

if modality_input_shape[-3:] != (
    128,
    128,
    3
):

    st.error(
        "Unexpected modality model input shape."
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


# ============================================================
# CHECK OUTPUT SHAPE
# ============================================================

if modality_output_shape[-1] != 3:

    st.error(
        "CT_Verifier.keras is not a 3-class "
        "modality classifier."
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
# COLOUR IMAGE DETECTION
# ============================================================

def is_genuinely_coloured_image(
    image,
    mean_difference_threshold=12.0,
    coloured_pixel_threshold=0.05
):
    """
    Detect genuinely coloured images.

    IMPORTANT:
    An X-ray may be stored as RGB but still be
    visually grayscale.

    Therefore we do NOT reject every RGB image.

    We examine actual channel differences.
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
        channel_difference
        >
        mean_difference_threshold
    )

    coloured_ratio = float(
        np.mean(coloured_pixels)
    )

    is_colour = (
        mean_difference
        >
        mean_difference_threshold
        and
        coloured_ratio
        >
        coloured_pixel_threshold
    )

    return (
        is_colour,
        mean_difference,
        coloured_ratio
    )


# ============================================================
# PDF REPORT FUNCTION
# ============================================================

def create_pdf_report(
    filename,
    modality,
    modality_confidence,
    diagnosis=None,
    diagnosis_confidence=None,
    xray_probability=None,
    ct_probability=None,
    mri_probability=None
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
    # ALL MODALITY PROBABILITIES
    # --------------------------------------------------------

    if xray_probability is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            55,
            8,
            "Chest X-ray Probability:",
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
            f"{xray_probability * 100:.2f}%",
            ln=True
        )

    if ct_probability is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            55,
            8,
            "CT Probability:",
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
            f"{ct_probability * 100:.2f}%",
            ln=True
        )

    if mri_probability is not None:

        pdf.set_font(
            "Arial",
            "B",
            11
        )

        pdf.cell(
            55,
            8,
            "MRI Probability:",
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
            f"{mri_probability * 100:.2f}%",
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
        "Disclaimer: This AI-generated result is "
        "intended for research purposes only and "
        "does not replace professional medical diagnosis."
    )

    return bytes(
        pdf.output()
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
### Processing Pipeline

**Image → Colour Check → Modality Classification → Pneumonia Detection**

The modality classifier identifies:

- **Chest X-ray**
- **CT Scan**
- **MRI**

Only a **Chest X-ray** is passed to the pneumonia model.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    st.write(
        "Modality Model"
    )

    st.code(
        "CT_Verifier.keras"
    )

    st.write(
        "Input: 128 × 128 × 3"
    )

    st.write(
        "Output: 3 classes"
    )

    st.divider()

    st.write(
        "Pneumonia Model"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )

    st.write(
        "Input: 224 × 224 × 3"
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

        original_image = Image.open(
            io.BytesIO(file_bytes)
        )

        original_image.load()

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
            f"{original_image.width} × "
            f"{original_image.height}"
        )

        st.write(
            f"**Original image mode:** "
            f"{original_image.mode}"
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            # =================================================
            # STEP 0 — COLOUR CHECK
            # =================================================

            st.subheader(
                "Step 0 — Colour Image Check"
            )

            (
                colour_detected,
                mean_colour_difference,
                coloured_ratio
            ) = is_genuinely_coloured_image(
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
                    "Download Rejection Report",
                    data=pdf_data,
                    file_name=(
                        "Colour_Image_Rejection_Report.pdf"
                    ),
                    mime="application/pdf"
                )

                st.stop()

            st.success(
                "Image passes the colour check."
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
            # CRITICAL:
            #
            # TRAINING:
            #
            # ImageDataGenerator(
            #     rescale=1.0 / 255.0
            # )
            #
            # APP:
            #
            # EXACTLY /255.0
            #
            # DO NOT USE:
            # MobileNetV2 preprocess_input()
            # =================================================

            modality_array = (
                modality_array / 255.0
            )

            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )

            # =================================================
            # DEBUG INFORMATION
            # =================================================

            st.write(
                f"Model input shape: "
                f"{modality_input.shape}"
            )

            st.write(
                f"Input range: "
                f"{modality_input.min():.4f} "
                f"to "
                f"{modality_input.max():.4f}"
            )

            # =================================================
            # PREDICTION
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

            if modality_prediction.shape != (
                1,
                3
            ):

                st.error(
                    "Unexpected modality model output."
                )

                st.write(
                    "Output:",
                    modality_prediction
                )

                st.write(
                    "Shape:",
                    modality_prediction.shape
                )

                st.stop()

            # =================================================
            # GET OUTPUT
            # =================================================

            probabilities = (
                modality_prediction[0]
            )

            # =================================================
            # SAFETY NORMALIZATION
            # =================================================

            if (
                np.any(probabilities < 0)
                or
                np.any(probabilities > 1)
                or
                not np.isclose(
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

            # =================================================
            # EXTRACT
            # =================================================

            xray_probability = float(
                probabilities[0]
            )

            ct_probability = float(
                probabilities[1]
            )

            mri_probability = float(
                probabilities[2]
            )

            # =================================================
            # PREDICT CLASS
            # =================================================

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            predicted_modality = (
                MODALITY_CLASS_NAMES[
                    predicted_index
                ]
            )

            predicted_confidence = float(
                probabilities[
                    predicted_index
                ]
            )

            # =================================================
            # DISPLAY RAW MODEL RESULT
            # =================================================

            st.write(
                "### Modality Probabilities"
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

            st.write(
                f"**Predicted class:** "
                f"{predicted_modality}"
            )

            st.write(
                f"**Confidence:** "
                f"{predicted_confidence * 100:.2f}%"
            )

            # =================================================
            # ROUTING
            # =================================================
            #
            # IMPORTANT:
            #
            # NO arbitrary X-ray threshold here.
            #
            # The model's highest-probability class
            # determines the modality.
            #
            # This prevents:
            #
            # X-ray = 45%
            # CT     = 35%
            # MRI    = 20%
            #
            # from being incorrectly rejected as
            # "low-confidence X-ray".
            # =================================================

            # =================================================
            # CT
            # =================================================

            if predicted_index == 1:

                st.error(
                    "CT Scan detected"
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )

                st.info(
                    "The image is a CT scan. "
                    "Pneumonia detection is not performed."
                )

                st.session_state.history.append(
                    "CT Scan - "
                    + uploaded_file.name
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="CT Scan",
                    modality_confidence=(
                        ct_probability
                    ),
                    xray_probability=(
                        xray_probability
                    ),
                    ct_probability=(
                        ct_probability
                    ),
                    mri_probability=(
                        mri_probability
                    )
                )

                st.download_button(
                    "Download CT Report",
                    data=pdf_data,
                    file_name="CT_Scan_Report.pdf",
                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # MRI
            # =================================================

            if predicted_index == 2:

                st.error(
                    "MRI image detected"
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )

                st.info(
                    "The image is an MRI scan. "
                    "Pneumonia detection is not performed."
                )

                st.session_state.history.append(
                    "MRI - "
                    + uploaded_file.name
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="MRI",
                    modality_confidence=(
                        mri_probability
                    ),
                    xray_probability=(
                        xray_probability
                    ),
                    ct_probability=(
                        ct_probability
                    ),
                    mri_probability=(
                        mri_probability
                    )
                )

                st.download_button(
                    "Download MRI Report",
                    data=pdf_data,
                    file_name="MRI_Report.pdf",
                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # CHEST X-RAY
            # =================================================

            if predicted_index == 0:

                st.success(
                    "Chest X-ray detected"
                )

                st.write(
                    f"Chest X-ray confidence: "
                    f"{xray_probability * 100:.2f}%"
                )

                # =============================================
                # STEP 2
                # =============================================

                st.subheader(
                    "Step 2 — Pneumonia Detection"
                )

                # =============================================
                # PREPARE IMAGE
                # =============================================

                pneumonia_image = image_rgb.resize(
                    PNEUMONIA_IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                pneumonia_array = np.asarray(
                    pneumonia_image,
                    dtype=np.float32
                )

                # =============================================
                # PNEUMONIA PREPROCESSING
                #
                # Assumes your pneumonia model was trained
                # using /255.0.
                # =============================================

                pneumonia_array = (
                    pneumonia_array / 255.0
                )

                pneumonia_input = np.expand_dims(
                    pneumonia_array,
                    axis=0
                )

                # =============================================
                # PREDICTION
                # =============================================

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

                # =============================================
                # OUTPUT
                # =============================================

                if prediction_values.size == 1:

                    pneumonia_probability = float(
                        prediction_values
                    )

                    normal_probability = (
                        1.0
                        -
                        pneumonia_probability
                    )

                elif prediction_values.size == 2:

                    pneumonia_probabilities = (
                        prediction_values.astype(
                            np.float64
                        )
                    )

                    # If output is logits
                    if not (
                        np.all(
                            pneumonia_probabilities >= 0
                        )
                        and
                        np.all(
                            pneumonia_probabilities <= 1
                        )
                        and
                        np.isclose(
                            np.sum(
                                pneumonia_probabilities
                            ),
                            1.0,
                            atol=1e-3
                        )
                    ):

                        pneumonia_probabilities = (
                            tf.nn.softmax(
                                pneumonia_probabilities
                            ).numpy()
                        )

                    # Assumed:
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
                        "Model output:",
                        prediction_values
                    )

                    st.stop()

                # =============================================
                # CLAMP
                # =============================================

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

                # =============================================
                # DIAGNOSIS
                # =============================================

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

                # =============================================
                # FINAL RESULT
                # =============================================

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

                # =============================================
                # HISTORY
                # =============================================

                st.session_state.history.append(
                    f"{diagnosis} - "
                    f"{uploaded_file.name}"
                )

                # =============================================
                # REPORT
                # =============================================

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

                # =============================================
                # PDF
                # =============================================

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
                    xray_probability=(
                        xray_probability
                    ),
                    ct_probability=(
                        ct_probability
                    ),
                    mri_probability=(
                        mri_probability
                    )
                )

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
                    "Download Diagnostic Report (PDF)",
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
