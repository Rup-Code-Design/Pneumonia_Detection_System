# ============================================================
# app.py
# ============================================================
#
# SYSTEM:
#
# 1. Reject obvious colour images
#
# 2. Classify modality:
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

# MUST MATCH:
#
# CHEST_XRAY = 0
# CT         = 1
# MRI        = 2

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
            "Modality model not found:\n"
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
            "Pneumonia model not found:\n"
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
# MODEL INFORMATION
# ============================================================

st.sidebar.header(
    "Model Information"
)

st.sidebar.write(
    "Modality Model:"
)

st.sidebar.code(
    "CT_Verifier.keras"
)

st.sidebar.write(
    "Pneumonia Model:"
)

st.sidebar.code(
    "best_xception_pneumonia_model.keras"
)


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
# CHECK MODEL SHAPE
# ============================================================

if (
    modality_input_shape[-3:]
    != (128, 128, 3)
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


if (
    modality_output_shape[-1]
    != 3
):

    st.error(
        "CT_Verifier.keras must have 3 outputs."
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
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
This system first identifies the medical image modality.

**Supported modalities**

- Chest X-ray
- CT Scan
- MRI

Only a detected **Chest X-ray** is passed to the
pneumonia detection model.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.divider()

    st.write(
        "**Modality Input Shape:**"
    )

    st.code(
        str(modality_input_shape)
    )

    st.write(
        "**Modality Output Shape:**"
    )

    st.code(
        str(modality_output_shape)
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
# COLOUR IMAGE DETECTOR
# ============================================================

def is_obvious_colour_image(
    image,
    channel_difference_threshold=18.0,
    coloured_pixel_ratio_threshold=0.08
):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    maximum = np.maximum(
        np.maximum(r, g),
        b
    )

    minimum = np.minimum(
        np.minimum(r, g),
        b
    )

    difference = (
        maximum - minimum
    )

    mean_difference = float(
        np.mean(difference)
    )

    coloured_pixels = (
        difference >
        channel_difference_threshold
    )

    coloured_ratio = float(
        np.mean(coloured_pixels)
    )

    is_colour = (
        mean_difference >
        channel_difference_threshold
        and
        coloured_ratio >
        coloured_pixel_ratio_threshold
    )

    return (
        is_colour,
        mean_difference,
        coloured_ratio
    )


# ============================================================
# PDF REPORT
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

        # ----------------------------------------------------
        # DIAGNOSIS CONFIDENCE
        # ----------------------------------------------------

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
# IMAGE PROCESSING
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

            if colour_detected:

                st.error(
                    "Colour image detected."
                )

                st.warning(
                    "Please upload a grayscale "
                    "Chest X-ray image."
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
                    file_name="Colour_Image_Rejection_Report.pdf",
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
            # IMPORTANT:
            #
            # flow_from_directory() training:
            #
            # target_size=(128,128)
            #
            # default interpolation = nearest
            #
            # Therefore use NEAREST here.
            # =================================================

            resized_image = image_rgb.resize(
                MODALITY_IMAGE_SIZE,
                Image.Resampling.NEAREST
            )

            modality_array = np.asarray(
                resized_image,
                dtype=np.float32
            )

            # =================================================
            # EXACT TRAINING SCALE
            # =================================================

            modality_array = (
                modality_array / 255.0
            )

            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )

            # =================================================
            # DEBUG
            # =================================================

            st.write(
                "**Modality input shape:**",
                modality_input.shape
            )

            st.write(
                "**Modality input range:**",
                f"{modality_input.min():.4f} "
                f"to "
                f"{modality_input.max():.4f}"
            )

            # =================================================
            # PREDICT
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
            # OUTPUT CHECK
            # =================================================

            if raw_prediction.shape != (1, 3):

                st.error(
                    "Unexpected modality model output."
                )

                st.write(
                    "Expected:",
                    "(1, 3)"
                )

                st.write(
                    "Actual:",
                    raw_prediction.shape
                )

                st.stop()

            # =================================================
            # GET PROBABILITIES
            # =================================================

            modality_probabilities = (
                raw_prediction[0]
            )

            # =================================================
            # SAFETY NORMALIZATION
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
            # PROBABILITIES
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
            # PREDICTION
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
            # DISPLAY RESULTS
            # =================================================

            st.markdown(
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

            # =================================================
            # DEBUG INFORMATION
            # =================================================

            with st.expander(
                "Technical Model Output"
            ):

                st.write(
                    "Raw model output:",
                    raw_prediction
                )

                st.write(
                    "Probability sum:",
                    float(
                        np.sum(
                            modality_probabilities
                        )
                    )
                )

                st.write(
                    "Predicted index:",
                    predicted_index
                )

                st.write(
                    "Predicted modality:",
                    predicted_modality
                )

                st.write(
                    "Predicted confidence:",
                    f"{predicted_confidence * 100:.4f}%"
                )

            # =================================================
            # DECISION
            # =================================================
            #
            # IMPORTANT CHANGE:
            #
            # We do NOT use:
            #
            # xray_probability >= 0.50
            #
            # because that threshold was not part of
            # your model training.
            #
            # The classifier's highest probability determines
            # the modality.
            #
            # =================================================

            # =================================================
            # CT
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
                    "This application does not perform "
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
                    "MRI image detected."
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )

                st.info(
                    "This application does not perform "
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
                    "Chest X-ray detected."
                )

                st.write(
                    f"Chest X-ray confidence: "
                    f"{xray_probability * 100:.2f}%"
                )

            # =================================================
            # STEP 2 — PNEUMONIA
            # =================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            # =================================================
            # RESIZE PNEUMONIA IMAGE
            # =================================================

            pneumonia_image = image_rgb.resize(
                PNEUMONIA_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            pneumonia_array = np.asarray(
                pneumonia_image,
                dtype=np.float32
            )

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

                if not (
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
            # PDF
            # =================================================

            pdf_data = create_pdf_report(
                filename=uploaded_file.name,
                modality="Chest X-ray",
                modality_confidence=xray_probability,
                diagnosis=diagnosis,
                diagnosis_confidence=diagnosis_confidence
            )

            clean_filename = (
                os.path.splitext(
                    uploaded_file.name
                )[0]
                .replace(" ", "_")
            )

            st.download_button(
                label="Download Diagnostic Report (PDF)",
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
