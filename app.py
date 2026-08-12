# ============================================================
# app.py
# CT vs MRI vs CHEST X-RAY + PNEUMONIA DETECTION
# ============================================================
#
# PIPELINE
#
# 1. Upload image
# 2. Reject strongly coloured image
# 3. CT_Verifier.keras
#       0 = CHEST_XRAY
#       1 = CT
#       2 = MRI
# 4. If CHEST_XRAY -> pneumonia model
# 5. CT -> CT detected
# 6. MRI -> MRI detected
# 7. Generate PDF report
#
# IMPORTANT:
#
# CT_Verifier.keras training:
#
# ImageDataGenerator(
#     rescale=1.0 / 255.0
# )
#
# Therefore the app uses EXACTLY:
#
# RGB -> resize 128x128 -> float32 -> /255.0
#
# NO MobileNetV2 preprocess_input()
# NO grayscale conversion
# NO contrast enhancement
# ============================================================


# ============================================================
# 1. IMPORTS
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
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# 3. MODEL PATHS
# ============================================================

MODALITY_MODEL_PATH = "CT_Verifier.keras"

PNEUMONIA_MODEL_PATH = (
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# 4. IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# 5. CLASS MAPPING
# ============================================================

# MUST MATCH TRAINING:
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
# 6. THRESHOLDS
# ============================================================

# We do NOT use a high arbitrary X-ray threshold.
#
# The model prediction itself determines the modality.
#
# A small minimum confidence is used only to reject
# obviously uncertain predictions.

MODALITY_MIN_CONFIDENCE = 0.50

PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# 7. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# 8. LOAD MODALITY MODEL
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
# 9. LOAD PNEUMONIA MODEL
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
# 10. LOAD MODELS
# ============================================================

try:

    modality_model = load_modality_model()

except Exception as e:

    st.error(
        "Failed to load CT_Verifier.keras"
    )

    st.exception(e)

    st.stop()


try:

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error(
        "Failed to load pneumonia model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 11. VERIFY MODALITY MODEL
# ============================================================

if modality_model.input_shape != (
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
        modality_model.input_shape
    )

    st.stop()


if modality_model.output_shape != (
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
        modality_model.output_shape
    )

    st.stop()


# ============================================================
# 12. HELPER — COLOUR DETECTION
# ============================================================

def is_obvious_colour_image(
    image,
    saturation_threshold=18.0,
    colour_pixel_ratio=0.08
):

    """
    Detect strongly coloured images.

    IMPORTANT:
    We do not reject an image merely because its
    PIL mode is RGB.

    Many medical X-rays are stored as RGB files
    even though they are visually grayscale.
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
# 13. HELPER — EXACT MODALITY PREPROCESSING
# ============================================================

def preprocess_for_modality_model(
    image
):

    """
    EXACTLY MATCHES THE KAGGLE TEST.

    Training:
        ImageDataGenerator(rescale=1/255)

    Deployment:
        RGB
        -> resize 128x128
        -> float32
        -> /255.0
        -> batch dimension

    DO NOT add MobileNetV2 preprocess_input().
    """

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image_rgb = image.convert("RGB")

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized_image = image_rgb.resize(
        MODALITY_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # EXACT TRAINING SCALING
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    model_input = np.expand_dims(
        image_array,
        axis=0
    )

    return model_input


# ============================================================
# 14. HELPER — MODALITY PREDICTION
# ============================================================

def predict_modality(
    image
):

    model_input = (
        preprocess_for_modality_model(
            image
        )
    )

    prediction = modality_model(
        model_input,
        training=False
    ).numpy()[0]

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if prediction.shape != (3,):

        raise ValueError(
            "CT_Verifier produced an unexpected "
            f"output shape: {prediction.shape}"
        )

    # --------------------------------------------------------
    # Softmax safety
    #
    # The model already contains softmax.
    # This is only a safety check.
    # --------------------------------------------------------

    if (
        np.any(prediction < 0)
        or
        np.any(prediction > 1)
        or
        not np.isclose(
            np.sum(prediction),
            1.0,
            atol=1e-3
        )
    ):

        prediction = (
            tf.nn.softmax(
                prediction
            ).numpy()
        )

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = (
        MODALITY_CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = float(
        prediction[
            predicted_index
        ]
    )

    return (
        prediction,
        predicted_index,
        predicted_class,
        confidence,
        model_input
    )


# ============================================================
# 15. HELPER — PNEUMONIA PREPROCESSING
# ============================================================

def preprocess_for_pneumonia_model(
    image
):

    image_rgb = image.convert(
        "RGB"
    )

    resized_image = image_rgb.resize(
        PNEUMONIA_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    # Keep this ONLY because your previous
    # pneumonia pipeline used /255.0.
    image_array = (
        image_array / 255.0
    )

    model_input = np.expand_dims(
        image_array,
        axis=0
    )

    return model_input


# ============================================================
# 16. HELPER — PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    image
):

    model_input = (
        preprocess_for_pneumonia_model(
            image
        )
    )

    prediction = pneumonia_model(
        model_input,
        training=False
    ).numpy()

    prediction_values = np.squeeze(
        prediction
    )

    # --------------------------------------------------------
    # SINGLE OUTPUT
    # --------------------------------------------------------

    if prediction_values.size == 1:

        pneumonia_probability = float(
            prediction_values
        )

        # Handle sigmoid-logit safety
        if (
            pneumonia_probability < 0
            or
            pneumonia_probability > 1
        ):

            pneumonia_probability = float(
                tf.nn.sigmoid(
                    pneumonia_probability
                ).numpy()
            )

        normal_probability = (
            1.0 -
            pneumonia_probability
        )

    # --------------------------------------------------------
    # TWO OUTPUTS
    # --------------------------------------------------------

    elif prediction_values.size == 2:

        probabilities = (
            prediction_values.astype(
                np.float64
            )
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

        # Assumed:
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

        raise ValueError(
            "Unsupported pneumonia model output."
        )

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
        diagnosis_confidence,
        normal_probability,
        pneumonia_probability
    )


# ============================================================
# 17. PDF REPORT
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
# 18. HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
### Image Analysis Pipeline

1. Colour-image rejection
2. CT / MRI / Chest X-ray classification
3. Pneumonia detection only for Chest X-ray
4. PDF report generation
"""
)


# ============================================================
# 19. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        "Modality Model"
    )

    st.code(
        "CT_Verifier.keras"
    )

    st.write(
        "Pneumonia Model"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )

    st.divider()

    st.write(
        "Modality input:"
    )

    st.code(
        "(128, 128, 3)"
    )

    st.write(
        "Modality preprocessing:"
    )

    st.code(
        "RGB → resize → float32 → /255.0"
    )

    st.divider()

    st.header(
        "Recent Scans"
    )

    if not st.session_state.history:

        st.write(
            "No scans yet."
        )

    else:

        for item in reversed(
            st.session_state.history[-10:]
        ):

            st.text(item)


# ============================================================
# 20. FILE UPLOADER
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
# 21. PROCESS IMAGE
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

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Original size:** "
                f"{image.width} × "
                f"{image.height}"
            )

        with col2:

            st.write(
                f"**Original mode:** "
                f"{image.mode}"
            )

        # ====================================================
        # ANALYZE
        # ====================================================

        analyze = st.button(
            "Analyze Image",
            type="primary"
        )

        if analyze:

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

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Mean Colour Difference",
                    f"{mean_colour_difference:.2f}"
                )

            with col2:

                st.metric(
                    "Coloured Pixel Ratio",
                    f"{coloured_ratio * 100:.2f}%"
                )

            if colour_detected:

                st.error(
                    "Colour image detected."
                )

                st.warning(
                    "Please upload a grayscale "
                    "medical image."
                )

                st.session_state.history.append(
                    f"Rejected - Colour - "
                    f"{uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality=(
                        "Rejected - Colour Image"
                    ),
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
                "Image passes grayscale check."
            )

            # =================================================
            # STEP 1 — MODALITY
            # =================================================

            st.subheader(
                "Step 1 — Modality Classification"
            )

            # =================================================
            # IMPORTANT:
            #
            # This function is deliberately identical
            # to the Kaggle diagnostic test that correctly
            # classified your 14332 X-ray training images.
            # =================================================

            (
                modality_probabilities,
                predicted_index,
                predicted_modality,
                modality_confidence,
                modality_input
            ) = predict_modality(
                image
            )

            # =================================================
            # DEBUG INFORMATION
            #
            # Keep this visible temporarily.
            # It allows us to determine whether Streamlit
            # and Kaggle are feeding the model the same data.
            # =================================================

            with st.expander(
                "Technical Modality Diagnostics",
                expanded=True
            ):

                st.write(
                    "**Model input shape:**",
                    modality_input.shape
                )

                st.write(
                    "**Input minimum:**",
                    float(
                        modality_input.min()
                    )
                )

                st.write(
                    "**Input maximum:**",
                    float(
                        modality_input.max()
                    )
                )

                st.write(
                    "**Input mean:**",
                    float(
                        modality_input.mean()
                    )
                )

                st.write(
                    "**Model output shape:**",
                    modality_model.output_shape
                )

                st.write(
                    "**Raw model probabilities:**",
                    modality_probabilities
                )

            # =================================================
            # PROBABILITY DISPLAY
            # =================================================

            st.write(
                "### Modality Probabilities"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Chest X-ray",
                    (
                        f"{modality_probabilities[0] * 100:.4f}%"
                    )
                )

            with col2:

                st.metric(
                    "CT",
                    (
                        f"{modality_probabilities[1] * 100:.4f}%"
                    )
                )

            with col3:

                st.metric(
                    "MRI",
                    (
                        f"{modality_probabilities[2] * 100:.4f}%"
                    )
                )

            # =================================================
            # MODALITY DECISION
            # =================================================

            if (
                modality_confidence
                < MODALITY_MIN_CONFIDENCE
            ):

                st.error(
                    "Unsupported medical image detected."
                )

                st.warning(
                    "The modality classifier is "
                    "not sufficiently confident."
                )

                st.session_state.history.append(
                    f"Rejected - Low confidence - "
                    f"{uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality=(
                        "Unsupported / Uncertain"
                    ),
                    modality_confidence=(
                        modality_confidence
                    )
                )

                st.download_button(
                    "Download Rejection Report",
                    data=pdf_data,
                    file_name=(
                        "Unsupported_Image_Report.pdf"
                    ),
                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # CT
            # =================================================

            if predicted_index == 1:

                st.error(
                    "CT scan detected."
                )

                st.write(
                    f"CT confidence: "
                    f"{modality_probabilities[1] * 100:.2f}%"
                )

                st.info(
                    "This system sends only Chest "
                    "X-ray images to the pneumonia "
                    "classifier."
                )

                st.session_state.history.append(
                    f"CT - {uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="CT Scan",
                    modality_confidence=(
                        modality_probabilities[1]
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
                    "MRI image detected."
                )

                st.write(
                    f"MRI confidence: "
                    f"{modality_probabilities[2] * 100:.2f}%"
                )

                st.info(
                    "This system sends only Chest "
                    "X-ray images to the pneumonia "
                    "classifier."
                )

                st.session_state.history.append(
                    f"MRI - {uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="MRI",
                    modality_confidence=(
                        modality_probabilities[2]
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

            if predicted_index != 0:

                st.error(
                    "Unsupported medical image detected."
                )

                st.stop()

            st.success(
                "Chest X-ray detected."
            )

            st.write(
                f"Chest X-ray confidence: "
                f"{modality_probabilities[0] * 100:.2f}%"
            )

            # =================================================
            # STEP 2 — PNEUMONIA
            # =================================================

            st.subheader(
                "Step 2 — Pneumonia Detection"
            )

            (
                diagnosis,
                diagnosis_confidence,
                normal_probability,
                pneumonia_probability
            ) = predict_pneumonia(
                image
            )

            # =================================================
            # DIAGNOSIS
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
                    (
                        f"{modality_probabilities[0] * 100:.2f}%"
                    )
                )

            with col2:

                st.metric(
                    "Normal Probability",
                    (
                        f"{normal_probability * 100:.2f}%"
                    )
                )

            with col3:

                st.metric(
                    "Pneumonia Probability",
                    (
                        f"{pneumonia_probability * 100:.2f}%"
                    )
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
                "**Image Modality:** "
                "Chest X-ray"
            )

            st.write(
                f"**X-ray Confidence:** "
                f"{modality_probabilities[0] * 100:.2f}%"
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
                modality_confidence=(
                    modality_probabilities[0]
                ),
                diagnosis=diagnosis,
                diagnosis_confidence=(
                    diagnosis_confidence
                )
            )

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
