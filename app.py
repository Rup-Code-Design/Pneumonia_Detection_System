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
# IMPORTANT:
#
# CT_Verifier.keras was trained using:
#
#     ImageDataGenerator(
#         rescale=1.0 / 255.0
#     )
#
# Therefore this application uses:
#
#     image / 255.0
#
# NOT:
#
#     MobileNetV2.preprocess_input()
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
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
    page_title="Pneumonia AI Detection System",
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
# 5. CLASS NAMES
# ============================================================

# MUST MATCH TRAINING CODE

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# 6. THRESHOLDS
# ============================================================

# This is only a safety threshold.
#
# IMPORTANT:
# The predicted class must ALSO be CHEST_XRAY.
#
CHEST_XRAY_THRESHOLD = 0.50

PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# 7. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# 8. LOAD CT / MRI / X-RAY MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModality model not found:\n"
            f"{MODALITY_MODEL_PATH}\n\n"
            f"Place CT_Verifier.keras in the "
            f"same directory as app.py."
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
            f"\nPneumonia model not found:\n"
            f"{PNEUMONIA_MODEL_PATH}\n\n"
            f"Place "
            f"{PNEUMONIA_MODEL_PATH} "
            f"in the same directory as app.py."
        )

    model = tf.keras.models.load_model(
        PNEUMONIA_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# 10. LOAD BOTH MODELS
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
# 11. VERIFY MODALITY MODEL
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


# ============================================================
# 12. VERIFY INPUT SHAPE
# ============================================================

if modality_input_shape != (
    None,
    128,
    128,
    3
):

    st.warning(
        f"Unexpected modality model input shape: "
        f"{modality_input_shape}"
    )


# ============================================================
# 13. VERIFY OUTPUT SHAPE
# ============================================================

if (
    modality_output_shape is None
    or
    modality_output_shape[-1] != 3
):

    st.error(
        "CT_Verifier.keras is not a "
        "3-class modality classifier."
    )

    st.write(
        "Expected output shape:",
        "(None, 3)"
    )

    st.write(
        "Actual output shape:",
        modality_output_shape
    )

    st.stop()


# ============================================================
# 14. COLOUR IMAGE DETECTOR
# ============================================================

def is_obvious_colour_image(
    image,
    saturation_threshold=18.0,
    colour_pixel_ratio=0.08
):
    """
    Detect strongly coloured images.

    IMPORTANT:
    RGB file format does NOT automatically mean
    the image is a colour image.

    Many Chest X-rays are stored as RGB images
    containing essentially grayscale information.

    Therefore this function checks actual
    channel differences.
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
# 15. PDF REPORT FUNCTION
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
        str(filename)
        .encode(
            "ascii",
            "ignore"
        )
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
        "Disclaimer: This AI-generated result is "
        "intended for research purposes only and "
        "does not replace professional medical diagnosis."
    )

    return bytes(
        pdf.output()
    )


# ============================================================
# 16. HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
### Image Processing Pipeline

**Step 1:** Reject obvious colour images

**Step 2:** Identify modality

- Chest X-ray
- CT
- MRI

**Step 3:** Only confirmed Chest X-rays are passed
to the pneumonia classifier.

**Step 4:** Pneumonia model predicts:

- Normal
- Pneumonia
"""
)


# ============================================================
# 17. SIDEBAR
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
        "Modality Input"
    )

    st.code(
        "(128, 128, 3)"
    )

    st.write(
        "Modality Preprocessing"
    )

    st.code(
        "image / 255.0"
    )

    st.write(
        "Pneumonia Model"
    )

    st.code(
        "best_xception_pneumonia_model.keras"
    )

    st.divider()

    st.write(
        "Modality classes"
    )

    st.code(
        "0 = CHEST_XRAY\n"
        "1 = CT\n"
        "2 = MRI"
    )

    st.divider()

    st.write(
        "Chest X-ray threshold:"
    )

    st.write(
        f"{CHEST_XRAY_THRESHOLD * 100:.0f}%"
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
# 18. FILE UPLOADER
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
# 19. PROCESS IMAGE
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
        # DISPLAY ORIGINAL IMAGE
        # ====================================================

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image_rgb,
            caption=uploaded_file.name,
            use_container_width=True
        )

        # ====================================================
        # IMAGE INFORMATION
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Image size:** "
                f"{image.width} × {image.height}"
            )

        with col2:

            st.write(
                f"**Image mode:** "
                f"{image.mode}"
            )

        # ====================================================
        # ANALYZE BUTTON
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
                "Step 0 — Colour Image Check"
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
                    "medical image."
                )

                st.session_state.history.append(
                    "Rejected - Colour image - "
                    + uploaded_file.name
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
            # RESIZE TO EXACT TRAINING SIZE
            # =================================================

            modality_image = image_rgb.resize(
                MODALITY_IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            # =================================================
            # CONVERT TO FLOAT32
            # =================================================

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
            # APPLICATION:
            #
            # image / 255.0
            #
            # EXACT MATCH.
            # =================================================

            modality_array = (
                modality_array / 255.0
            )

            # =================================================
            # ADD BATCH DIMENSION
            # =================================================

            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )

            # =================================================
            # DEBUG INFORMATION
            # =================================================

            with st.expander(
                "Model preprocessing information"
            ):

                st.write(
                    "Model input shape:",
                    modality_model.input_shape
                )

                st.write(
                    "Model output shape:",
                    modality_model.output_shape
                )

                st.write(
                    "Application preprocessing:",
                    "image / 255.0"
                )

                st.write(
                    "Input minimum:",
                    float(
                        np.min(
                            modality_input
                        )
                    )
                )

                st.write(
                    "Input maximum:",
                    float(
                        np.max(
                            modality_input
                        )
                    )
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
                    "CT_Verifier.keras must "
                    "output exactly 3 classes."
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

            # =================================================
            # CHECK WHETHER OUTPUT IS VALID
            #
            # Your model uses softmax, so normally:
            #
            # [0,1] and sum = 1
            #
            # This section is only a safety mechanism.
            # =================================================

            probability_sum = float(
                np.sum(
                    modality_probabilities
                )
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
            # EXTRACT CLASS PROBABILITIES
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

            st.write(
                "**Modality prediction probabilities:**"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Chest X-ray",
                    f"{xray_probability * 100:.2f}%"
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
            # SHOW PREDICTED CLASS
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
            # DECISION LOGIC
            # =================================================
            #
            # Chest X-ray accepted only if:
            #
            # condition 1:
            # X-ray has highest probability
            #
            # condition 2:
            # X-ray probability >= threshold
            #
            # =================================================

            is_chest_xray = (
                predicted_index == 0
                and
                xray_probability
                >= CHEST_XRAY_THRESHOLD
            )

            # =================================================
            # CT
            # =================================================

            if predicted_index == 1:

                st.error(
                    "CT Scan detected."
                )

                st.info(
                    "This application accepts "
                    "Chest X-ray images for "
                    "pneumonia detection."
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
                    )
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

            if predicted_index == 2:

                st.error(
                    "MRI image detected."
                )

                st.info(
                    "This application accepts "
                    "Chest X-ray images for "
                    "pneumonia detection."
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
                    )
                )

                st.download_button(
                    label="Download MRI Report",
                    data=pdf_data,
                    file_name="MRI_Report.pdf",
                    mime="application/pdf"
                )

                st.stop()

            # =================================================
            # LOW-CONFIDENCE / UNCONFIRMED
            # =================================================

            if not is_chest_xray:

                st.error(
                    "Chest X-ray could not be confirmed."
                )

                st.warning(
                    "The modality classifier "
                    "did not produce a sufficiently "
                    "confident Chest X-ray prediction."
                )

                st.session_state.history.append(
                    "Rejected - Low X-ray confidence - "
                    + uploaded_file.name
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
                    label=(
                        "Download Rejection Report"
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
            # RESIZE
            # =================================================

            pneumonia_image = (
                image_rgb.resize(
                    PNEUMONIA_IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )
            )

            pneumonia_array = np.asarray(
                pneumonia_image,
                dtype=np.float32
            )

            # =================================================
            # IMPORTANT
            #
            # Keep /255.0 ONLY if your pneumonia
            # model was trained with /255.0.
            # =================================================

            pneumonia_array = (
                pneumonia_array / 255.0
            )

            pneumonia_input = (
                np.expand_dims(
                    pneumonia_array,
                    axis=0
                )
            )

            # =================================================
            # PNEUMONIA PREDICTION
            # =================================================

            with st.spinner(
                "Analyzing Chest X-ray..."
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
            # HANDLE OUTPUT
            # =================================================

            normal_probability = None

            pneumonia_probability = None

            # =================================================
            # SINGLE OUTPUT
            #
            # Assumption:
            #
            # output = probability of Pneumonia
            #
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
            #
            # Expected:
            #
            # 0 = Normal
            # 1 = Pneumonia
            #
            # =================================================

            elif prediction_values.size == 2:

                probabilities = (
                    prediction_values
                    .astype(np.float64)
                )

                probability_sum = float(
                    np.sum(
                        probabilities
                    )
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
                        probability_sum,
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

                normal_probability = float(
                    probabilities[0]
                )

                pneumonia_probability = float(
                    probabilities[1]
                )

            # =================================================
            # UNSUPPORTED OUTPUT
            # =================================================

            else:

                st.error(
                    "Unsupported pneumonia "
                    "model output."
                )

                st.write(
                    "Model output:",
                    prediction_values
                )

                st.stop()

            # =================================================
            # CLAMP
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

            st.session_state.history.append(
                f"{diagnosis} - "
                + uploaded_file.name
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
            # DOWNLOAD PDF
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

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while "
            "processing the image."
        )

        st.exception(e)
