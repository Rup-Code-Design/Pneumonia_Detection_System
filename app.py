# ============================================================
# app.py
# ============================================================
#
# PIPELINE
#
# 1. Reject obvious colour images
#
# 2. CT_Verifier.keras
#       0 = CHEST_XRAY
#       1 = CT
#       2 = MRI
#
# 3. If CHEST_XRAY:
#       Run pneumonia model
#
#       -> Normal
#       -> Pneumonia
#
# 4. If CT:
#       CT Scan detected
#
# 5. If MRI:
#       MRI detected
#
# 6. Generate PDF report
#
# IMPORTANT:
#
# CT_Verifier.keras was trained using:
#
# ImageDataGenerator(
#     rescale=1.0 / 255.0
# )
#
# Therefore this app ALSO uses /255.0.
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
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
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

# MUST MATCH TRAINING CODE

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# 5. THRESHOLDS
# ============================================================

# IMPORTANT:
#
# We do NOT use a high X-ray threshold such as 0.80 or 0.90.
#
# The classifier must simply select CHEST_XRAY as its
# highest-probability class.
#
# This avoids unnecessarily rejecting valid X-rays.
#
CHEST_XRAY_THRESHOLD = 0.50

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

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# 10. VERIFY MODALITY MODEL
# ============================================================

if modality_model.input_shape != (
    None,
    128,
    128,
    3
):

    st.error(
        "CT_Verifier.keras has an unexpected input shape."
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
        "CT_Verifier.keras is not a 3-class classifier."
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
# 11. HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.markdown(
    """
### Supported Medical Images

1. **Chest X-ray** → Pneumonia / Normal
2. **CT Scan** → CT Scan detected
3. **MRI** → MRI detected

The pneumonia model is executed **only after a Chest X-ray
has been confirmed**.
"""
)


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.write("Modality classifier")

    st.code(
        "CT_Verifier.keras"
    )

    st.write("Pneumonia classifier")

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
        "pixel / 255.0"
    )

    st.divider()

    st.header("Recent Scans")

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
# 13. COLOUR IMAGE DETECTOR
# ============================================================

def is_obvious_colour_image(
    image,
    saturation_threshold=18.0,
    colour_pixel_ratio=0.08
):

    """
    Reject strongly coloured images.

    We DO NOT reject an image merely because its PIL mode
    is RGB.

    A grayscale X-ray can be stored as an RGB JPEG/PNG.

    Instead, we measure actual differences between RGB
    channels.
    """

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
        difference > saturation_threshold
    )

    coloured_ratio = float(
        np.mean(coloured_pixels)
    )

    colour_detected = (

        mean_difference >
        saturation_threshold

        and

        coloured_ratio >
        colour_pixel_ratio
    )

    return (
        colour_detected,
        mean_difference,
        coloured_ratio
    )


# ============================================================
# 14. PREPARE MODALITY IMAGE
# ============================================================

def prepare_modality_image(image):

    resized = image.resize(
        (128, 128),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    )

    # EXACTLY MATCH TRAINING
    array = array / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array

# ============================================================
# DEBUG MODALITY PREDICTION
# ============================================================

modality_input = prepare_modality_image(image_rgb)

raw_prediction = modality_model.predict(
    modality_input,
    verbose=0
)

raw_prediction = np.asarray(
    raw_prediction,
    dtype=np.float64
)

st.write("### DEBUG — CT_Verifier.keras")

st.write(
    "Model input shape:",
    modality_model.input_shape
)

st.write(
    "Model output shape:",
    modality_model.output_shape
)

st.write(
    "Raw model output:",
    raw_prediction
)

probabilities = raw_prediction[0]

st.write(
    "CHEST_XRAY:",
    f"{probabilities[0] * 100:.4f}%"
)

st.write(
    "CT:",
    f"{probabilities[1] * 100:.4f}%"
)

st.write(
    "MRI:",
    f"{probabilities[2] * 100:.4f}%"
)

predicted_index = int(
    np.argmax(probabilities)
)

st.write(
    "Predicted index:",
    predicted_index
)

st.write(
    "Predicted class:",
    MODALITY_CLASS_NAMES[predicted_index]
)

    # --------------------------------------------------------
    # Softmax safety
    # --------------------------------------------------------

    if (

        np.any(
            probabilities < 0
        )

        or

        np.any(
            probabilities > 1
        )

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

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = (
        MODALITY_CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    return (
        probabilities,
        predicted_index,
        predicted_class,
        confidence
    )


# ============================================================
# 16. PREPARE PNEUMONIA IMAGE
# ============================================================

def prepare_pneumonia_image(
    image
):

    resized = image.resize(
        PNEUMONIA_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    )

    # Keep this ONLY if the pneumonia model was also
    # trained with /255.0.

    array = array / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# 17. PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    image
):

    model_input = (
        prepare_pneumonia_image(
            image
        )
    )

    prediction = (
        pneumonia_model.predict(
            model_input,
            verbose=0
        )
    )

    values = np.squeeze(
        prediction
    )

    # --------------------------------------------------------
    # SINGLE OUTPUT
    # --------------------------------------------------------

    if values.size == 1:

        pneumonia_probability = float(
            values
        )

        # If output is a logit, sigmoid would be required.
        # For your existing model we assume probability.

        if not (
            0.0 <=
            pneumonia_probability <=
            1.0
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

    elif values.size == 2:

        probabilities = (
            values.astype(
                np.float64
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
                np.sum(probabilities),
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

        raise ValueError(
            "Unsupported pneumonia model output: "
            f"{values}"
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
        normal_probability,
        pneumonia_probability,
        diagnosis,
        diagnosis_confidence
    )


# ============================================================
# 18. CREATE PDF
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
        "intended for research purposes only and does "
        "not replace professional medical diagnosis."
    )

    return bytes(
        pdf.output()
    )


# ============================================================
# 19. FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


# ============================================================
# 20. PROCESS IMAGE
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

        st.write(
            f"**Image size:** "
            f"{image.width} × {image.height}"
        )

        st.write(
            f"**Original image mode:** "
            f"{image.mode}"
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            # =================================================
            # STEP 1
            # COLOUR CHECK
            # =================================================

            st.subheader(
                "Step 1 — Image Quality Check"
            )

            (
                colour_detected,
                mean_difference,
                coloured_ratio
            ) = is_obvious_colour_image(
                image_rgb
            )

            st.write(
                f"Mean RGB channel difference: "
                f"{mean_difference:.2f}"
            )

            st.write(
                f"Strongly coloured pixels: "
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
                    "Rejected — Colour image — "
                    f"{uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="Rejected — Colour Image",
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
                "Image passed grayscale check."
            )

            # =================================================
            # STEP 2
            # MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 2 — Modality Classification"
            )

            with st.spinner(
                "Classifying image modality..."
            ):

                (
                    probabilities,
                    predicted_index,
                    predicted_modality,
                    predicted_confidence
                ) = predict_modality(
                    image_rgb
                )

            # =================================================
            # DISPLAY RAW MODEL OUTPUT
            # =================================================

            st.write(
                "### Model Prediction"
            )

            st.write(
                f"CHEST_XRAY: "
                f"{probabilities[0] * 100:.2f}%"
            )

            st.write(
                f"CT: "
                f"{probabilities[1] * 100:.2f}%"
            )

            st.write(
                f"MRI: "
                f"{probabilities[2] * 100:.2f}%"
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
            # IMPORTANT DECISION
            # =================================================
            #
            # Do NOT require X-ray to exceed an arbitrary
            # high confidence.
            #
            # If X-ray is the highest class, it is accepted
            # provided it reaches 50%.
            #
            # =================================================

            is_chest_xray = (

                predicted_index == 0

                and

                probabilities[0]
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
                    "Pneumonia detection is not performed "
                    "because the uploaded image is classified "
                    "as a CT scan."
                )

                st.session_state.history.append(
                    "CT Scan — "
                    f"{uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="CT Scan",
                    modality_confidence=float(
                        probabilities[1]
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

                st.info(
                    "Pneumonia detection is not performed "
                    "because the uploaded image is classified "
                    "as an MRI image."
                )

                st.session_state.history.append(
                    "MRI — "
                    f"{uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="MRI",
                    modality_confidence=float(
                        probabilities[2]
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
            # LOW X-RAY CONFIDENCE
            # =================================================

            if not is_chest_xray:

                st.error(
                    "Chest X-ray could not be confirmed."
                )

                st.warning(
                    "The modality classifier did not "
                    "produce sufficient confidence for "
                    "Chest X-ray."
                )

                st.write(
                    f"Chest X-ray probability: "
                    f"{probabilities[0] * 100:.2f}%"
                )

                st.session_state.history.append(
                    "Rejected — Low X-ray confidence — "
                    f"{uploaded_file.name}"
                )

                pdf_data = create_pdf_report(
                    filename=uploaded_file.name,
                    modality="Unconfirmed Medical Image",
                    modality_confidence=(
                        predicted_confidence
                    )
                )

                st.download_button(
                    "Download Rejection Report",
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
                f"{probabilities[0] * 100:.2f}%"
            )

            # =================================================
            # STEP 3
            # PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 3 — Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing Chest X-ray..."
            ):

                (
                    normal_probability,
                    pneumonia_probability,
                    diagnosis,
                    diagnosis_confidence
                ) = predict_pneumonia(
                    image_rgb
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
                    f"{probabilities[0] * 100:.2f}%"
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
                f"{diagnosis} — "
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
                f"{probabilities[0] * 100:.2f}%"
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
                modality_confidence=float(
                    probabilities[0]
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
