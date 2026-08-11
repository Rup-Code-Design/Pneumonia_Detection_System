import os
import io

import cv2
import numpy as np
import streamlit as st

from PIL import Image
from fpdf import FPDF

from model_builder import build_model
from xray_model_builder import build_xray_classifier


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

XRAY_MODEL_PATH = "best_xray_verifier.weights.h5"

PNEUMONIA_MODEL_PATH = "best_xception_pneumonia_model.keras"


# ============================================================
# MODEL SETTINGS
# ============================================================

XRAY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# X-RAY VERIFIER CLASS MAPPING
# ============================================================

XRAY_CLASS_MAP = {
    0: "NON-XRAY",
    1: "X-RAY"
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    if not os.path.isfile(XRAY_MODEL_PATH):

        raise FileNotFoundError(
            f"X-ray verifier weights not found:\n"
            f"{XRAY_MODEL_PATH}"
        )

    model = build_xray_classifier(
        input_shape=(128, 128, 3)
    )

    model.load_weights(
        XRAY_MODEL_PATH
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

    model = build_model(
        input_shape=(224, 224, 3)
    )

    model.load_weights(
        PNEUMONIA_MODEL_PATH
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model = load_xray_model()

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🫁 Pneumonia Detection System")

st.markdown(
    """
    Upload an image to the system.

    **Step 1:** The system verifies whether the image is a chest X-ray.

    **Step 2:** If it is a chest X-ray, the pneumonia detection model
    analyzes it.

    **Step 3:** The system reports Normal or Pneumonia.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the System")

    st.write(
        "This application uses two deep-learning models:"
    )

    st.write(
        "1. X-ray / Non-X-ray verification"
    )

    st.write(
        "2. Pneumonia detection"
    )

    st.divider()

    st.header("Recent Scans")

    if len(st.session_state.history) == 0:

        st.write("No scans yet.")

    else:

        for item in reversed(
            st.session_state.history[-10:]
        ):

            st.text(item)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image."
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        file_bytes = uploaded_file.getvalue()

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                # ==================================================
                # STEP 1 — BASIC IMAGE VALIDATION
                # ==================================================

                image_array = np.array(image)

                if image_array is None:

                    st.error(
                        "Could not read the uploaded image."
                    )

                    st.stop()

                # ==================================================
                # STEP 2 — X-RAY VERIFICATION MODEL
                # ==================================================

                verifier_image = cv2.resize(
                    image_array,
                    XRAY_IMAGE_SIZE
                )

                verifier_image = (
                    verifier_image.astype(
                        np.float32
                    ) / 255.0
                )

                verifier_input = np.expand_dims(
                    verifier_image,
                    axis=0
                )

                verifier_prediction = (
                    xray_model.predict(
                        verifier_input,
                        verbose=0
                    )
                )

                verifier_prediction = np.asarray(
                    verifier_prediction
                )

                # --------------------------------------------------
                # CHECK OUTPUT
                # --------------------------------------------------

                if (
                    verifier_prediction.ndim != 2
                    or verifier_prediction.shape[1] != 2
                ):

                    st.error(
                        "X-ray verifier output is not "
                        "configured as a 2-class classifier."
                    )

                    st.write(
                        "Verifier output shape:",
                        verifier_prediction.shape
                    )

                    st.stop()

                # --------------------------------------------------
                # GET CLASS
                # --------------------------------------------------

                verifier_class_index = int(
                    np.argmax(
                        verifier_prediction[0]
                    )
                )

                verifier_confidence = float(
                    verifier_prediction[
                        0,
                        verifier_class_index
                    ]
                )

                verifier_result = XRAY_CLASS_MAP.get(
                    verifier_class_index,
                    "UNKNOWN"
                )

                # ==================================================
                # STEP 3 — REJECT NON-X-RAY
                # ==================================================

                if verifier_result == "NON-XRAY":

                    st.error(
                        "❌ This is not a Chest X-ray image."
                    )

                    st.write(
                        f"Verifier confidence: "
                        f"{verifier_confidence * 100:.2f}%"
                    )

                    st.warning(
                        "Please upload a valid chest X-ray image."
                    )

                    # History

                    history_entry = (
                        f"Rejected - "
                        f"{uploaded_file.name}"
                    )

                    if history_entry not in st.session_state.history:

                        st.session_state.history.append(
                            history_entry
                        )

                    st.stop()

                # ==================================================
                # STEP 4 — CHEST X-RAY CONFIRMED
                # ==================================================

                st.success(
                    "✅ Chest X-ray image detected."
                )

                st.write(
                    f"X-ray verification confidence: "
                    f"{verifier_confidence * 100:.2f}%"
                )

                # ==================================================
                # STEP 5 — PREPARE PNEUMONIA MODEL INPUT
                # ==================================================

                pneumonia_image = cv2.resize(
                    image_array,
                    PNEUMONIA_IMAGE_SIZE
                )

                pneumonia_image = (
                    pneumonia_image.astype(
                        np.float32
                    ) / 255.0
                )

                pneumonia_input = np.expand_dims(
                    pneumonia_image,
                    axis=0
                )

                # ==================================================
                # STEP 6 — PNEUMONIA PREDICTION
                # ==================================================

                prediction = pneumonia_model.predict(
                    pneumonia_input,
                    verbose=0
                )

                prediction = np.asarray(prediction)

                if prediction.ndim != 2 or prediction.shape[1] != 2:

                    st.error(
                        "Pneumonia model output is not "
                        "configured as a 2-class classifier."
                    )

                    st.write(
                        "Model output shape:",
                        prediction.shape
                    )

                    st.stop()

                normal_probability = float(prediction[0][0])
                pneumonia_probability = float(prediction[0][1])

                # --------------------------------------------------
                # SAFETY CLAMP
                # --------------------------------------------------

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

                # ==================================================
                # STEP 7 — CLASSIFICATION
                # ==================================================

                if pneumonia_probability >= 0.5:

                    diagnosis = "Pneumonia"

                    diagnosis_confidence = (
                        pneumonia_probability
                    )

                    st.error(
                        f"Diagnosis: {diagnosis}"
                    )

                    st.metric(
                        "Pneumonia Probability",
                        f"{pneumonia_probability * 100:.2f}%"
                    )

                else:

                    diagnosis = "Normal"

                    diagnosis_confidence = (
                        normal_probability
                    )

                    st.success(
                        f"Diagnosis: {diagnosis}"
                    )

                    st.metric(
                        "Normal Probability",
                        f"{normal_probability * 100:.2f}%"
                    )

                # ==================================================
                # STEP 8 — DISPLAY RESULTS
                # ==================================================

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "X-ray Confidence",
                        f"{verifier_confidence * 100:.2f}%"
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

                # ==================================================
                # STEP 9 — HISTORY
                # ==================================================

                history_entry = (
                    f"{diagnosis} - "
                    f"{uploaded_file.name}"
                )

                if history_entry not in st.session_state.history:

                    st.session_state.history.append(
                        history_entry
                    )

                # ==================================================
                # STEP 10 — PDF REPORT
                # ==================================================

                st.divider()

                st.subheader(
                    "Diagnostic Report"
                )

                st.write(
                    f"**File:** {uploaded_file.name}"
                )

                st.write(
                    f"**X-ray verification:** "
                    f"{verifier_result}"
                )

                st.write(
                    f"**Diagnosis:** {diagnosis}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{diagnosis_confidence * 100:.2f}%"
                )

                # --------------------------------------------------
                # Generate PDF
                # --------------------------------------------------

                clean_filename = (
                    uploaded_file.name
                    .encode(
                        "ascii",
                        "ignore"
                    )
                    .decode("ascii")
                )

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

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    45,
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

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    45,
                    10,
                    "X-ray Status:",
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
                    verifier_result,
                    ln=True
                )

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    45,
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

                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    45,
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

                pdf.ln(15)

                pdf.set_font(
                    "Arial",
                    "I",
                    10
                )

                pdf.multi_cell(
                    0,
                    7,
                    "Disclaimer: "
                    "This AI-generated result is intended "
                    "for research purposes only and does not "
                    "replace professional medical diagnosis."
                )

                # --------------------------------------------------
                # PDF OUTPUT
                # --------------------------------------------------

                pdf_output = pdf.output()

                st.download_button(
                    label="Download Diagnostic Report",
                    data=bytes(pdf_output),
                    file_name=(
                        f"Report_{clean_filename}.pdf"
                    ),
                    mime="application/pdf"
                )

    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)
