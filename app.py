import os
import io
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

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
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

# X-ray verifier
XRAY_MODEL_PATH = "best_xray_verifier.weights.h5"

# Pneumonia model
PNEUMONIA_MODEL_PATH = "best_xception_model.keras"

# Probability threshold
XRAY_THRESHOLD = 0.50
PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_verifier():

    if not os.path.exists(XRAY_MODEL_PATH):
        raise FileNotFoundError(
            f"X-ray verifier weights not found: "
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

    if not os.path.exists(
        PNEUMONIA_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Pneumonia model not found: "
            f"{PNEUMONIA_MODEL_PATH}"
        )

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=1
    )

    model.load_weights(
        PNEUMONIA_MODEL_PATH
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model = load_xray_verifier()

    pneumonia_model = load_pneumonia_model()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        f"Model loading error:\n\n{e}"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫁 Pneumonia Detection System"
)

st.markdown(
    """
    Upload an image for automated analysis.

    **Stage 1:** Verify that the uploaded image is a Chest X-ray.

    **Stage 2:** If valid, analyze the Chest X-ray for pneumonia.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the System"
    )

    st.info(
        """
        This application uses a two-stage
        deep-learning pipeline:

        1. X-ray Verifier
        2. Pneumonia Detector
        """
    )

    st.write(
        "**Input size:** 128 × 128"
    )

    st.write(
        "**X-ray classes:**"
    )

    st.write(
        "- Chest X-ray"
    )

    st.write(
        "- Non-X-ray"
    )

    st.divider()

    st.header(
        "Recent Scans"
    )

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
# CHECK MODELS
# ============================================================

if not models_loaded:

    st.stop()


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
    ]
)


# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

def prepare_image(
    file_bytes,
    image_size
):

    nparr = np.frombuffer(
        file_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return None

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        image_size
    )

    img = img.astype(
        np.float32
    ) / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


# ============================================================
# PDF REPORT FUNCTION
# ============================================================

def create_pdf_report(
    filename,
    xray_result,
    xray_confidence,
    pneumonia_result=None,
    pneumonia_confidence=None
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
        190,
        12,
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

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    clean_filename = (
        str(filename)
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
        140,
        10,
        clean_filename,
        ln=True
    )

    # --------------------------------------------------------
    # X-RAY VERIFICATION
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        45,
        10,
        "Image Type:",
        ln=False
    )

    pdf.set_font(
        "Arial",
        "",
        12
    )

    pdf.cell(
        140,
        10,
        xray_result,
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
        "X-ray Confidence:",
        ln=False
    )

    pdf.set_font(
        "Arial",
        "",
        12
    )

    pdf.cell(
        140,
        10,
        f"{xray_confidence:.2f}%",
        ln=True
    )

    # --------------------------------------------------------
    # PNEUMONIA RESULT
    # --------------------------------------------------------

    if pneumonia_result is not None:

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
            140,
            10,
            pneumonia_result,
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
            140,
            10,
            f"{pneumonia_confidence:.2f}%",
            ln=True
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    pdf.ln(20)

    pdf.set_font(
        "Arial",
        "I",
        10
    )

    pdf.multi_cell(
        190,
        8,
        "Disclaimer: This report is generated by an "
        "artificial intelligence system and is intended "
        "for research and educational purposes only. "
        "It is not a substitute for professional medical "
        "diagnosis. Please consult a qualified radiologist "
        "or medical practitioner."
    )

    output = pdf.output()

    return bytes(output)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    try:

        original_image = Image.open(
            io.BytesIO(file_bytes)
        )

        original_image.load()

    except Exception:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.image(
            original_image,
            caption="Uploaded Image",
            use_container_width=True
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    with col2:

        st.write(
            "### Image Information"
        )

        st.write(
            f"**Filename:** "
            f"{uploaded_file.name}"
        )

        st.write(
            f"**Format:** "
            f"{original_image.format}"
        )

        st.write(
            f"**Size:** "
            f"{original_image.size[0]} × "
            f"{original_image.size[1]}"
        )

        analyze = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        # ----------------------------------------------------
        # STAGE 1 — X-RAY VERIFICATION
        # ----------------------------------------------------

        with st.spinner(
            "Stage 1: Verifying image type..."
        ):

            try:

                xray_input = prepare_image(
                    file_bytes,
                    IMG_SIZE
                )

                if xray_input is None:

                    st.error(
                        "Could not decode the uploaded image."
                    )

                    st.stop()


                # Predict

                xray_prediction = (
                    xray_model.predict(
                        xray_input,
                        verbose=0
                    )[0]
                )


                # ------------------------------------------------
                # IMPORTANT:
                # Class 0 = Chest_Xray
                # Class 1 = Non_Xray
                # ------------------------------------------------

                chest_xray_score = float(
                    xray_prediction[0]
                )

                non_xray_score = float(
                    xray_prediction[1]
                )


                if (
                    chest_xray_score
                    >= XRAY_THRESHOLD
                ):

                    xray_result = (
                        "Chest X-ray"
                    )

                    xray_confidence = (
                        chest_xray_score * 100
                    )

                    is_xray = True

                else:

                    xray_result = (
                        "Non-Chest X-ray"
                    )

                    xray_confidence = (
                        non_xray_score * 100
                    )

                    is_xray = False


            except Exception as e:

                st.error(
                    f"X-ray verification failed: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # SHOW X-RAY VERIFICATION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Stage 1 — X-ray Verification"
        )


        metric1, metric2 = st.columns(
            2
        )

        with metric1:

            st.metric(
                "Chest X-ray Probability",
                f"{chest_xray_score * 100:.2f}%"
            )

        with metric2:

            st.metric(
                "Non-X-ray Probability",
                f"{non_xray_score * 100:.2f}%"
            )


        # ====================================================
        # NON-X-RAY → REJECT
        # ====================================================

        if not is_xray:

            st.error(
                "Rejected: The uploaded image "
                "was classified as a non-Chest X-ray."
            )

            st.warning(
                "Please upload a valid Chest X-ray image."
            )


            # Save history

            entry = (
                f"Rejected — "
                f"{uploaded_file.name}"
            )

            if entry not in st.session_state.history:

                st.session_state.history.append(
                    entry
                )


            # Create rejection report

            pdf_data = create_pdf_report(
                filename=uploaded_file.name,
                xray_result="Non-Chest X-ray — Rejected",
                xray_confidence=xray_confidence
            )


            st.download_button(
                label="Download Rejection Report",
                data=pdf_data,
                file_name=(
                    f"Report_"
                    f"{uploaded_file.name}.pdf"
                ),
                mime="application/pdf"
            )


            st.stop()


        # ====================================================
        # CHEST X-RAY → CONTINUE
        # ====================================================

        st.success(
            "Valid Chest X-ray detected."
        )


        # ----------------------------------------------------
        # STAGE 2 — PNEUMONIA DETECTION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Stage 2 — Pneumonia Detection"
        )


        with st.spinner(
            "Analyzing Chest X-ray for pneumonia..."
        ):

            try:

                # --------------------------------------------
                # Prepare pneumonia input
                # --------------------------------------------

                pneumonia_input = prepare_image(
                    file_bytes,
                    (224, 224)
                )


                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                prediction = (
                    pneumonia_model.predict(
                        pneumonia_input,
                        verbose=0
                    )
                )


                # --------------------------------------------
                # Handle model output
                # --------------------------------------------

                prediction = np.asarray(
                    prediction
                )


                # Binary sigmoid output
                if (
                    prediction.ndim == 2
                    and prediction.shape[1] == 1
                ):

                    pneumonia_probability = float(
                        prediction[0][0]
                    )

                # Two-class softmax output
                elif (
                    prediction.ndim == 2
                    and prediction.shape[1] == 2
                ):

                    pneumonia_probability = float(
                        prediction[0][1]
                    )

                else:

                    raise ValueError(
                        "Unexpected pneumonia model "
                        f"output shape: "
                        f"{prediction.shape}"
                    )


                # --------------------------------------------
                # Diagnosis
                # --------------------------------------------

                if (
                    pneumonia_probability
                    >= PNEUMONIA_THRESHOLD
                ):

                    diagnosis = (
                        "Pneumonia"
                    )

                    diagnosis_confidence = (
                        pneumonia_probability
                        * 100
                    )

                else:

                    diagnosis = (
                        "Normal"
                    )

                    diagnosis_confidence = (
                        (1.0 - pneumonia_probability)
                        * 100
                    )


            except Exception as e:

                st.error(
                    f"Pneumonia analysis failed: {e}"
                )

                st.stop()


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Final Diagnosis"
        )


        result_col1, result_col2 = st.columns(
            2
        )


        with result_col1:

            if diagnosis == "Pneumonia":

                st.error(
                    f"Diagnosis: {diagnosis}"
                )

            else:

                st.success(
                    f"Diagnosis: {diagnosis}"
                )


        with result_col2:

            st.metric(
                "Confidence",
                f"{diagnosis_confidence:.2f}%"
            )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        st.write(
            f"**Pneumonia probability:** "
            f"{pneumonia_probability * 100:.2f}%"
        )

        st.progress(
            min(
                max(
                    pneumonia_probability,
                    0.0
                ),
                1.0
            )
        )


        # ----------------------------------------------------
        # Medical disclaimer
        # ----------------------------------------------------

        st.warning(
            "This AI prediction is for research and "
            "educational purposes. It is not a clinical "
            "diagnosis. A qualified medical professional "
            "should review the Chest X-ray."
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        entry = (
            f"{diagnosis} — "
            f"{uploaded_file.name}"
        )

        if entry not in st.session_state.history:

            st.session_state.history.append(
                entry
            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        pdf_data = create_pdf_report(

            filename=uploaded_file.name,

            xray_result="Chest X-ray",

            xray_confidence=xray_confidence,

            pneumonia_result=diagnosis,

            pneumonia_confidence=diagnosis_confidence
        )


        st.download_button(

            label="Download Diagnostic Report",

            data=pdf_data,

            file_name=(
                f"Report_"
                f"{uploaded_file.name}.pdf"
            ),

            mime="application/pdf",

            use_container_width=True
        )
