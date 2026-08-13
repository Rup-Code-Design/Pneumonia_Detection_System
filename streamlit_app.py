# ============================================================
# STREAMLIT APP
# PNEUMONIA DETECTION SYSTEM
#
# PIPELINE:
#
# 1. Upload image
# 2. Reject color images
# 3. Verify Chest X-ray using best_xray_verifier.keras
# 4. If X-ray -> Pneumonia detection
# 5. Display Normal / Pneumonia
#
# NO MODALITY CLASSIFIER IS REQUIRED.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import io
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

from PIL import Image
from fpdf import FPDF


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================
#
# This is important for Streamlit Cloud.
#
# Instead of:
#
#     "best_xray_verifier.keras"
#
# use:
#
#     os.path.join(BASE_DIR, "best_xray_verifier.keras")
#
# Therefore the app always looks in the same directory
# where streamlit_app.py is located.
#
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL FILES
# ============================================================

# Preferred X-ray verifier:
# Complete Keras model.
XRAY_MODEL_KERAS = os.path.join(
    BASE_DIR,
    "best_xray_verifier.keras"
)


# Fallback X-ray verifier weights.
XRAY_MODEL_WEIGHTS = os.path.join(
    BASE_DIR,
    "best_xray_verifier.weights.h5"
)


# Complete pneumonia model.
PNEUMONIA_MODEL_KERAS = os.path.join(
    BASE_DIR,
    "best_exception_pneumonia_model.keras"
)


# ============================================================
# MODEL INPUT SIZES
# ============================================================

XRAY_IMAGE_SIZE = (
    128,
    128
)


PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# COLOR IMAGE SETTINGS
# ============================================================

COLOR_TOLERANCE = 5.0


# ============================================================
# X-RAY VERIFIER SETTINGS
# ============================================================

XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# CLASS MAPPINGS
# ============================================================

# X-ray verifier:
#
# 0 = X-RAY
# 1 = NON-XRAY
#
# IMPORTANT:
# This must match your X-ray verifier training code.
#
XRAY_CLASS_MAP = {
    0: "X-RAY",
    1: "NON-XRAY"
}


# Pneumonia model:
#
# 0 = Normal
# 1 = Pneumonia
#
# IMPORTANT:
# This must match your pneumonia training code.
#
PNEUMONIA_CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# HELPER FUNCTION:
# FILE STATUS
# ============================================================

def get_file_status():

    return {
        "xray_keras": os.path.isfile(
            XRAY_MODEL_KERAS
        ),

        "xray_weights": os.path.isfile(
            XRAY_MODEL_WEIGHTS
        ),

        "pneumonia_keras": os.path.isfile(
            PNEUMONIA_MODEL_KERAS
        )
    }


# ============================================================
# HELPER FUNCTION:
# COLOR IMAGE DETECTION
# ============================================================

def calculate_color_difference(
    image_array
):

    """
    Calculate the average difference between
    RGB channels.

    A grayscale image has approximately:

        R ≈ G ≈ B

    A true color image has larger differences.
    """

    rgb_image = image_array.astype(
        np.float32
    )

    red = rgb_image[:, :, 0]

    green = rgb_image[:, :, 1]

    blue = rgb_image[:, :, 2]


    rg_difference = np.mean(
        np.abs(
            red - green
        )
    )

    gb_difference = np.mean(
        np.abs(
            green - blue
        )
    )

    rb_difference = np.mean(
        np.abs(
            red - blue
        )
    )


    average_difference = (
        rg_difference
        + gb_difference
        + rb_difference
    ) / 3.0


    return float(
        average_difference
    )


# ============================================================
# HELPER FUNCTION:
# CHECK COLOR IMAGE
# ============================================================

def is_color_image(
    image_array,
    tolerance=COLOR_TOLERANCE
):

    difference = calculate_color_difference(
        image_array
    )

    return (
        difference > tolerance
    ), difference


# ============================================================
# LOAD X-RAY MODEL
# ============================================================

@st.cache_resource
def load_xray_model():

    # --------------------------------------------------------
    # OPTION 1:
    # Load complete .keras model
    # --------------------------------------------------------

    if os.path.isfile(
        XRAY_MODEL_KERAS
    ):

        try:

            model = tf.keras.models.load_model(
                XRAY_MODEL_KERAS,
                compile=False
            )

            return model, "keras"

        except Exception as e:

            raise RuntimeError(
                "The file "
                f"{XRAY_MODEL_KERAS} "
                "exists but could not be loaded.\n\n"
                "This usually means the .keras model is "
                "corrupted or incompatible with the installed "
                "TensorFlow/Keras version.\n\n"
                f"Original error: {e}"
            )


    # --------------------------------------------------------
    # OPTION 2:
    # Fallback to weights
    # --------------------------------------------------------

    if os.path.isfile(
        XRAY_MODEL_WEIGHTS
    ):

        try:

            # Import only when needed.
            from xray_model_builder import (
                build_xray_classifier
            )


            model = build_xray_classifier(
                input_shape=(
                    128,
                    128,
                    3
                )
            )


            model.load_weights(
                XRAY_MODEL_WEIGHTS
            )


            return model, "weights"


        except Exception as e:

            raise RuntimeError(
                "X-ray verifier weights were found, "
                "but they could not be loaded.\n\n"
                "The architecture generated by "
                "xray_model_builder.py must exactly match "
                "the architecture used during training.\n\n"
                f"Weights file: "
                f"{XRAY_MODEL_WEIGHTS}\n\n"
                f"Original error: {e}"
            )


    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    raise FileNotFoundError(
        "X-ray verifier model was not found.\n\n"
        "Expected one of:\n"
        f"1. {XRAY_MODEL_KERAS}\n"
        f"2. {XRAY_MODEL_WEIGHTS}\n\n"
        "Your GitHub repository should contain at least "
        "one of these files."
    )


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.isfile(
        PNEUMONIA_MODEL_KERAS
    ):

        raise FileNotFoundError(
            "Pneumonia model was not found.\n\n"
            f"Expected location:\n"
            f"{PNEUMONIA_MODEL_KERAS}\n\n"
            "Make sure "
            "'best_exception_pneumonia_model.keras' "
            "is committed to the same GitHub repository "
            "as streamlit_app.py."
        )


    try:

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_KERAS,
            compile=False
        )

        return model


    except Exception as e:

        raise RuntimeError(
            "The pneumonia model file exists, "
            "but it could not be loaded.\n\n"
            f"File:\n"
            f"{PNEUMONIA_MODEL_KERAS}\n\n"
            f"Original error: {e}"
        )


# ============================================================
# MODEL OUTPUT PROCESSING
# ============================================================

def convert_to_probabilities(
    prediction
):

    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # Flatten batch output
    # --------------------------------------------------------

    if prediction.ndim == 2:

        scores = prediction[0]

    elif prediction.ndim == 1:

        scores = prediction

    else:

        raise ValueError(
            "Unexpected model output shape: "
            f"{prediction.shape}"
        )


    scores = scores.astype(
        np.float64
    )


    # --------------------------------------------------------
    # Case 1:
    # Already probabilities
    # --------------------------------------------------------

    if (
        np.all(scores >= 0.0)
        and
        np.all(scores <= 1.0)
        and
        np.isclose(
            np.sum(scores),
            1.0,
            atol=1e-3
        )
    ):

        return scores


    # --------------------------------------------------------
    # Case 2:
    # Logits
    # --------------------------------------------------------

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# X-RAY PREDICTION
# ============================================================

def predict_xray(
    model,
    image_array
):

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        image_array,
        XRAY_IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    resized = (
        resized.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    model_input = np.expand_dims(
        resized,
        axis=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        model_input,
        verbose=0
    )


    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # Binary classifier expected
    # --------------------------------------------------------

    if (
        prediction.ndim != 2
        or
        prediction.shape[1] != 2
    ):

        raise ValueError(
            "The X-ray verifier must output "
            "2 classes.\n\n"
            f"Received output shape: "
            f"{prediction.shape}"
        )


    probabilities = convert_to_probabilities(
        prediction
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    result = XRAY_CLASS_MAP.get(
        predicted_index,
        "UNKNOWN"
    )


    return (
        result,
        confidence,
        probabilities
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    model,
    image_array
):

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        image_array,
        PNEUMONIA_IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    resized = (
        resized.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    model_input = np.expand_dims(
        resized,
        axis=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        model_input,
        verbose=0
    )


    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # Determine output format
    # --------------------------------------------------------

    if prediction.ndim == 2:

        output_units = prediction.shape[1]

    elif prediction.ndim == 1:

        output_units = prediction.shape[0]

    else:

        raise ValueError(
            "Unexpected pneumonia model "
            f"output shape: {prediction.shape}"
        )


    # --------------------------------------------------------
    # Two-class model
    # --------------------------------------------------------

    if output_units == 2:

        probabilities = (
            convert_to_probabilities(
                prediction
            )
        )

        normal_probability = float(
            probabilities[0]
        )

        pneumonia_probability = float(
            probabilities[1]
        )


        if (
            pneumonia_probability
            >= normal_probability
        ):

            diagnosis = "Pneumonia"

            confidence = (
                pneumonia_probability
            )

        else:

            diagnosis = "Normal"

            confidence = (
                normal_probability
            )


        return (
            diagnosis,
            confidence,
            normal_probability,
            pneumonia_probability
        )


    # --------------------------------------------------------
    # One-output sigmoid model
    # --------------------------------------------------------

    if output_units == 1:

        raw_value = float(
            prediction.reshape(-1)[0]
        )


        # If the output is already between 0 and 1,
        # treat it as probability.
        if (
            0.0
            <= raw_value
            <= 1.0
        ):

            pneumonia_probability = (
                raw_value
            )

        else:

            pneumonia_probability = float(
                tf.sigmoid(
                    raw_value
                ).numpy()
            )


        normal_probability = (
            1.0
            - pneumonia_probability
        )


        if (
            pneumonia_probability
            >= 0.5
        ):

            diagnosis = "Pneumonia"

            confidence = (
                pneumonia_probability
            )

        else:

            diagnosis = "Normal"

            confidence = (
                normal_probability
            )


        return (
            diagnosis,
            confidence,
            normal_probability,
            pneumonia_probability
        )


    raise ValueError(
        "Unsupported pneumonia model output. "
        "Expected either 1 output neuron or "
        "2 output neurons.\n\n"
        f"Received shape: {prediction.shape}"
    )


# ============================================================
# HISTORY FUNCTION
# ============================================================

def add_history(
    text
):

    if text not in st.session_state.history:

        st.session_state.history.append(
            text
        )


# ============================================================
# CREATE PDF
# ============================================================

def create_pdf(
    filename,
    diagnosis,
    diagnosis_confidence,
    xray_confidence,
    normal_probability,
    pneumonia_probability
):

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
    # FILE NAME
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
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


    # --------------------------------------------------------
    # MODALITY
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
        10,
        "Image Modality:",
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
        "Chest X-ray",
        ln=True
    )


    # --------------------------------------------------------
    # X-RAY CONFIDENCE
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
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
        0,
        10,
        f"{xray_confidence * 100:.2f}%",
        ln=True
    )


    # --------------------------------------------------------
    # NORMAL PROBABILITY
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
        10,
        "Normal Probability:",
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
        f"{normal_probability * 100:.2f}%",
        ln=True
    )


    # --------------------------------------------------------
    # PNEUMONIA PROBABILITY
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
        10,
        "Pneumonia Probability:",
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
        f"{pneumonia_probability * 100:.2f}%",
        ln=True
    )


    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
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


    # --------------------------------------------------------
    # DIAGNOSIS CONFIDENCE
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        55,
        10,
        "Diagnosis Confidence:",
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


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    pdf.set_font(
        "Arial",
        "I",
        10
    )

    pdf.multi_cell(
        0,
        7,
        "Disclaimer: This AI-generated result "
        "is intended for research purposes only "
        "and does not replace professional "
        "medical diagnosis."
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    try:

        output = pdf.output(
            dest="S"
        )

        if isinstance(
            output,
            str
        ):

            return output.encode(
                "latin-1"
            )

        return bytes(
            output
        )

    except TypeError:

        output = pdf.output()

        if isinstance(
            output,
            bytes
        ):

            return output

        if isinstance(
            output,
            bytearray
        ):

            return bytes(
                output
            )

        return str(
            output
        ).encode(
            "latin-1"
        )


# ============================================================
# CHECK MODEL FILES
# ============================================================

file_status = get_file_status()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model, xray_model_type = (
        load_xray_model()
    )

    pneumonia_model = (
        load_pneumonia_model()
    )


except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)


    st.markdown(
        "### Model files detected"
    )


    if file_status["xray_keras"]:

        st.success(
            "✓ best_xray_verifier.keras"
        )

    else:

        st.error(
            "✗ best_xray_verifier.keras"
        )


    if file_status["xray_weights"]:

        st.success(
            "✓ best_xray_verifier.weights.h5"
        )

    else:

        st.warning(
            "✗ best_xray_verifier.weights.h5"
        )


    if file_status["pneumonia_keras"]:

        st.success(
            "✓ best_exception_pneumonia_model.keras"
        )

    else:

        st.error(
            "✗ best_exception_pneumonia_model.keras"
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
### AI Chest X-ray Analysis

The system performs the following sequence:

**1. Color Image Rejection**  
Color images are rejected before model inference.

**2. Chest X-ray Verification**  
The X-ray verifier determines whether the uploaded image
is a Chest X-ray.

**3. Pneumonia Detection**  
Only a verified Chest X-ray is passed to the pneumonia
classification model.

**4. Final Result**  
The system reports **Normal** or **Pneumonia**.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )


    st.write(
        "X-ray verifier:"
    )

    if xray_model_type == "keras":

        st.success(
            "best_xray_verifier.keras"
        )

    else:

        st.success(
            "best_xray_verifier.weights.h5"
        )


    st.write(
        "Pneumonia model:"
    )

    st.success(
        "best_exception_pneumonia_model.keras"
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

            st.text(
                item
            )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload a grayscale chest X-ray image. "
        "Color images are automatically rejected."
    )
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        file_bytes = (
            uploaded_file.getvalue()
        )


        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )


        # ----------------------------------------------------
        # Convert to RGB only for processing
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )


        image_array = np.array(
            image
        )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if (
            image_array is None
            or
            image_array.size == 0
        ):

            st.error(
                "Could not read the uploaded image."
            )

            st.stop()


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

        analyze = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        )


        if analyze:

            # =================================================
            # STEP 1
            # COLOR VALIDATION
            # =================================================

            st.subheader(
                "Step 1 — Image Validation"
            )


            is_color, color_difference = (
                is_color_image(
                    image_array
                )
            )


            st.write(
                "RGB channel difference: "
                f"{color_difference:.2f}"
            )


            # -------------------------------------------------
            # REJECT COLOR IMAGE
            # -------------------------------------------------

            if is_color:

                st.error(
                    "This is not a Chest X-ray image."
                )

                st.warning(
                    "Color images are not accepted. "
                    "Please upload a grayscale chest X-ray."
                )


                add_history(
                    "Rejected - Color image - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            # -------------------------------------------------
            # GRAYSCALE CONFIRMED
            # -------------------------------------------------

            st.success(
                "Grayscale image detected."
            )


            # =================================================
            # STEP 2
            # X-RAY VERIFICATION
            # =================================================

            st.subheader(
                "Step 2 — Chest X-ray Verification"
            )


            with st.spinner(
                "Verifying whether the image is a Chest X-ray..."
            ):

                (
                    xray_result,
                    xray_confidence,
                    xray_probabilities
                ) = predict_xray(
                    xray_model,
                    image_array
                )


            # -------------------------------------------------
            # X-RAY RESULT
            # -------------------------------------------------

            xray_probability = float(
                xray_probabilities[0]
            )


            non_xray_probability = float(
                xray_probabilities[1]
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Chest X-ray Probability",
                    f"{xray_probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Non-X-ray Probability",
                    f"{non_xray_probability * 100:.2f}%"
                )


            # =================================================
            # REJECT NON-XRAY
            # =================================================

            if (
                xray_result != "X-RAY"
                or
                xray_confidence
                < XRAY_CONFIDENCE_THRESHOLD
            ):

                st.error(
                    "This is not a Chest X-ray image."
                )

                st.warning(
                    "The X-ray verifier rejected this image."
                )


                add_history(
                    "Rejected - Non-X-ray - "
                    f"{uploaded_file.name}"
                )


                st.stop()


            # =================================================
            # X-RAY CONFIRMED
            # =================================================

            st.success(
                "Chest X-ray verified."
            )


            st.write(
                "X-ray verification confidence: "
                f"{xray_confidence * 100:.2f}%"
            )


            # =================================================
            # STEP 3
            # PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 3 — Pneumonia Detection"
            )


            with st.spinner(
                "Analyzing the Chest X-ray..."
            ):

                (
                    diagnosis,
                    diagnosis_confidence,
                    normal_probability,
                    pneumonia_probability
                ) = predict_pneumonia(
                    pneumonia_model,
                    image_array
                )


            # =================================================
            # FINAL DIAGNOSIS
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
            # PROBABILITIES
            # =================================================

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Normal",
                    f"{normal_probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Pneumonia",
                    f"{pneumonia_probability * 100:.2f}%"
                )


            st.write(
                f"**Final Diagnosis:** {diagnosis}"
            )


            st.write(
                "**Diagnosis Confidence:** "
                f"{diagnosis_confidence * 100:.2f}%"
            )


            # =================================================
            # HISTORY
            # =================================================

            add_history(
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
                f"**File:** {uploaded_file.name}"
            )


            st.write(
                "**Image modality:** Chest X-ray"
            )


            st.write(
                "**X-ray verification:** "
                f"{xray_result}"
            )


            st.write(
                "**X-ray confidence:** "
                f"{xray_confidence * 100:.2f}%"
            )


            st.write(
                "**Normal probability:** "
                f"{normal_probability * 100:.2f}%"
            )


            st.write(
                "**Pneumonia probability:** "
                f"{pneumonia_probability * 100:.2f}%"
            )


            st.write(
                f"**Diagnosis:** {diagnosis}"
            )


            st.write(
                "**Diagnosis confidence:** "
                f"{diagnosis_confidence * 100:.2f}%"
            )


            # =================================================
            # PDF REPORT
            # =================================================

            pdf_data = create_pdf(
                filename=uploaded_file.name,
                diagnosis=diagnosis,
                diagnosis_confidence=diagnosis_confidence,
                xray_confidence=xray_confidence,
                normal_probability=normal_probability,
                pneumonia_probability=pneumonia_probability
            )


            clean_filename = (
                uploaded_file.name
                .encode(
                    "ascii",
                    "ignore"
                )
                .decode(
                    "ascii"
                )
            )


            st.download_button(
                label="Download Diagnostic Report",
                data=pdf_data,
                file_name=(
                    f"Report_{clean_filename}.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the image."
        )

        st.exception(e)
