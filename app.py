# ============================================================
# PNEUMONIA AI — MODALITY VERIFICATION + PNEUMONIA DETECTION
# ============================================================
#
# Modality model:
#   0 = CHEST_XRAY
#   1 = CT
#   2 = MRI
#
# Workflow:
#
#   Upload image
#        |
#        v
#   Colour image?
#      YES ---> Reject
#      NO
#        |
#        v
#   Modality classifier
#        |
#   +----+----+
#   |         |
# X-RAY     CT/MRI
#   |         |
#   v         v
# Pneumonia  Reject
# Detection
#   |
#   v
# Normal / Pneumonia
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

from PIL import Image, ImageStat
from fpdf import FPDF


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia AI",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# 3. MODEL PATHS
# ============================================================

# ------------------------------------------------------------
# NEW MODALITY MODEL
#
# This is the model from your latest training code:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
# ------------------------------------------------------------

MODALITY_MODEL_PATH = "CT_Verifier.keras"


# ------------------------------------------------------------
# PNEUMONIA MODEL
# ------------------------------------------------------------

PNEUMONIA_MODEL_PATH = (
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# 4. IMAGE SIZE
# ============================================================

MODALITY_IMAGE_SIZE = (
    128,
    128
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# 5. MODALITY CLASS NAMES
# ============================================================

# IMPORTANT:
#
# This MUST match the training code:
#
# CHEST_XRAY = 0
# CT         = 1
# MRI        = 2
#
# flow_from_directory() normally sorts these
# alphabetically, so this is correct.
# ============================================================

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# 6. CHEST X-RAY THRESHOLD
# ============================================================
#
# The X-ray must:
#
# 1. Be predicted as CHEST_XRAY
# 2. Have at least this probability
#
# Do NOT make this extremely low.
#
# Start with 0.50.
#
# If valid X-rays are still rejected after testing,
# this can be reduced to 0.40.
#
# ============================================================

CHEST_XRAY_THRESHOLD = 0.50


# ============================================================
# 7. PNEUMONIA THRESHOLD
# ============================================================

PNEUMONIA_THRESHOLD = 0.50


# ============================================================
# 8. COLOUR IMAGE DETECTION
# ============================================================
#
# Your requirement:
#
# "If I give Colour image reject it."
#
# We therefore perform an explicit colour check BEFORE
# sending the image to the modality model.
#
# A grayscale image saved as RGB can still have 3 channels,
# so checking only image.mode == "RGB" is WRONG.
#
# Instead, we compare RGB channels.
#
# COLOR_TOLERANCE:
#   Smaller = stricter grayscale requirement.
#
# ============================================================

COLOR_TOLERANCE = 8.0


def is_colour_image(
    image,
    tolerance=COLOR_TOLERANCE
):
    """
    Returns True if the image contains meaningful
    colour information.

    Grayscale images may be stored internally as RGB.
    Therefore, simply checking image.mode is not enough.
    """

    rgb_image = image.convert("RGB")

    array = np.asarray(
        rgb_image,
        dtype=np.int16
    )

    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]

    # Difference between colour channels
    rg_difference = np.abs(
        red - green
    )

    rb_difference = np.abs(
        red - blue
    )

    gb_difference = np.abs(
        green - blue
    )

    maximum_difference = np.maximum(
        np.maximum(
            rg_difference,
            rb_difference
        ),
        gb_difference
    )

    # Percentage of pixels containing meaningful colour
    coloured_pixels = (
        maximum_difference > tolerance
    )

    colour_ratio = np.mean(
        coloured_pixels
    )

    # Reject if more than 1% of pixels
    # contain meaningful colour information.
    return colour_ratio > 0.01


# ============================================================
# 9. CONVERT IMAGE TO GRAYSCALE
# ============================================================

def prepare_grayscale_image(image):
    """
    Converts the accepted image to grayscale and
    then back to RGB.

    This gives the CNN a 3-channel input while preserving
    the grayscale medical-image information.
    """

    grayscale = image.convert("L")

    return grayscale.convert("RGB")


# ============================================================
# 10. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# 11. LOAD MODALITY MODEL
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(
        MODALITY_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Modality model not found:\n"
            f"{MODALITY_MODEL_PATH}\n\n"
            "Upload CT_Verifier.keras to the "
            "same directory as app.py."
        )

    model = tf.keras.models.load_model(
        MODALITY_MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# 12. LOAD PNEUMONIA MODEL
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

    # --------------------------------------------------------
    # First try complete .keras model
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

        return model

    except Exception:

        pass


    # --------------------------------------------------------
    # Fallback: build architecture and load weights
    # --------------------------------------------------------

    from model_builder import build_model

    model = build_model(
        input_shape=(
            224,
            224,
            3
        )
    )

    model.load_weights(
        PNEUMONIA_MODEL_PATH
    )

    return model


# ============================================================
# 13. LOAD MODELS
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
# 14. VERIFY MODALITY MODEL OUTPUT
# ============================================================

try:

    modality_output_shape = (
        modality_model.output_shape
    )

    if (
        modality_output_shape[-1]
        != 3
    ):

        st.error(
            "Incorrect modality model detected."
        )

        st.write(
            "Expected output classes: 3"
        )

        st.write(
            "Expected mapping:"
        )

        st.write(
            "0 = CHEST_XRAY"
        )

        st.write(
            "1 = CT"
        )

        st.write(
            "2 = MRI"
        )

        st.write(
            "Actual model output:",
            modality_output_shape
        )

        st.stop()

except Exception as e:

    st.error(
        "Could not verify modality model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 15. HEADER
# ============================================================

st.title(
    "🫁 Pneumonia Detection System"
)

st.markdown(
    """
### Medical Image Verification Pipeline

1. **Colour images are rejected**
2. **CT scans are identified and rejected**
3. **MRI images are identified and rejected**
4. **Chest X-rays proceed to pneumonia detection**
5. **Chest X-ray → Normal or Pneumonia**
6. **A PDF diagnostic report can be downloaded**
"""
)


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        "**Modality Model:**"
    )

    st.write(
        "CT_Verifier.keras"
    )

    st.write(
        "**Classes:**"
    )

    st.write(
        "0 → Chest X-ray"
    )

    st.write(
        "1 → CT"
    )

    st.write(
        "2 → MRI"
    )

    st.divider()

    st.write(
        "**X-ray threshold:** "
        f"{CHEST_XRAY_THRESHOLD:.2f}"
    )

    st.write(
        "**Pneumonia threshold:** "
        f"{PNEUMONIA_THRESHOLD:.2f}"
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
# 17. FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Upload an image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],

    help=(
        "Upload a grayscale medical image. "
        "Colour images will be rejected."
    )
)


# ============================================================
# 18. IMAGE PROCESSING
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


        # ====================================================
        # BASIC IMAGE VALIDATION
        # ====================================================

        if image.width < 32 or image.height < 32:

            st.error(
                "❌ Image is too small."
            )

            st.stop()


        # ====================================================
        # DISPLAY ORIGINAL IMAGE
        # ====================================================

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
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
                "Step 0 — Image Type Check"
            )


            colour_detected = is_colour_image(
                image
            )


            # =================================================
            # REJECT COLOUR IMAGE
            # =================================================

            if colour_detected:

                st.error(
                    "❌ Colour image detected."
                )

                st.warning(
                    "Please upload a grayscale "
                    "medical image such as a "
                    "Chest X-ray, CT scan, or MRI."
                )


                history_entry = (
                    "Rejected - Colour image - "
                    f"{uploaded_file.name}"
                )


                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                st.stop()


            # =================================================
            # GRAYSCALE IMAGE ACCEPTED
            # =================================================

            st.success(
                "✓ Grayscale image detected."
            )


            # =================================================
            # STEP 1 — MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 1 — Medical Image Modality"
            )


            # -------------------------------------------------
            # Convert grayscale image to RGB
            #
            # MobileNetV2 expects 3 channels.
            # -------------------------------------------------

            model_image = (
                prepare_grayscale_image(
                    image
                )
            )


            modality_image = (
                model_image.resize(
                    MODALITY_IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )
            )


            modality_array = np.asarray(
                modality_image,
                dtype=np.float32
            )


            # =================================================
            # IMPORTANT PREPROCESSING
            # =================================================
            #
            # Your training code uses:
            #
            # rescale=1.0/255.0
            #
            # Therefore DO NOT use MobileNetV2
            # preprocess_input() here.
            #
            # This is a critical correction.
            # =================================================

            modality_array = (
                modality_array / 255.0
            )


            modality_input = np.expand_dims(
                modality_array,
                axis=0
            )


            # =================================================
            # MODALITY PREDICTION
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
                modality_prediction
            )


            # =================================================
            # OUTPUT VALIDATION
            # =================================================

            if modality_prediction.ndim != 2:

                st.error(
                    "Invalid modality model output."
                )

                st.stop()


            if (
                modality_prediction.shape[1]
                != 3
            ):

                st.error(
                    "The loaded modality model "
                    "does not have 3 outputs."
                )

                st.write(
                    "Output shape:",
                    modality_prediction.shape
                )

                st.stop()


            # =================================================
            # GET PROBABILITIES
            # =================================================

            modality_probabilities = (
                modality_prediction[0]
                .astype(np.float64)
            )


            # =================================================
            # SOFTMAX SAFETY CHECK
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

            chest_xray_probability = float(
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
            # DISPLAY MODALITY PROBABILITIES
            # =================================================

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Chest X-ray",
                    f"{chest_xray_probability * 100:.2f}%"
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
            # DECISION
            # =================================================
            #
            # Chest X-ray is accepted ONLY when:
            #
            # 1. Chest X-ray is the highest probability
            # 2. X-ray probability >= threshold
            #
            # =================================================

            is_chest_xray = (

                predicted_index == 0

                and

                chest_xray_probability
                >= CHEST_XRAY_THRESHOLD
            )


            # =================================================
            # CT DETECTED
            # =================================================

            if predicted_index == 1:

                st.error(
                    "❌ CT Scan image detected."
                )

                st.warning(
                    "This application accepts only "
                    "Chest X-ray images for pneumonia detection."
                )

                st.write(
                    f"CT confidence: "
                    f"{ct_probability * 100:.2f}%"
                )


                history_entry = (
                    "CT detected - "
                    f"{uploaded_file.name}"
                )


                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                st.stop()


            # =================================================
            # MRI DETECTED
            # =================================================

            if predicted_index == 2:

                st.error(
                    "❌ MRI image detected."
                )

                st.warning(
                    "This application accepts only "
                    "Chest X-ray images for pneumonia detection."
                )

                st.write(
                    f"MRI confidence: "
                    f"{mri_probability * 100:.2f}%"
                )


                history_entry = (
                    "MRI detected - "
                    f"{uploaded_file.name}"
                )


                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                st.stop()


            # =================================================
            # LOW-CONFIDENCE MODALITY
            # =================================================

            if not is_chest_xray:

                st.error(
                    "❌ Image could not be confidently "
                    "identified as a Chest X-ray."
                )

                st.warning(
                    "Please upload a clear Chest X-ray image."
                )

                st.write(
                    f"Predicted modality: "
                    f"**{predicted_modality}**"
                )

                st.write(
                    f"Confidence: "
                    f"{predicted_confidence * 100:.2f}%"
                )


                history_entry = (
                    "Low confidence - "
                    f"{uploaded_file.name}"
                )


                if (
                    history_entry
                    not in st.session_state.history
                ):

                    st.session_state.history.append(
                        history_entry
                    )


                st.stop()


            # =================================================
            # CHEST X-RAY CONFIRMED
            # =================================================

            st.success(
                "✅ Chest X-ray image detected."
            )

            st.write(
                f"Chest X-ray confidence: "
                f"{chest_xray_probability * 100:.2f}%"
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

            pneumonia_image = (
                model_image.resize(
                    PNEUMONIA_IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )
            )


            pneumonia_array = np.asarray(
                pneumonia_image,
                dtype=np.float32
            )


            # -------------------------------------------------
            # Your pneumonia model uses /255 preprocessing
            # -------------------------------------------------

            pneumonia_array = (
                pneumonia_array / 255.0
            )


            pneumonia_input = np.expand_dims(
                pneumonia_array,
                axis=0
            )


            # =================================================
            # PNEUMONIA PREDICTION
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


            prediction = np.asarray(
                prediction
            )


            prediction_values = np.squeeze(
                prediction
            )


            # =================================================
            # HANDLE MODEL OUTPUT
            # =================================================
            #
            # Supports:
            #
            # 1. Binary sigmoid:
            #       [pneumonia_probability]
            #
            # 2. Two-class softmax:
            #       [normal, pneumonia]
            #
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

                # If the model output is outside
                # probability range, apply sigmoid.
                if (
                    pneumonia_probability < 0
                    or
                    pneumonia_probability > 1
                ):

                    pneumonia_probability = (
                        1.0 /
                        (
                            1.0
                            +
                            np.exp(
                                -pneumonia_probability
                            )
                        )
                    )

                normal_probability = (
                    1.0
                    -
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

                    probabilities = (
                        probabilities
                    )

                else:

                    probabilities = (
                        tf.nn.softmax(
                            probabilities
                        ).numpy()
                    )


                # ------------------------------------------------
                # ASSUMED TRAINING CLASS ORDER:
                #
                # NORMAL = 0
                # PNEUMONIA = 1
                # ------------------------------------------------

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
                    "Output shape:",
                    prediction.shape
                )

                st.stop()


            # =================================================
            # CLAMP PROBABILITIES
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
                    f"{chest_xray_probability * 100:.2f}%"
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
                f"**Image Modality:** "
                f"Chest X-ray"
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

            history_entry = (
                f"{diagnosis} - "
                f"{uploaded_file.name}"
            )


            if (
                history_entry
                not in st.session_state.history
            ):

                st.session_state.history.append(
                    history_entry
                )


            # =================================================
            # DIAGNOSTIC REPORT
            # =================================================

            st.divider()

            st.subheader(
                "Diagnostic Report"
            )


            report_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )


            st.write(
                f"**File:** "
                f"{uploaded_file.name}"
            )

            st.write(
                f"**Date:** "
                f"{report_time}"
            )

            st.write(
                "**Image Modality:** "
                "Chest X-ray"
            )

            st.write(
                f"**X-ray Confidence:** "
                f"{chest_xray_probability * 100:.2f}%"
            )

            st.write(
                f"**Normal Probability:** "
                f"{normal_probability * 100:.2f}%"
            )

            st.write(
                f"**Pneumonia Probability:** "
                f"{pneumonia_probability * 100:.2f}%"
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
            # PDF REPORT
            # =================================================

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


            # Remove problematic characters
            clean_filename = (
                clean_filename
                .replace(
                    ".",
                    "_"
                )
                .replace(
                    " ",
                    "_"
                )
            )


            pdf = FPDF()

            pdf.add_page()


            # =================================================
            # PDF TITLE
            # =================================================

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


            pdf.ln(
                10
            )


            # =================================================
            # REPORT DATE
            # =================================================

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
                report_time,
                ln=True
            )


            # =================================================
            # FILE NAME
            # =================================================

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


            # =================================================
            # MODALITY
            # =================================================

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
                "Chest X-ray",
                ln=True
            )


            # =================================================
            # X-RAY CONFIDENCE
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                11
            )

            pdf.cell(
                50,
                8,
                "X-ray Confidence:",
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
                (
                    f"{chest_xray_probability * 100:.2f}%"
                ),
                ln=True
            )


            # =================================================
            # NORMAL PROBABILITY
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                11
            )

            pdf.cell(
                50,
                8,
                "Normal Probability:",
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
                (
                    f"{normal_probability * 100:.2f}%"
                ),
                ln=True
            )


            # =================================================
            # PNEUMONIA PROBABILITY
            # =================================================

            pdf.set_font(
                "Arial",
                "B",
                11
            )

            pdf.cell(
                50,
                8,
                "Pneumonia Probability:",
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
                (
                    f"{pneumonia_probability * 100:.2f}%"
                ),
                ln=True
            )


            # =================================================
            # DIAGNOSIS
            # =================================================

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


            # =================================================
            # DIAGNOSIS CONFIDENCE
            # =================================================

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
                (
                    f"{diagnosis_confidence * 100:.2f}%"
                ),
                ln=True
            )


            # =================================================
            # DISCLAIMER
            # =================================================

            pdf.ln(
                15
            )

            pdf.set_font(
                "Arial",
                "I",
                9
            )

            pdf.multi_cell(
                0,
                6,
                (
                    "Disclaimer: This AI-generated result "
                    "is intended for research purposes only "
                    "and does not replace professional "
                    "medical diagnosis."
                )
            )


            # =================================================
            # GENERATE PDF
            # =================================================

            pdf_output = pdf.output(
                dest="S"
            )


            if isinstance(
                pdf_output,
                str
            ):

                pdf_bytes = (
                    pdf_output.encode(
                        "latin-1"
                    )
                )

            else:

                pdf_bytes = bytes(
                    pdf_output
                )


            # =================================================
            # DOWNLOAD BUTTON
            # =================================================

            st.download_button(

                label=(
                    "Download Diagnostic Report (PDF)"
                ),

                data=pdf_bytes,

                file_name=(
                    f"Report_{clean_filename}.pdf"
                ),

                mime="application/pdf"
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
