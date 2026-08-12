import os
import io

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from fpdf import FPDF

from model_builder import build_model
from xray_model_builder import build_xray_classifier
from modality_model_builder import build_modality_classifier


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

PNEUMONIA_MODEL_PATH = (
    "best_xception_pneumonia_model.keras"
)

MODALITY_MODEL_PATH = (
    "best_modality_classifier.weights.h5"
)


# ============================================================
# IMAGE SIZES
# ============================================================

XRAY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)

MODALITY_IMAGE_SIZE = (224, 224)


# ============================================================
# IMAGE VALIDATION SETTINGS
# ============================================================

# DO NOT CHANGE THIS COLOR IMAGE VALIDATION
COLOR_TOLERANCE = 5.0


# Existing X-ray verifier threshold
XRAY_CONFIDENCE_THRESHOLD = 0.50


# CT / MRI / modality classifier threshold
MODALITY_CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# X-RAY VERIFIER CLASS MAPPING
# ============================================================

#
# Existing verifier:
#
# 0 = X-RAY
# 1 = NON-XRAY
#

XRAY_CLASS_MAP = {
    0: "X-RAY",
    1: "NON-XRAY"
}


# ============================================================
# MEDICAL IMAGE MODALITY CLASS MAPPING
# ============================================================

#
# IMPORTANT:
#
# This must exactly match the class order used
# when training the modality classifier.
#
# 0 = Chest X-ray
# 1 = CT
# 2 = MRI
# 3 = Other
#

MODALITY_CLASS_MAP = {
    0: "CHEST_XRAY",
    1: "CT",
    2: "MRI",
    3: "OTHER"
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# LOAD MODALITY CLASSIFIER
# ============================================================

@st.cache_resource
def load_modality_model():

    if not os.path.isfile(MODALITY_MODEL_PATH):

        raise FileNotFoundError(
            "Medical image modality classifier "
            "weights not found:\n"
            f"{MODALITY_MODEL_PATH}"
        )

    model = build_modality_classifier(
        input_shape=(224, 224, 3),
        num_classes=4
    )

    model.load_weights(
        MODALITY_MODEL_PATH
    )

    return model


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    if not os.path.isfile(XRAY_MODEL_PATH):

        raise FileNotFoundError(
            "X-ray verifier weights not found:\n"
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
            "Pneumonia model not found:\n"
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
# LOAD ALL MODELS
# ============================================================

try:

    modality_model = modality_classifier_model()

    xray_model = load_xray_model()

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫁 Pneumonia Detection System"
)

st.markdown(
    """
Upload an image to the system.

**Step 1:** Verify the image type.

**Step 2:** Reject color images, CT scans, MRI images,
and unsupported images.

**Step 3:** If the image is a chest X-ray,
analyze it for pneumonia.

**Step 4:** Report whether the X-ray is Normal
or Pneumonia.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the System"
    )

    st.write(
        "This application uses three "
        "deep-learning models:"
    )

    st.write(
        "1. Medical image modality verification"
    )

    st.write(
        "2. Chest X-ray / Non-X-ray verification"
    )

    st.write(
        "3. Pneumonia detection"
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
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload a chest X-ray image."
    )
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        file_bytes = (
            uploaded_file.getvalue()
        )

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        image_array = np.array(
            image
        )


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

                if (
                    image_array is None
                    or image_array.size == 0
                ):

                    st.error(
                        "Could not read the uploaded image."
                    )

                    st.stop()


                # ==================================================
                # STEP 2 — IMAGE TYPE VALIDATION
                # ==================================================

                st.subheader(
                    "Step 1 — Image Verification"
                )


                # ==================================================
                # COLOR IMAGE REJECTION
                # ==================================================

                # --------------------------------------------------
                # CHECK WHETHER IMAGE IS ACTUALLY COLOR
                # --------------------------------------------------

                # The uploaded image has already been converted
                # to RGB.
                #
                # Compare the RGB channels.
                # A genuine grayscale chest X-ray should have
                # very similar R, G and B values.

                rgb_image = (
                    image_array.astype(
                        np.float32
                    )
                )

                red_channel = (
                    rgb_image[:, :, 0]
                )

                green_channel = (
                    rgb_image[:, :, 1]
                )

                blue_channel = (
                    rgb_image[:, :, 2]
                )

                channel_difference = (
                    np.mean(
                        np.abs(
                            red_channel
                            - green_channel
                        )
                    )
                    +
                    np.mean(
                        np.abs(
                            green_channel
                            - blue_channel
                        )
                    )
                    +
                    np.mean(
                        np.abs(
                            red_channel
                            - blue_channel
                        )
                    )
                ) / 3.0


                # --------------------------------------------------
                # REJECT COLOR IMAGES
                # --------------------------------------------------

                if (
                    channel_difference
                    > COLOR_TOLERANCE
                ):

                    st.error(
                        "❌ This is not a Chest X-ray image."
                    )

                    st.warning(
                        "Color images are not accepted. "
                        "Please upload a grayscale chest X-ray."
                    )

                    history_entry = (
                        f"Rejected - Color image - "
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


                # ==================================================
                # STEP 3 — CT / MRI / MODALITY REJECTION
                # ==================================================

                st.subheader(
                    "Medical Image Modality Verification"
                )


                # --------------------------------------------------
                # PREPARE MODALITY IMAGE
                # --------------------------------------------------

                modality_image = cv2.resize(
                    image_array,
                    MODALITY_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )

                modality_image = (
                    modality_image.astype(
                        np.float32
                    ) / 255.0
                )

                modality_input = (
                    np.expand_dims(
                        modality_image,
                        axis=0
                    )
                )


                # --------------------------------------------------
                # RUN MODALITY CLASSIFIER
                # --------------------------------------------------

                modality_prediction = (
                    modality_model.predict(
                        modality_input,
                        verbose=0
                    )
                )

                modality_prediction = (
                    np.asarray(
                        modality_prediction
                    )
                )


                # --------------------------------------------------
                # VALIDATE MODALITY OUTPUT
                # --------------------------------------------------

                if (
                    modality_prediction.ndim
                    != 2
                    or
                    modality_prediction.shape[1]
                    != 4
                ):

                    st.error(
                        "Unable to determine "
                        "the medical image modality."
                    )

                    st.stop()


                # --------------------------------------------------
                # GET MODALITY SCORES
                # --------------------------------------------------

                modality_scores = (
                    modality_prediction[0]
                    .astype(
                        np.float64
                    )
                )


                # --------------------------------------------------
                # CONVERT TO PROBABILITIES
                # --------------------------------------------------

                if (
                    np.all(
                        modality_scores
                        >= 0.0
                    )
                    and
                    np.all(
                        modality_scores
                        <= 1.0
                    )
                    and
                    np.isclose(
                        np.sum(
                            modality_scores
                        ),
                        1.0,
                        atol=1e-3
                    )
                ):

                    modality_probabilities = (
                        modality_scores
                    )

                else:

                    modality_probabilities = (
                        tf.nn.softmax(
                            modality_scores
                        ).numpy()
                    )


                # --------------------------------------------------
                # GET MODALITY CLASS
                # --------------------------------------------------

                modality_class_index = int(
                    np.argmax(
                        modality_probabilities
                    )
                )

                modality_confidence = float(
                    modality_probabilities[
                        modality_class_index
                    ]
                )

                modality_result = (
                    MODALITY_CLASS_MAP.get(
                        modality_class_index,
                        "UNKNOWN"
                    )
                )


                # ==================================================
                # DISPLAY MODALITY RESULT
                # ==================================================

                if modality_result == "CHEST_XRAY":

                    if (
                        modality_confidence
                        >= MODALITY_CONFIDENCE_THRESHOLD
                    ):

                        st.success(
                            "✅ Chest X-ray image detected."
                        )

                        st.write(
                            "Chest X-ray confidence: "
                            f"{modality_confidence * 100:.2f}%"
                        )

                    else:

                        st.error(
                            "❌ Chest X-ray confidence is too low."
                        )

                        st.warning(
                            "Please upload a clear chest X-ray image."
                        )

                        history_entry = (
                            f"Rejected - Low X-ray confidence - "
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


                elif modality_result == "CT":

                    st.error(
                        "❌ CT scan detected."
                    )

                    st.warning(
                        "This system accepts only "
                        "chest X-ray images."
                    )

                    st.write(
                        "CT confidence: "
                        f"{modality_confidence * 100:.2f}%"
                    )

                    history_entry = (
                        f"Rejected - CT scan - "
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


                elif modality_result == "MRI":

                    st.error(
                        "❌ MRI image detected."
                    )

                    st.warning(
                        "This system accepts only "
                        "chest X-ray images."
                    )

                    st.write(
                        "MRI confidence: "
                        f"{modality_confidence * 100:.2f}%"
                    )

                    history_entry = (
                        f"Rejected - MRI - "
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


                elif modality_result == "OTHER":

                    st.error(
                        "❌ Unsupported medical image detected."
                    )

                    st.warning(
                        "Please upload a chest X-ray image."
                    )

                    history_entry = (
                        f"Rejected - Other image - "
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


                else:

                    st.error(
                        "❌ Unknown image modality."
                    )

                    st.warning(
                        "Please upload a valid chest X-ray."
                    )

                    history_entry = (
                        f"Rejected - Unknown modality - "
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


                # ==================================================
                # STEP 4 — X-RAY VERIFIER
                # ==================================================

                st.subheader(
                    "Step 2 — Chest X-ray Verification"
                )


                # --------------------------------------------------
                # RESIZE IMAGE FOR X-RAY VERIFIER
                # --------------------------------------------------

                verifier_image = cv2.resize(
                    image_array,
                    XRAY_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )


                # --------------------------------------------------
                # CONVERT TO FLOAT
                # --------------------------------------------------

                verifier_image = (
                    verifier_image.astype(
                        np.float32
                    ) / 255.0
                )


                # --------------------------------------------------
                # ADD BATCH DIMENSION
                # --------------------------------------------------

                verifier_input = (
                    np.expand_dims(
                        verifier_image,
                        axis=0
                    )
                )


                # --------------------------------------------------
                # RUN X-RAY VERIFIER
                # --------------------------------------------------

                verifier_prediction = (
                    xray_model.predict(
                        verifier_input,
                        verbose=0
                    )
                )

                verifier_prediction = (
                    np.asarray(
                        verifier_prediction
                    )
                )


                # --------------------------------------------------
                # VALIDATE VERIFIER OUTPUT
                # --------------------------------------------------

                if (
                    verifier_prediction.ndim
                    != 2
                    or
                    verifier_prediction.shape[1]
                    != 2
                ):

                    st.error(
                        "Unable to verify the image type."
                    )

                    st.stop()


                # --------------------------------------------------
                # GET VERIFIER SCORES
                # --------------------------------------------------

                raw_scores = (
                    verifier_prediction[0]
                    .astype(
                        np.float64
                    )
                )


                # --------------------------------------------------
                # CONVERT TO PROBABILITIES
                # --------------------------------------------------

                if (
                    np.all(
                        raw_scores >= 0.0
                    )
                    and
                    np.all(
                        raw_scores <= 1.0
                    )
                    and
                    np.isclose(
                        np.sum(
                            raw_scores
                        ),
                        1.0,
                        atol=1e-3
                    )
                ):

                    verifier_probabilities = (
                        raw_scores
                    )

                else:

                    verifier_probabilities = (
                        tf.nn.softmax(
                            raw_scores
                        ).numpy()
                    )


                # --------------------------------------------------
                # GET PREDICTED CLASS
                # --------------------------------------------------

                verifier_class_index = int(
                    np.argmax(
                        verifier_probabilities
                    )
                )

                verifier_confidence = float(
                    verifier_probabilities[
                        verifier_class_index
                    ]
                )

                verifier_result = (
                    XRAY_CLASS_MAP.get(
                        verifier_class_index,
                        "UNKNOWN"
                    )
                )


                # ==================================================
                # REJECT NON-X-RAY
                # ==================================================

                if (
                    verifier_result != "X-RAY"
                    or
                    verifier_confidence
                    < XRAY_CONFIDENCE_THRESHOLD
                ):

                    st.error(
                        "❌ This is not a Chest X-ray image."
                    )

                    st.warning(
                        "Please upload a valid chest X-ray image."
                    )

                    history_entry = (
                        f"Rejected - Non-X-ray - "
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


                # ==================================================
                # X-RAY CONFIRMED
                # ==================================================

                st.success(
                    "✅ Chest X-ray verified."
                )

                st.write(
                    "X-ray verification confidence: "
                    f"{verifier_confidence * 100:.2f}%"
                )


                # --------------------------------------------------
                # X-RAY PROBABILITIES
                # --------------------------------------------------

                xray_probability = float(
                    verifier_probabilities[0]
                )

                non_xray_probability = float(
                    verifier_probabilities[1]
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


                # ==================================================
                # STEP 5 — PNEUMONIA DETECTION
                # ==================================================

                st.subheader(
                    "Step 3 — Pneumonia Detection"
                )


                # --------------------------------------------------
                # PREPARE PNEUMONIA MODEL INPUT
                # --------------------------------------------------

                pneumonia_image = cv2.resize(
                    image_array,
                    PNEUMONIA_IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )

                pneumonia_image = (
                    pneumonia_image.astype(
                        np.float32
                    ) / 255.0
                )

                pneumonia_input = (
                    np.expand_dims(
                        pneumonia_image,
                        axis=0
                    )
                )


                # --------------------------------------------------
                # RUN PNEUMONIA MODEL
                # --------------------------------------------------

                prediction = (
                    pneumonia_model.predict(
                        pneumonia_input,
                        verbose=0
                    )
                )

                prediction = (
                    np.asarray(
                        prediction
                    )
                )


                # ==================================================
                # STEP 6 — PROCESS 2-CLASS OUTPUT
                # ==================================================

                # Expected model output:
                #
                # Class 0 = Normal
                # Class 1 = Pneumonia

                if (
                    prediction.ndim != 2
                    or
                    prediction.shape[1] != 2
                ):

                    st.error(
                        "Unable to process "
                        "the pneumonia prediction."
                    )

                    st.stop()


                # --------------------------------------------------
                # GET MODEL SCORES
                # --------------------------------------------------

                pneumonia_scores = (
                    prediction[0]
                    .astype(
                        np.float64
                    )
                )


                # --------------------------------------------------
                # CONVERT LOGITS TO PROBABILITIES
                # --------------------------------------------------

                if (
                    np.all(
                        pneumonia_scores
                        >= 0.0
                    )
                    and
                    np.all(
                        pneumonia_scores
                        <= 1.0
                    )
                    and
                    np.isclose(
                        np.sum(
                            pneumonia_scores
                        ),
                        1.0,
                        atol=1e-3
                    )
                ):

                    pneumonia_probabilities = (
                        pneumonia_scores
                    )

                else:

                    pneumonia_probabilities = (
                        tf.nn.softmax(
                            pneumonia_scores
                        ).numpy()
                    )


                # --------------------------------------------------
                # CLASS MAPPING
                # --------------------------------------------------
                #
                # 0 = Normal
                # 1 = Pneumonia
                #

                normal_probability = float(
                    pneumonia_probabilities[0]
                )

                pneumonia_probability = float(
                    pneumonia_probabilities[1]
                )


                # ==================================================
                # STEP 7 — FINAL DIAGNOSIS
                # ==================================================

                if (
                    pneumonia_probability
                    >= normal_probability
                ):

                    diagnosis = (
                        "Pneumonia"
                    )

                    diagnosis_confidence = (
                        pneumonia_probability
                    )

                else:

                    diagnosis = (
                        "Normal"
                    )

                    diagnosis_confidence = (
                        normal_probability
                    )


                # --------------------------------------------------
                # DISPLAY DIAGNOSIS
                # --------------------------------------------------

                if diagnosis == "Pneumonia":

                    st.error(
                        "Diagnosis: Pneumonia"
                    )

                else:

                    st.success(
                        "Diagnosis: Normal"
                    )


                # --------------------------------------------------
                # FINAL RESULT
                # --------------------------------------------------

                st.subheader(
                    "Final Result"
                )


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
                    f"**Final Diagnosis:** "
                    f"{diagnosis}"
                )

                st.write(
                    f"**Diagnosis Confidence:** "
                    f"{diagnosis_confidence * 100:.2f}%"
                )


                # ==================================================
                # STEP 8 — HISTORY
                # ==================================================

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


                # ==================================================
                # STEP 9 — PDF REPORT
                # ==================================================

                st.divider()

                st.subheader(
                    "Diagnostic Report"
                )


                st.write(
                    f"**File:** "
                    f"{uploaded_file.name}"
                )

                st.write(
                    f"**Image modality:** "
                    f"Chest X-ray"
                )

                st.write(
                    f"**Modality confidence:** "
                    f"{modality_confidence * 100:.2f}%"
                )

                st.write(
                    f"**X-ray verification:** "
                    f"{verifier_result}"
                )

                st.write(
                    f"**X-ray confidence:** "
                    f"{verifier_confidence * 100:.2f}%"
                )

                st.write(
                    f"**Diagnosis:** "
                    f"{diagnosis}"
                )

                st.write(
                    f"**Diagnosis confidence:** "
                    f"{diagnosis_confidence * 100:.2f}%"
                )


                # --------------------------------------------------
                # CLEAN FILE NAME
                # --------------------------------------------------

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


                # --------------------------------------------------
                # CREATE PDF
                # --------------------------------------------------

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


                pdf.set_font(
                    "Arial",
                    "B",
                    12
                )

                pdf.cell(
                    45,
                    10,
                    "Modality Confidence:",
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
                    f"{modality_confidence * 100:.2f}%",
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
                    f"{verifier_confidence * 100:.2f}%",
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

                pdf_output = (
                    pdf.output()
                )


                st.download_button(
                    label=(
                        "Download Diagnostic Report"
                    ),
                    data=bytes(
                        pdf_output
                    ),
                    file_name=(
                        f"Report_{clean_filename}.pdf"
                    ),
                    mime="application/pdf"
                )


    except Exception as e:

        st.error(
            "An error occurred while "
            "processing the image."
        )

        st.exception(e)
