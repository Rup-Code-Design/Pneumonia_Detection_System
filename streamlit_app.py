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
# BASE DIRECTORY
# ============================================================
#
# Always resolve model files relative to this app.py file.
#
# This avoids problems such as:
#
# /mount/src/pneumonia_detection_system/
#
# versus the current working directory.
#
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATHS
# ============================================================

XRAY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xray_verifier.weights.h5"
)

MODALITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_modality_classifier.weights.h5"
)

PNEUMONIA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_xception_pneumonia_model.keras"
)


# ============================================================
# IMAGE SIZES
# ============================================================

MODALITY_IMAGE_SIZE = (224, 224)

XRAY_IMAGE_SIZE = (128, 128)

PNEUMONIA_IMAGE_SIZE = (224, 224)


# ============================================================
# THRESHOLDS
# ============================================================

# Color rejection threshold.
#
# If the average RGB channel difference is greater than
# this value, the image is treated as a color image.
#
COLOR_TOLERANCE = 5.0


# Modality classifier threshold.
#
# Chest X-ray must have at least this probability.
#
MODALITY_CONFIDENCE_THRESHOLD = 0.90


# X-ray verifier threshold.
#
XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# CLASS MAPPINGS
# ============================================================

# ------------------------------------------------------------
# Medical modality classifier
# ------------------------------------------------------------
#
# MUST match the training class order.
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
# 3 = OTHER
#
# ------------------------------------------------------------

MODALITY_CLASS_MAP = {
    0: "CHEST_XRAY",
    1: "CT",
    2: "MRI",
    3: "OTHER"
}


# ------------------------------------------------------------
# X-ray verifier
# ------------------------------------------------------------
#
# MUST match the training class order.
#
# 0 = X-RAY
# 1 = NON-XRAY
#
# ------------------------------------------------------------

XRAY_CLASS_MAP = {
    0: "X-RAY",
    1: "NON-XRAY"
}


# ------------------------------------------------------------
# Pneumonia classifier
# ------------------------------------------------------------
#
# Expected:
#
# 0 = Normal
# 1 = Pneumonia
#
# ------------------------------------------------------------

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
# UTILITY — ADD HISTORY
# ============================================================

def add_history(entry):

    if entry not in st.session_state.history:

        st.session_state.history.append(
            entry
        )

        # Keep only the latest 20 records.
        st.session_state.history = (
            st.session_state.history[-20:]
        )


# ============================================================
# UTILITY — CHECK FILE
# ============================================================

def check_model_file(
    path,
    description
):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"{description} was not found.\n\n"
            f"Expected location:\n{path}\n\n"
            "Make sure the model file is committed to "
            "the same GitHub repository as this application."
        )

    # Check for empty or suspiciously small files.
    file_size = os.path.getsize(path)

    if file_size == 0:

        raise RuntimeError(
            f"{description} exists but is empty.\n\n"
            f"File: {path}"
        )

    return True


# ============================================================
# LOAD MODALITY CLASSIFIER
# ============================================================

@st.cache_resource
def load_modality_model():

    check_model_file(
        MODALITY_MODEL_PATH,
        "Medical modality classifier weights"
    )

    try:

        # ----------------------------------------------------
        # BUILD EXACT ARCHITECTURE FROM
        # modality_model_builder.py
        # ----------------------------------------------------

        model = build_modality_classifier(
            input_shape=(
                224,
                224,
                3
            ),
            num_classes=4
        )

        # ----------------------------------------------------
        # EXPLICITLY BUILD MODEL
        # ----------------------------------------------------

        dummy_input = tf.zeros(
            (
                1,
                224,
                224,
                3
            ),
            dtype=tf.float32
        )

        model(
            dummy_input,
            training=False
        )

        # ----------------------------------------------------
        # LOAD WEIGHTS
        # ----------------------------------------------------

        model.load_weights(
            MODALITY_MODEL_PATH
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "Medical modality classifier weights "
            "could not be loaded.\n\n"
            f"File:\n{MODALITY_MODEL_PATH}\n\n"
            "The architecture generated by "
            "modality_model_builder.py must EXACTLY "
            "match the architecture used during training.\n\n"
            "Also make sure the file was created using "
            "save_weights=True or "
            "ModelCheckpoint(save_weights_only=True).\n\n"
            "This is not a Streamlit threshold problem.\n\n"
            f"Original error:\n{e}"
        ) from e


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    check_model_file(
        XRAY_MODEL_PATH,
        "X-ray verifier weights"
    )

    try:

        model = build_xray_classifier(
            input_shape=(
                128,
                128,
                3
            )
        )

        # Explicitly build.
        dummy_input = tf.zeros(
            (
                1,
                128,
                128,
                3
            ),
            dtype=tf.float32
        )

        model(
            dummy_input,
            training=False
        )

        model.load_weights(
            XRAY_MODEL_PATH
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "X-ray verifier weights could not "
            "be loaded.\n\n"
            f"File:\n{XRAY_MODEL_PATH}\n\n"
            "The architecture generated by "
            "xray_model_builder.py must match "
            "the architecture used during training.\n\n"
            f"Original error:\n{e}"
        ) from e


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    check_model_file(
        PNEUMONIA_MODEL_PATH,
        "Pneumonia model"
    )

    try:

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # This is a .keras model.
        #
        # Therefore we load the COMPLETE MODEL.
        #
        # We do NOT call:
        #
        # model.load_weights(...)
        #
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False
        )

        # Explicitly build/check model.
        dummy_input = tf.zeros(
            (
                1,
                224,
                224,
                3
            ),
            dtype=tf.float32
        )

        model(
            dummy_input,
            training=False
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "Pneumonia model could not be loaded.\n\n"
            f"File:\n{PNEUMONIA_MODEL_PATH}\n\n"
            "The .keras file must be a complete Keras "
            "model compatible with the TensorFlow/Keras "
            "version used by this application.\n\n"
            f"Original error:\n{e}"
        ) from e


# ============================================================
# UTILITY — RGB COLOR CHECK
# ============================================================

def calculate_color_difference(
    image_array
):

    rgb = image_array.astype(
        np.float32
    )

    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]

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
# UTILITY — CONVERT SCORES TO PROBABILITIES
# ============================================================

def scores_to_probabilities(
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Already probabilities?
    # --------------------------------------------------------

    if (
        np.all(
            scores >= 0.0
        )
        and
        np.all(
            scores <= 1.0
        )
        and
        np.isclose(
            np.sum(scores),
            1.0,
            atol=1e-3
        )
    ):

        return scores

    # --------------------------------------------------------
    # Otherwise treat as logits.
    # --------------------------------------------------------

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫁 Pneumonia Detection System"
)

st.markdown(
    """
### Multi-stage Chest X-ray Analysis

The system performs the following checks:

1. **Color image rejection**
2. **Medical image modality verification**
3. **Chest X-ray verification**
4. **Pneumonia classification**

Only a valid grayscale chest X-ray is passed to
the pneumonia detection model.
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
        "The application uses three trained models:"
    )

    st.write(
        "1. Medical Modality Classifier"
    )

    st.write(
        "2. Chest X-ray Verifier"
    )

    st.write(
        "3. Pneumonia Detection Model"
    )

    st.divider()

    st.write(
        "**Accepted image:** "
        "grayscale chest X-ray"
    )

    st.write(
        "**Rejected:** color images, CT, MRI, "
        "non-X-ray images and unsupported images"
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
# MODEL STATUS
# ============================================================

with st.expander(
    "Model files",
    expanded=False
):

    st.write(
        f"Modality weights: `{MODALITY_MODEL_PATH}`"
    )

    st.write(
        f"X-ray verifier weights: `{XRAY_MODEL_PATH}`"
    )

    st.write(
        f"Pneumonia model: `{PNEUMONIA_MODEL_PATH}`"
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    modality_model = (
        load_modality_model()
    )

    xray_model = (
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

    st.stop()


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
        "Upload a grayscale chest X-ray image."
    )
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        file_bytes = (
            uploaded_file.getvalue()
        )

        if len(file_bytes) == 0:

            st.error(
                "The uploaded file is empty."
            )

            st.stop()

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        original_image = Image.open(
            io.BytesIO(file_bytes)
        )

        # Verify image format.
        original_image.verify()

        # Re-open after verify().
        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        image_array = np.array(
            image
        )

        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if (
            image_array is None
            or image_array.size == 0
        ):

            st.error(
                "Could not read the uploaded image."
            )

            st.stop()

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ----------------------------------------------------
        # IMAGE INFORMATION
        # ----------------------------------------------------

        with st.expander(
            "Image information",
            expanded=False
        ):

            st.write(
                f"**File:** {uploaded_file.name}"
            )

            st.write(
                f"**Width:** {image.width}px"
            )

            st.write(
                f"**Height:** {image.height}px"
            )

            st.write(
                f"**Original mode:** "
                f"{original_image.mode}"
            )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            # =================================================
            # STEP 1 — COLOR CHECK
            # =================================================

            st.subheader(
                "Step 1 — Color Image Verification"
            )

            color_difference = (
                calculate_color_difference(
                    image_array
                )
            )

            st.write(
                f"Average RGB channel difference: "
                f"{color_difference:.4f}"
            )

            # ------------------------------------------------
            # REJECT COLOR IMAGE
            # ------------------------------------------------

            if (
                color_difference
                > COLOR_TOLERANCE
            ):

                st.error(
                    "❌ Color image rejected."
                )

                st.warning(
                    "This system accepts grayscale "
                    "chest X-ray images only."
                )

                add_history(
                    f"Rejected - Color image - "
                    f"{uploaded_file.name}"
                )

                st.stop()

            st.success(
                "✓ Grayscale image detected."
            )


            # =================================================
            # STEP 2 — MEDICAL MODALITY CLASSIFICATION
            # =================================================

            st.subheader(
                "Step 2 — Medical Image Modality Verification"
            )

            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            modality_image = cv2.resize(
                image_array,
                MODALITY_IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            modality_image = (
                modality_image.astype(
                    np.float32
                ) / 255.0
            )

            modality_input = np.expand_dims(
                modality_image,
                axis=0
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

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

            # ------------------------------------------------
            # VALIDATE OUTPUT
            # ------------------------------------------------

            if (
                modality_prediction.ndim != 2
                or
                modality_prediction.shape[0] != 1
                or
                modality_prediction.shape[1] != 4
            ):

                st.error(
                    "Invalid modality classifier output."
                )

                st.write(
                    f"Received shape: "
                    f"{modality_prediction.shape}"
                )

                st.stop()

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            modality_probabilities = (
                scores_to_probabilities(
                    modality_prediction[0]
                )
            )

            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # DISPLAY ALL MODALITY PROBABILITIES
            # ------------------------------------------------

            modality_cols = st.columns(4)

            modality_names = [
                "Chest X-ray",
                "CT",
                "MRI",
                "Other"
            ]

            for i, column in enumerate(
                modality_cols
            ):

                with column:

                    st.metric(
                        modality_names[i],
                        (
                            f"{modality_probabilities[i] * 100:.2f}%"
                        )
                    )

            # =================================================
            # REJECTION LOGIC
            # =================================================

            if (
                modality_result
                == "CHEST_XRAY"
            ):

                if (
                    modality_confidence
                    < MODALITY_CONFIDENCE_THRESHOLD
                ):

                    st.error(
                        "❌ Chest X-ray confidence is too low."
                    )

                    st.warning(
                        "The image cannot be reliably "
                        "identified as a chest X-ray."
                    )

                    add_history(
                        f"Rejected - Low modality confidence - "
                        f"{uploaded_file.name}"
                    )

                    st.stop()

                st.success(
                    "✓ Chest X-ray modality detected."
                )

                st.write(
                    f"Chest X-ray confidence: "
                    f"**{modality_confidence * 100:.2f}%**"
                )

            elif (
                modality_result
                == "CT"
            ):

                st.error(
                    "❌ CT scan detected."
                )

                st.warning(
                    "This system accepts chest X-ray "
                    "images only."
                )

                st.write(
                    f"CT confidence: "
                    f"{modality_confidence * 100:.2f}%"
                )

                add_history(
                    f"Rejected - CT - "
                    f"{uploaded_file.name}"
                )

                st.stop()

            elif (
                modality_result
                == "MRI"
            ):

                st.error(
                    "❌ MRI image detected."
                )

                st.warning(
                    "This system accepts chest X-ray "
                    "images only."
                )

                st.write(
                    f"MRI confidence: "
                    f"{modality_confidence * 100:.2f}%"
                )

                add_history(
                    f"Rejected - MRI - "
                    f"{uploaded_file.name}"
                )

                st.stop()

            elif (
                modality_result
                == "OTHER"
            ):

                st.error(
                    "❌ Unsupported medical image detected."
                )

                st.warning(
                    "Please upload a chest X-ray."
                )

                add_history(
                    f"Rejected - Other modality - "
                    f"{uploaded_file.name}"
                )

                st.stop()

            else:

                st.error(
                    "❌ Unknown medical image modality."
                )

                add_history(
                    f"Rejected - Unknown modality - "
                    f"{uploaded_file.name}"
                )

                st.stop()


            # =================================================
            # STEP 3 — X-RAY VERIFICATION
            # =================================================

            st.subheader(
                "Step 3 — Chest X-ray Verification"
            )

            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            verifier_image = cv2.resize(
                image_array,
                XRAY_IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            verifier_image = (
                verifier_image.astype(
                    np.float32
                ) / 255.0
            )

            verifier_input = np.expand_dims(
                verifier_image,
                axis=0
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

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

            # ------------------------------------------------
            # OUTPUT CHECK
            # ------------------------------------------------

            if (
                verifier_prediction.ndim != 2
                or
                verifier_prediction.shape[0] != 1
                or
                verifier_prediction.shape[1] != 2
            ):

                st.error(
                    "Invalid X-ray verifier output."
                )

                st.write(
                    f"Received shape: "
                    f"{verifier_prediction.shape}"
                )

                st.stop()

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            verifier_probabilities = (
                scores_to_probabilities(
                    verifier_prediction[0]
                )
            )

            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

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

            xray_probability = float(
                verifier_probabilities[0]
            )

            non_xray_probability = float(
                verifier_probabilities[1]
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Chest X-ray",
                    f"{xray_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Non-X-ray",
                    f"{non_xray_probability * 100:.2f}%"
                )

            # ------------------------------------------------
            # REJECT NON-X-RAY
            # ------------------------------------------------

            if (
                verifier_result
                != "X-RAY"
                or
                verifier_confidence
                < XRAY_CONFIDENCE_THRESHOLD
            ):

                st.error(
                    "❌ This image failed Chest X-ray verification."
                )

                st.warning(
                    "The pneumonia model will not be "
                    "run on this image."
                )

                add_history(
                    f"Rejected - Non-X-ray - "
                    f"{uploaded_file.name}"
                )

                st.stop()

            st.success(
                "✓ Chest X-ray verified."
            )

            st.write(
                f"X-ray verification confidence: "
                f"**{verifier_confidence * 100:.2f}%**"
            )


            # =================================================
            # STEP 4 — PNEUMONIA DETECTION
            # =================================================

            st.subheader(
                "Step 4 — Pneumonia Detection"
            )

            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            pneumonia_image = cv2.resize(
                image_array,
                PNEUMONIA_IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            pneumonia_image = (
                pneumonia_image.astype(
                    np.float32
                ) / 255.0
            )

            pneumonia_input = np.expand_dims(
                pneumonia_image,
                axis=0
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            pneumonia_prediction = (
                pneumonia_model.predict(
                    pneumonia_input,
                    verbose=0
                )
            )

            pneumonia_prediction = (
                np.asarray(
                    pneumonia_prediction
                )
            )

            # ------------------------------------------------
            # OUTPUT CHECK
            # ------------------------------------------------

            if (
                pneumonia_prediction.ndim != 2
                or
                pneumonia_prediction.shape[0] != 1
                or
                pneumonia_prediction.shape[1] != 2
            ):

                st.error(
                    "Invalid pneumonia model output."
                )

                st.write(
                    f"Received shape: "
                    f"{pneumonia_prediction.shape}"
                )

                st.stop()

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            pneumonia_probabilities = (
                scores_to_probabilities(
                    pneumonia_prediction[0]
                )
            )

            # ------------------------------------------------
            # CLASS PROBABILITIES
            # ------------------------------------------------

            normal_probability = float(
                pneumonia_probabilities[0]
            )

            pneumonia_probability = float(
                pneumonia_probabilities[1]
            )

            # ------------------------------------------------
            # DIAGNOSIS
            # ------------------------------------------------

            diagnosis_index = int(
                np.argmax(
                    pneumonia_probabilities
                )
            )

            diagnosis = (
                PNEUMONIA_CLASS_MAP.get(
                    diagnosis_index,
                    "Unknown"
                )
            )

            diagnosis_confidence = float(
                pneumonia_probabilities[
                    diagnosis_index
                ]
            )

            # =================================================
            # DISPLAY DIAGNOSIS
            # =================================================

            if diagnosis == "Pneumonia":

                st.error(
                    "Diagnosis: Pneumonia"
                )

            elif diagnosis == "Normal":

                st.success(
                    "Diagnosis: Normal"
                )

            else:

                st.warning(
                    "Diagnosis: Unknown"
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

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
                f"**Diagnosis Confidence:** "
                f"{diagnosis_confidence * 100:.2f}%"
            )


            # =================================================
            # STEP 5 — HISTORY
            # =================================================

            add_history(
                f"{diagnosis} - "
                f"{uploaded_file.name}"
            )


            # =================================================
            # STEP 6 — DIAGNOSTIC REPORT
            # =================================================

            st.divider()

            st.subheader(
                "Diagnostic Report"
            )

            st.write(
                f"**File:** {uploaded_file.name}"
            )

            st.write(
                "**Image Modality:** Chest X-ray"
            )

            st.write(
                f"**Modality Confidence:** "
                f"{modality_confidence * 100:.2f}%"
            )

            st.write(
                f"**X-ray Verification:** "
                f"{verifier_result}"
            )

            st.write(
                f"**X-ray Confidence:** "
                f"{verifier_confidence * 100:.2f}%"
            )

            st.write(
                f"**Diagnosis:** {diagnosis}"
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

            # Remove problematic characters.
            clean_filename = (
                clean_filename
                .replace(
                    " ",
                    "_"
                )
            )

            pdf = FPDF()

            pdf.add_page()

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # FILE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # MODALITY
            # ------------------------------------------------

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

            # ------------------------------------------------
            # MODALITY CONFIDENCE
            # ------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
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

            # ------------------------------------------------
            # X-RAY STATUS
            # ------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
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

            # ------------------------------------------------
            # X-RAY CONFIDENCE
            # ------------------------------------------------

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
                f"{verifier_confidence * 100:.2f}%",
                ln=True
            )

            # ------------------------------------------------
            # DIAGNOSIS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # DIAGNOSIS CONFIDENCE
            # ------------------------------------------------

            pdf.set_font(
                "Arial",
                "B",
                12
            )

            pdf.cell(
                55,
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

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            pdf.ln(
                8
            )

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

            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            pdf.ln(
                15
            )

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
                "and does not replace professional medical "
                "diagnosis."
            )

            # ------------------------------------------------
            # PDF BYTES
            # ------------------------------------------------

            pdf_output = pdf.output()

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

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


    # ========================================================
    # IMAGE PROCESSING ERROR
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the uploaded image."
        )

        st.exception(e)
