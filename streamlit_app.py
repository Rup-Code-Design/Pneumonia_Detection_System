# ============================================================
# streamlit_app.py
# Pneumonia Detection System
#
# PIPELINE
#
# Uploaded Image
#       |
#       v
# Basic Image Validation
#       |
#       v
# Grayscale / Color Check
#       |
#       v
# Medical Modality Classifier
#       |
#       +---- CT ------> REJECT
#       |
#       +---- MRI -----> REJECT
#       |
#       +---- OTHER ---> REJECT
#       |
#       +---- CHEST X-RAY
#                    |
#                    v
#             X-ray Verifier
#                    |
#                    v
#             X-ray Confirmed
#                    |
#                    v
#          Pneumonia Classifier
#                    |
#              +-----+-----+
#              |           |
#            Normal    Pneumonia
#
# ============================================================

import os
import io

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
# Always use the directory containing this Python file.
#
# This prevents errors caused by the current working directory
# being different on Streamlit Cloud.
#
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FILE RESOLUTION
# ============================================================

def find_existing_file(
    candidate_names,
    description
):
    """
    Search for a model file in the same directory
    as this Streamlit application.
    """

    for filename in candidate_names:

        path = os.path.join(
            BASE_DIR,
            filename
        )

        if os.path.isfile(path):

            if os.path.getsize(path) > 0:

                return path

    expected = "\n".join(
        os.path.join(BASE_DIR, name)
        for name in candidate_names
    )

    raise FileNotFoundError(
        f"{description} was not found.\n\n"
        f"Checked:\n{expected}\n\n"
        "Make sure the correct model file is committed "
        "to the same GitHub repository as this application."
    )


# ============================================================
# MODEL FILE CANDIDATES
# ============================================================

# ------------------------------------------------------------
# Pneumonia model
# ------------------------------------------------------------
#
# Your repository screenshot showed:
#
# best_xception_pneumonia_model.keras
#
# Your error mentioned:
#
# best_exception_pneumonia_model.keras
#
# Therefore both names are supported.
#
# ------------------------------------------------------------

PNEUMONIA_MODEL_CANDIDATES = [

    "best_xception_pneumonia_model.keras",

    "best_exception_pneumonia_model.keras",

    "best_pneumonia_model.keras"

]


# ------------------------------------------------------------
# X-ray verifier
# ------------------------------------------------------------

XRAY_MODEL_CANDIDATES = [

    # Prefer complete saved model
    "best_xray_verifier.keras",

    # Fall back to weights
    "best_xray_verifier.weights.h5"

]


# ------------------------------------------------------------
# Modality classifier
# ------------------------------------------------------------

MODALITY_MODEL_CANDIDATES = [

    "best_modality_classifier.keras",

    "best_modality_classifier.weights.h5"

]


# ============================================================
# IMAGE SIZES
# ============================================================

XRAY_IMAGE_SIZE = (
    128,
    128
)

MODALITY_IMAGE_SIZE = (
    224,
    224
)

PNEUMONIA_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

# Maximum mean RGB channel difference allowed.
#
# For a grayscale image:
#
# R ≈ G ≈ B
#
# For a color image:
#
# R, G, B differ significantly.
#
# ============================================================

COLOR_TOLERANCE = 5.0


# ============================================================
# X-RAY VERIFIER THRESHOLD
# ============================================================

XRAY_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# MODALITY CLASSIFIER THRESHOLD
# ============================================================

MODALITY_CONFIDENCE_THRESHOLD = 0.90


# ============================================================
# CLASS MAPPINGS
# ============================================================

# ------------------------------------------------------------
# X-RAY VERIFIER
# ------------------------------------------------------------
#
# This must match xray_model_builder.py training.
#
# Current expected mapping:
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
# MODALITY CLASSIFIER
# ------------------------------------------------------------
#
# Expected training mapping:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
# 3 = OTHER
#
# IMPORTANT:
# This MUST match modality_model_builder.py/training code.
#
# ------------------------------------------------------------

MODALITY_CLASS_MAP = {

    0: "CHEST_XRAY",

    1: "CT",

    2: "MRI",

    3: "OTHER"

}


# ------------------------------------------------------------
# PNEUMONIA CLASSIFIER
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
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    model_path = find_existing_file(
        PNEUMONIA_MODEL_CANDIDATES,
        "Pneumonia model"
    )

    # --------------------------------------------------------
    # .keras should normally be a complete saved model.
    # --------------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as complete_model_error:

        # ----------------------------------------------------
        # If the .keras file cannot be loaded as a complete
        # model, try architecture + weights as fallback.
        # ----------------------------------------------------

        try:

            model = build_model(
                input_shape=(
                    PNEUMONIA_IMAGE_SIZE[0],
                    PNEUMONIA_IMAGE_SIZE[1],
                    3
                )
            )

            model.load_weights(
                model_path
            )

            return model

        except Exception as weights_error:

            raise RuntimeError(
                "Pneumonia model could not be loaded.\n\n"
                f"Model file:\n{model_path}\n\n"
                "Complete-model loading error:\n"
                f"{complete_model_error}\n\n"
                "Architecture + weights loading error:\n"
                f"{weights_error}"
            )


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_model():

    model_path = find_existing_file(
        XRAY_MODEL_CANDIDATES,
        "X-ray verifier"
    )

    # --------------------------------------------------------
    # If complete .keras model exists, use it.
    # --------------------------------------------------------

    if model_path.lower().endswith(
        ".keras"
    ):

        try:

            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            return model

        except Exception as complete_model_error:

            raise RuntimeError(
                "The complete X-ray verifier model "
                "could not be loaded.\n\n"
                f"File:\n{model_path}\n\n"
                f"Error:\n{complete_model_error}"
            )

    # --------------------------------------------------------
    # Otherwise load .weights.h5 using architecture.
    # --------------------------------------------------------

    try:

        model = build_xray_classifier(
            input_shape=(
                XRAY_IMAGE_SIZE[0],
                XRAY_IMAGE_SIZE[1],
                3
            )
        )

        model.load_weights(
            model_path
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "X-ray verifier weights could not be loaded.\n\n"
            f"File:\n{model_path}\n\n"
            "This usually means the architecture in "
            "xray_model_builder.py does not exactly match "
            "the architecture used during training.\n\n"
            f"Error:\n{e}"
        )


# ============================================================
# LOAD MODALITY CLASSIFIER
# ============================================================

@st.cache_resource
def load_modality_model():

    model_path = find_existing_file(
        MODALITY_MODEL_CANDIDATES,
        "Medical modality classifier"
    )

    # --------------------------------------------------------
    # COMPLETE MODEL
    # --------------------------------------------------------

    if model_path.lower().endswith(
        ".keras"
    ):

        try:

            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            return model

        except Exception as complete_model_error:

            raise RuntimeError(
                "The complete modality classifier "
                "could not be loaded.\n\n"
                f"File:\n{model_path}\n\n"
                f"Error:\n{complete_model_error}"
            )

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    try:

        model = build_modality_classifier(
            input_shape=(
                MODALITY_IMAGE_SIZE[0],
                MODALITY_IMAGE_SIZE[1],
                3
            ),
            num_classes=4
        )

        model.load_weights(
            model_path
        )

        return model

    except Exception as e:

        raise RuntimeError(
            "Medical modality classifier weights "
            "could not be loaded.\n\n"
            f"File:\n{model_path}\n\n"
            "The architecture generated by "
            "modality_model_builder.py must exactly "
            "match the architecture used during training.\n\n"
            f"Error:\n{e}"
        )


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    pneumonia_model = (
        load_pneumonia_model()
    )

    xray_model = (
        load_xray_model()
    )

    modality_model = (
        load_modality_model()
    )

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MODEL OUTPUT VALIDATION
# ============================================================

def validate_model_output(
    model,
    expected_classes,
    model_name
):

    try:

        output_shape = model.output_shape

        if (
            output_shape is None
            or
            output_shape[-1] != expected_classes
        ):

            raise ValueError(
                f"{model_name} must output "
                f"{expected_classes} classes.\n\n"
                f"Actual output shape: "
                f"{output_shape}"
            )

    except Exception as e:

        raise RuntimeError(
            f"{model_name} output validation failed.\n\n"
            f"{e}"
        )


# Validate all models before allowing uploads.

try:

    validate_model_output(
        modality_model,
        4,
        "Medical modality classifier"
    )

    validate_model_output(
        xray_model,
        2,
        "X-ray verifier"
    )

    validate_model_output(
        pneumonia_model,
        2,
        "Pneumonia classifier"
    )

except Exception as e:

    st.error(
        "Model architecture validation failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_and_normalize(
    image,
    target_size
):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        target_size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# COLOR IMAGE DETECTION
# ============================================================

def check_color_image(
    image
):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )

    r = rgb[:, :, 0]

    g = rgb[:, :, 1]

    b = rgb[:, :, 2]

    rg_difference = np.mean(
        np.abs(r - g)
    )

    gb_difference = np.mean(
        np.abs(g - b)
    )

    rb_difference = np.mean(
        np.abs(r - b)
    )

    mean_difference = (

        rg_difference
        +
        gb_difference
        +
        rb_difference

    ) / 3.0

    is_color = (
        mean_difference
        > COLOR_TOLERANCE
    )

    return (
        is_color,
        float(mean_difference)
    )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_basic_image(
    image
):

    width, height = image.size

    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )

    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    if gray.size == 0:

        return (
            False,
            "Image is empty."
        )

    standard_deviation = (
        np.std(gray)
    )

    if standard_deviation < 8:

        return (
            False,
            "Image appears blank or invalid."
        )

    dark_ratio = np.mean(
        gray < 10
    )

    if dark_ratio > 0.98:

        return (
            False,
            "Image is almost completely black."
        )

    bright_ratio = np.mean(
        gray > 245
    )

    if bright_ratio > 0.98:

        return (
            False,
            "Image is almost completely white."
        )

    return (
        True,
        "Image passed basic validation."
    )


# ============================================================
# CONVERT MODEL OUTPUT TO PROBABILITIES
# ============================================================

def to_probabilities(
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Already softmax probabilities
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
    # Otherwise assume logits.
    # --------------------------------------------------------

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# MODALITY PREDICTION
# ============================================================

def predict_modality(
    image
):

    image_input = (
        resize_and_normalize(
            image,
            MODALITY_IMAGE_SIZE
        )
    )

    prediction = (
        modality_model.predict(
            image_input,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )

    if (

        prediction.ndim != 2

        or

        prediction.shape[1] != 4

    ):

        raise ValueError(
            "Modality classifier must output "
            "exactly 4 classes."
        )

    probabilities = (
        to_probabilities(
            prediction[0]
        )
    )

    class_index = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[class_index]
    )

    class_name = (
        MODALITY_CLASS_MAP.get(
            class_index,
            "UNKNOWN"
        )
    )

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": confidence,
        "probabilities": probabilities
    }


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def verify_xray(
    image
):

    image_input = (
        resize_and_normalize(
            image,
            XRAY_IMAGE_SIZE
        )
    )

    prediction = (
        xray_model.predict(
            image_input,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )

    if (

        prediction.ndim != 2

        or

        prediction.shape[1] != 2

    ):

        raise ValueError(
            "X-ray verifier must output "
            "exactly 2 classes."
        )

    probabilities = (
        to_probabilities(
            prediction[0]
        )
    )

    class_index = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[class_index]
    )

    class_name = (
        XRAY_CLASS_MAP.get(
            class_index,
            "UNKNOWN"
        )
    )

    xray_probability = float(
        probabilities[0]
    )

    non_xray_probability = float(
        probabilities[1]
    )

    is_xray = (

        class_name == "X-RAY"

        and

        xray_probability
        >= XRAY_CONFIDENCE_THRESHOLD

    )

    return {

        "class_index":
            class_index,

        "class_name":
            class_name,

        "confidence":
            confidence,

        "probabilities":
            probabilities,

        "xray_probability":
            xray_probability,

        "non_xray_probability":
            non_xray_probability,

        "is_xray":
            is_xray
    }


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(
    image
):

    image_input = (
        resize_and_normalize(
            image,
            PNEUMONIA_IMAGE_SIZE
        )
    )

    prediction = (
        pneumonia_model.predict(
            image_input,
            verbose=0
        )
    )

    prediction = np.asarray(
        prediction
    )

    if (

        prediction.ndim != 2

        or

        prediction.shape[1] != 2

    ):

        raise ValueError(
            "Pneumonia model must output "
            "exactly 2 classes."
        )

    probabilities = (
        to_probabilities(
            prediction[0]
        )
    )

    normal_probability = float(
        probabilities[0]
    )

    pneumonia_probability = float(
        probabilities[1]
    )

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    diagnosis = (
        PNEUMONIA_CLASS_MAP[
            predicted_index
        ]
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    return {

        "diagnosis":
            diagnosis,

        "confidence":
            confidence,

        "normal_probability":
            normal_probability,

        "pneumonia_probability":
            pneumonia_probability,

        "probabilities":
            probabilities
    }


# ============================================================
# HISTORY HELPER
# ============================================================

def add_history(
    text
):

    if text not in st.session_state.history:

        st.session_state.history.append(
            text
        )

    # Keep only latest 20 entries.

    st.session_state.history = (
        st.session_state.history[-20:]
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    filename,
    modality_confidence,
    xray_confidence,
    diagnosis,
    diagnosis_confidence
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

    pdf.set_font(
        "Arial",
        "B",
        18
    )

    pdf.cell(
        0,
        15,
        "Pneumonia AI Report",
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

    def add_row(
        label,
        value
    ):

        pdf.set_font(
            "Arial",
            "B",
            12
        )

        pdf.cell(
            55,
            10,
            label,
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
            value,
            ln=True
        )

    add_row(
        "File Name:",
        clean_filename
    )

    add_row(
        "Image Modality:",
        "Chest X-ray"
    )

    add_row(
        "Modality Confidence:",
        f"{modality_confidence * 100:.2f}%"
    )

    add_row(
        "X-ray Verification:",
        "X-RAY"
    )

    add_row(
        "X-ray Confidence:",
        f"{xray_confidence * 100:.2f}%"
    )

    add_row(
        "Diagnosis:",
        diagnosis
    )

    add_row(
        "Diagnosis Confidence:",
        f"{diagnosis_confidence * 100:.2f}%"
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
        "Disclaimer: This AI-generated result "
        "is intended for research and educational "
        "purposes only. It does not replace "
        "professional medical diagnosis."
    )

    output = pdf.output()

    if isinstance(
        output,
        bytearray
    ):

        output = bytes(
            output
        )

    elif isinstance(
        output,
        str
    ):

        output = output.encode(
            "latin-1"
        )

    return output


# ============================================================
# HEADER
# ============================================================

st.title(
    "Pneumonia Detection System"
)

st.write(
    "Chest X-ray verification followed by "
    "pneumonia classification."
)

st.info(
    "The system accepts grayscale chest X-ray "
    "images only. Color images, CT scans, MRI "
    "images and unsupported images are rejected."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Pneumonia AI Pipeline"
    )

    st.write(
        """
        Image Upload
        ↓
        Basic Validation
        ↓
        Color Check
        ↓
        Medical Modality Classifier
        ↓
        Chest X-ray
        ↓
        X-ray Verifier
        ↓
        Pneumonia Classifier
        ↓
        Normal / Pneumonia
        """
    )

    st.divider()

    st.write(
        "**Accepted:**"
    )

    st.write(
        "Grayscale Chest X-ray"
    )

    st.write(
        "**Rejected:**"
    )

    st.write(
        "Color images"
    )

    st.write(
        "CT"
    )

    st.write(
        "MRI"
    )

    st.write(
        "Other images"
    )

    st.divider()

    st.write(
        "**Pneumonia classes:**"
    )

    st.write(
        "0 = Normal"
    )

    st.write(
        "1 = Pneumonia"
    )

    st.divider()

    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "Pneumonia model:"
    )

    st.code(
        os.path.basename(
            find_existing_file(
                PNEUMONIA_MODEL_CANDIDATES,
                "Pneumonia model"
            )
        )
    )

    st.write(
        "X-ray verifier:"
    )

    st.code(
        os.path.basename(
            find_existing_file(
                XRAY_MODEL_CANDIDATES,
                "X-ray verifier"
            )
        )
    )

    st.write(
        "Modality classifier:"
    )

    st.code(
        os.path.basename(
            find_existing_file(
                MODALITY_MODEL_CANDIDATES,
                "Medical modality classifier"
            )
        )
    )

    st.write(
        "Input sizes:"
    )

    st.write(
        "Modality: 224 × 224 × 3"
    )

    st.write(
        "X-ray verifier: 128 × 128 × 3"
    )

    st.write(
        "Pneumonia: 224 × 224 × 3"
    )

    st.write(
        "Modality mapping:"
    )

    st.code(
        """
0 = CHEST_XRAY
1 = CT
2 = MRI
3 = OTHER
"""
    )

    st.write(
        "X-ray mapping:"
    )

    st.code(
        """
0 = X-RAY
1 = NON-XRAY
"""
    )

    st.write(
        "Pneumonia mapping:"
    )

    st.code(
        """
0 = Normal
1 = Pneumonia
"""
    )


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
    ],
    help=(
        "Upload a grayscale chest X-ray image."
    )
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        )

        image.load()

        image = image.convert(
            "RGB"
        )

    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True
    ):

        # ====================================================
        # STEP 1 — BASIC VALIDATION
        # ====================================================

        st.subheader(
            "Step 1 — Basic Image Validation"
        )

        valid, validation_message = (
            validate_basic_image(
                image
            )
        )

        if not valid:

            st.error(
                validation_message
            )

            add_history(
                f"Rejected - Invalid image - "
                f"{uploaded_file.name}"
            )

            st.stop()

        st.success(
            validation_message
        )


        # ====================================================
        # STEP 2 — COLOR IMAGE REJECTION
        # ====================================================

        st.subheader(
            "Step 2 — Color Image Verification"
        )

        is_color, color_difference = (
            check_color_image(
                image
            )
        )

        if is_color:

            st.error(
                "This is a color image."
            )

            st.warning(
                "Color images are not accepted. "
                "Please upload a grayscale chest X-ray."
            )

            st.write(
                f"RGB channel difference: "
                f"{color_difference:.2f}"
            )

            add_history(
                f"Rejected - Color image - "
                f"{uploaded_file.name}"
            )

            st.stop()

        st.success(
            "Grayscale image detected."
        )

        st.write(
            f"RGB channel difference: "
            f"{color_difference:.2f}"
        )


        # ====================================================
        # STEP 3 — MEDICAL MODALITY CLASSIFICATION
        # ====================================================

        st.subheader(
            "Step 3 — Medical Image Modality Verification"
        )

        with st.spinner(
            "Determining image modality..."
        ):

            try:

                modality_result = (
                    predict_modality(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "Medical modality classification failed."
                )

                st.exception(e)

                st.stop()


        modality_name = (
            modality_result[
                "class_name"
            ]
        )

        modality_confidence = (
            modality_result[
                "confidence"
            ]
        )

        modality_probabilities = (
            modality_result[
                "probabilities"
            ]
        )


        # ----------------------------------------------------
        # DISPLAY MODALITY PROBABILITIES
        # ----------------------------------------------------

        st.write(
            f"**Detected modality:** "
            f"{modality_name}"
        )

        st.write(
            f"**Confidence:** "
            f"{modality_confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # MODALITY REJECTION
        # ----------------------------------------------------

        if modality_name != "CHEST_XRAY":

            if modality_name == "CT":

                st.error(
                    "CT scan detected."
                )

                add_history(
                    f"Rejected - CT - "
                    f"{uploaded_file.name}"
                )

            elif modality_name == "MRI":

                st.error(
                    "MRI image detected."
                )

                add_history(
                    f"Rejected - MRI - "
                    f"{uploaded_file.name}"
                )

            else:

                st.error(
                    "Unsupported image modality detected."
                )

                add_history(
                    f"Rejected - Other modality - "
                    f"{uploaded_file.name}"
                )

            st.warning(
                "This system accepts only "
                "grayscale chest X-ray images."
            )

            st.stop()


        # ----------------------------------------------------
        # LOW CHEST X-RAY CONFIDENCE
        # ----------------------------------------------------

        if (
            modality_confidence
            < MODALITY_CONFIDENCE_THRESHOLD
        ):

            st.error(
                "Chest X-ray confidence is too low."
            )

            st.write(
                f"Required confidence: "
                f"{MODALITY_CONFIDENCE_THRESHOLD * 100:.0f}%"
            )

            st.write(
                f"Detected confidence: "
                f"{modality_confidence * 100:.2f}%"
            )

            add_history(
                f"Rejected - Low modality confidence - "
                f"{uploaded_file.name}"
            )

            st.stop()


        # ----------------------------------------------------
        # CHEST X-RAY CONFIRMED
        # ----------------------------------------------------

        st.success(
            "Chest X-ray modality confirmed."
        )


        # ====================================================
        # STEP 4 — X-RAY VERIFIER
        # ====================================================

        st.subheader(
            "Step 4 — Chest X-ray Verification"
        )

        with st.spinner(
            "Verifying chest X-ray..."
        ):

            try:

                xray_result = (
                    verify_xray(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "X-ray verification failed."
                )

                st.exception(e)

                st.stop()


        xray_probability = (
            xray_result[
                "xray_probability"
            ]
        )

        non_xray_probability = (
            xray_result[
                "non_xray_probability"
            ]
        )

        xray_confidence = (
            xray_result[
                "confidence"
            ]
        )

        xray_is_valid = (
            xray_result[
                "is_xray"
            ]
        )


        # ----------------------------------------------------
        # X-RAY PROBABILITIES
        # ----------------------------------------------------

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


        st.progress(
            float(
                np.clip(
                    xray_probability,
                    0.0,
                    1.0
                )
            )
        )


        # ----------------------------------------------------
        # X-RAY REJECTION
        # ----------------------------------------------------

        if not xray_is_valid:

            st.error(
                "This image failed X-ray verification."
            )

            st.warning(
                "Pneumonia detection has been stopped."
            )

            add_history(
                f"Rejected - X-ray verification - "
                f"{uploaded_file.name}"
            )

            st.stop()


        # ----------------------------------------------------
        # X-RAY VERIFIED
        # ----------------------------------------------------

        st.success(
            "Chest X-ray verified successfully."
        )

        st.write(
            f"X-ray confidence: "
            f"{xray_confidence * 100:.2f}%"
        )


        # ====================================================
        # STEP 5 — PNEUMONIA DETECTION
        # ====================================================

        st.subheader(
            "Step 5 — Pneumonia Detection"
        )

        with st.spinner(
            "Analyzing verified chest X-ray..."
        ):

            try:

                pneumonia_result = (
                    predict_pneumonia(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "Pneumonia prediction failed."
                )

                st.exception(e)

                st.stop()


        diagnosis = (
            pneumonia_result[
                "diagnosis"
            ]
        )

        diagnosis_confidence = (
            pneumonia_result[
                "confidence"
            ]
        )

        normal_probability = (
            pneumonia_result[
                "normal_probability"
            ]
        )

        pneumonia_probability = (
            pneumonia_result[
                "pneumonia_probability"
            ]
        )


        # ====================================================
        # DIAGNOSIS DISPLAY
        # ====================================================

        if diagnosis == "Pneumonia":

            st.error(
                "Diagnosis: Pneumonia"
            )

        else:

            st.success(
                "Diagnosis: Normal"
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.subheader(
            "Final Result"
        )

        st.write(
            f"**Diagnosis:** {diagnosis}"
        )

        st.write(
            f"**Diagnosis Confidence:** "
            f"{diagnosis_confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Normal",
                f"{normal_probability * 100:.2f}%"
            )

            st.progress(
                float(
                    np.clip(
                        normal_probability,
                        0.0,
                        1.0
                    )
                )
            )

        with col2:

            st.metric(
                "Pneumonia",
                f"{pneumonia_probability * 100:.2f}%"
            )

            st.progress(
                float(
                    np.clip(
                        pneumonia_probability,
                        0.0,
                        1.0
                    )
                )
            )


        # ====================================================
        # HISTORY
        # ====================================================

        add_history(
            f"{diagnosis} - "
            f"{uploaded_file.name}"
        )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "Technical Details"
        ):

            st.write(
                f"**Image modality:** "
                f"{modality_name}"
            )

            st.write(
                f"**Modality confidence:** "
                f"{modality_confidence * 100:.2f}%"
            )

            st.write(
                f"**X-ray probability:** "
                f"{xray_probability * 100:.2f}%"
            )

            st.write(
                f"**Non-X-ray probability:** "
                f"{non_xray_probability * 100:.2f}%"
            )

            st.write(
                f"**X-ray threshold:** "
                f"{XRAY_CONFIDENCE_THRESHOLD * 100:.0f}%"
            )

            st.write(
                f"**Diagnosis:** "
                f"{diagnosis}"
            )

            st.write(
                f"**Normal probability:** "
                f"{normal_probability * 100:.2f}%"
            )

            st.write(
                f"**Pneumonia probability:** "
                f"{pneumonia_probability * 100:.2f}%"
            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        st.divider()

        st.subheader(
            "Diagnostic Report"
        )

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Modality:** Chest X-ray"
        )

        st.write(
            f"**Modality confidence:** "
            f"{modality_confidence * 100:.2f}%"
        )

        st.write(
            f"**X-ray verification:** "
            f"X-RAY"
        )

        st.write(
            f"**X-ray confidence:** "
            f"{xray_confidence * 100:.2f}%"
        )

        st.write(
            f"**Diagnosis:** "
            f"{diagnosis}"
        )

        st.write(
            f"**Diagnosis confidence:** "
            f"{diagnosis_confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # GENERATE PDF
        # ----------------------------------------------------

        try:

            pdf_data = create_pdf_report(

                filename=uploaded_file.name,

                modality_confidence=(
                    modality_confidence
                ),

                xray_confidence=(
                    xray_confidence
                ),

                diagnosis=(
                    diagnosis
                ),

                diagnosis_confidence=(
                    diagnosis_confidence
                )
            )

            clean_filename = (
                uploaded_file.name
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    ".",
                    "_"
                )
            )

            st.download_button(

                label="Download Diagnostic Report",

                data=pdf_data,

                file_name=(
                    f"Report_{clean_filename}.pdf"
                ),

                mime="application/pdf"
            )

        except Exception as e:

            st.warning(
                "PDF report could not be generated."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype for educational and "
    "research purposes. This system is not "
    "intended to provide clinical diagnosis."
)
