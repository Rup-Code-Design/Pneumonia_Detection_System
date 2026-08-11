import os
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from fpdf import FPDF

from model_builder import build_model
from xray_model_builder import build_xray_classifier


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

XRAY_MODEL_PATH = "best_xray_verifier.weights.h5"
PNEUMONIA_MODEL_PATH = "best_xception_model.keras"


# ============================================================
# LOAD X-RAY VERIFIER
# ============================================================

@st.cache_resource
def load_xray_verifier():

    if not os.path.exists(XRAY_MODEL_PATH):
        raise FileNotFoundError(
            f"X-ray verifier weights not found: {XRAY_MODEL_PATH}"
        )

    model = build_xray_classifier(
        input_shape=(128, 128, 3)
    )

    model.load_weights(XRAY_MODEL_PATH)

    return model


# ============================================================
# LOAD PNEUMONIA MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    if not os.path.exists(PNEUMONIA_MODEL_PATH):
        raise FileNotFoundError(
            f"Pneumonia model not found: {PNEUMONIA_MODEL_PATH}"
        )

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=1
    )

    model.load_weights(PNEUMONIA_MODEL_PATH)

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    xray_model = load_xray_verifier()
    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error("Model loading failed.")
    st.exception(e)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🫁 Pneumonia Detection System")

st.write(
    "Upload an image. The system first verifies whether "
    "the image is a chest X-ray and then performs pneumonia detection."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Analyze Image", type="primary"):

        with st.spinner("Analyzing image..."):

            # ------------------------------------------------
            # Convert image
            # ------------------------------------------------

            img_array = np.array(image)

            # ------------------------------------------------
            # X-RAY VERIFIER
            # ------------------------------------------------

            verifier_img = cv2.resize(
                img_array,
                (128, 128)
            )

            verifier_img = (
                verifier_img.astype(np.float32) / 255.0
            )

            verifier_input = np.expand_dims(
                verifier_img,
                axis=0
            )

            xray_prediction = xray_model.predict(
                verifier_input,
                verbose=0
            )

            xray_class = int(
                np.argmax(xray_prediction[0])
            )

            xray_confidence = float(
                np.max(xray_prediction[0])
            )

            # ------------------------------------------------
            # CLASS CHECK
            #
            # IMPORTANT:
            # Your verifier training generator must have:
            #
            # NORMAL / PNEUMONIA -> X-RAY
            # NON_XRAY           -> NON-X-RAY
            #
            # Change these indices if your training class_indices
            # are different.
            # ------------------------------------------------

            XRAY_CLASSES = {
                0: "X-RAY",
                1: "NON-X-RAY"
            }

            predicted_type = XRAY_CLASSES.get(
                xray_class,
                "UNKNOWN"
            )

            # ------------------------------------------------
            # NON-X-RAY
            # ------------------------------------------------

            if predicted_type == "NON-X-RAY":

                st.error(
                    "❌ This is not a Chest X-ray image."
                )

                st.write(
                    f"Verifier confidence: "
                    f"{xray_confidence * 100:.2f}%"
                )

                st.stop()

            # ------------------------------------------------
            # X-RAY CONFIRMED
            # ------------------------------------------------

            st.success(
                f"✅ Chest X-ray detected "
                f"({xray_confidence * 100:.2f}%)"
            )

            # ------------------------------------------------
            # PNEUMONIA MODEL
            # ------------------------------------------------

            pneumonia_img = cv2.resize(
                img_array,
                (224, 224)
            )

            pneumonia_img = (
                pneumonia_img.astype(np.float32) / 255.0
            )

            pneumonia_input = np.expand_dims(
                pneumonia_img,
                axis=0
            )

            prediction = pneumonia_model.predict(
                pneumonia_input,
                verbose=0
            )

            # ------------------------------------------------
            # IMPORTANT:
            # This assumes the model output is ONE sigmoid
            # probability:
            #
            # 0 = Normal
            # 1 = Pneumonia
            # ------------------------------------------------

            pneumonia_probability = float(
                np.squeeze(prediction)
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if pneumonia_probability >= 0.5:

                diagnosis = "Pneumonia"

                st.error(
                    f"Diagnosis: {diagnosis}"
                )

                st.write(
                    f"Pneumonia probability: "
                    f"{pneumonia_probability * 100:.2f}%"
                )

            else:

                diagnosis = "Normal"

                st.success(
                    f"Diagnosis: {diagnosis}"
                )

                st.write(
                    f"Normal probability: "
                    f"{(1 - pneumonia_probability) * 100:.2f}%"
                )

            st.warning(
                "This system is for research purposes only "
                "and does not replace professional radiological diagnosis."
            )
