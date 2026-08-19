==============================
# MODEL FILE VALIDATION
# ============================================================

def validate_model_file(path, model_name):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"""
{model_name} was not found.

Expected location:
{path}

Make sure the file is committed to the same
GitHub repository as streamlit_app.py.
"""
        )

    if os.path.getsize(path) == 0:

        raise ValueError(
            f"{model_name} exists but is empty:\n{path}"
        )


# ============================================================
# GELU COMPATIBILITY
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="PneuX-ModNet"
)
def gelu(x):

    return tf.nn.gelu(x)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_modality_model():

    validate_model_file(
        MODALITY_MODEL_PATH,
        "Modality classifier"
    )

    try:

        model = tf.keras.models.load_model(
            MODALITY_MODEL_PATH,
            compile=False,
            custom_objects={
                "gelu": gelu,
                "GELU": gelu
            }
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load modality_classifier.keras.\n\n"
            f"Original error:\n{e}"
        ) from e

    if model.output_shape[-1] != 3:

        raise ValueError(
            "The modality classifier must have "
            "exactly 3 output classes.\n"
            f"Received: {model.output_shape}"
        )

    if model.input_shape[-1] != 3:

        raise ValueError(
            "The modality classifier must accept "
            "3-channel RGB input.\n"
            f"Received: {model.input_shape}"
        )

    return model


@st.cache_resource
def load_xray_verifier():

    validate_model_file(
        XRAY_VERIFIER_PATH,
        "X-ray verifier"
    )

    try:

        model = tf.keras.models.load_model(
            XRAY_VERIFIER_PATH,
            compile=False,
            custom_objects={
                "gelu": gelu,
                "GELU": gelu
            }
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load xray_verifier.keras.\n\n"
            f"Original error:\n{e}"
        ) from e

    return model


@st.cache_resource
def load_pneumonia_model():

    validate_model_file(
        PNEUMONIA_MODEL_PATH,
        "Pneumonia model"
    )

    try:

        model = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False,
            custom_objects={
                "gelu": gelu,
                "GELU": gelu
            }
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load "
            "best_xception_pneumonia_model.keras.\n\n"
            f"Original error:\n{e}"
        ) from e

    return model


# ============================================================
# LOAD MODELS
# ============================================================

try:

    modality_model = load_modality_model()

    xray_verifier_model = load_xray_verifier()

    pneumonia_model = load_pneumonia_model()

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ============================================================
# PNEUMONIA OUTPUT VALIDATION
# ============================================================

if pneumonia_model.output_shape[-1] != 2:

    st.error(
        "The pneumonia model must have exactly "
        "2 output classes."
    )

    st.write(
        f"Received output shape: "
        f"{pneumonia_model.output_shape}"
    )

    st.stop()


# ============================================================
# OUTPUT PROBABILITIES
# ============================================================

def convert_to_probabilities(scores):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

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

    return tf.nn.softmax(
        scores
    ).numpy()


# ============================================================
# COLOR IMAGE CHECK
# ============================================================

def check_color_image(image):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )

    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]

    rg_difference = np.mean(
        np.abs(red - green)
    )

    gb_difference = np.mean(
        np.abs(green - blue)
    )

    rb_difference = np.mean(
        np.abs(red - blue)
    )

    average_difference = (
        rg_difference
        +
        gb_difference
        +
        rb_difference
    ) / 3.0

    is_color = (
        average_difference
        >
        COLOR_TOLERANCE
    )

    return (
        is_color,
        float(average_difference)
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image):

    width, height = image.size

    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )

    array = np.asarray(image)

    if array.size == 0:

        return (
            False,
            "Image is empty."
        )

    is_color, difference = (
        check_color_image(image)
    )

    if is_color:

        return (
            False,
            "Color image detected. Please input grayscale medical image."
        )

    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    if np.std(gray) < 8:

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
        "Image passed validation."
    )


# ============================================================
# GENERAL PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    target_size
):

    image = ImageOps.exif_transpose(
        image
    )

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
# MODALITY PREPROCESSING
# ============================================================

def preprocess_modality_image(image):

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        MODALITY_IMAGE_SIZE,
        Image.Resampling.NEAREST
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
# MODALITY PREDICTION
# ============================================================

def predict_modality(image):

    image_array = preprocess_modality_image(
        image
    )

    prediction = modality_model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )

    if (
        prediction.ndim != 2
        or
        prediction.shape[1] != 3
    ):

        raise ValueError(
            "Modality classifier must output "
            f"3 classes. Received: {prediction.shape}"
        )

    probabilities = convert_to_probabilities(
        prediction[0]
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    modality = MODALITY_CLASS_MAP[
        predicted_index
    ]

    return {
        "index": predicted_index,
        "class": modality,
        "confidence": confidence,
        "probabilities": probabilities
    }


# ============================================================
# X-RAY VERIFICATION
# ============================================================

def predict_xray_verification(image):

    image_array = preprocess_image(
        image,
        XRAY_VERIFIER_IMAGE_SIZE
    )

    prediction = xray_verifier_model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )

    if prediction.shape[-1] == 1:

        probability = float(
            prediction[0][0]
        )

        if (
            probability < 0.0
            or
            probability > 1.0
        ):

            probability = float(
                tf.sigmoid(
                    prediction[0][0]
                ).numpy()
            )

        is_xray = (
            probability
            >=
            XRAY_VERIFIER_THRESHOLD
        )

        confidence = (
            probability
            if is_xray
            else
            1.0 - probability
        )

        return {
            "is_xray": is_xray,
            "confidence": float(confidence),
            "xray_probability": probability,
            "non_xray_probability": 1.0 - probability
        }

    if prediction.shape[-1] == 2:

        probabilities = convert_to_probabilities(
            prediction[0]
        )

        non_xray_probability = float(
            probabilities[0]
        )

        xray_probability = float(
            probabilities[1]
        )

        is_xray = (
            xray_probability
            >=
            XRAY_VERIFIER_THRESHOLD
        )

        confidence = (
            xray_probability
            if is_xray
            else
            non_xray_probability
        )

        return {
            "is_xray": is_xray,
            "confidence": float(confidence),
            "xray_probability": xray_probability,
            "non_xray_probability": non_xray_probability
        }

    raise ValueError(
        "Unexpected X-ray verifier output shape: "
        f"{prediction.shape}"
    )


# ============================================================
# PNEUMONIA PREDICTION
# ============================================================

def predict_pneumonia(image):

    image_array = preprocess_image(
        image,
        PNEUMONIA_IMAGE_SIZE
    )

    prediction = pneumonia_model.predict(
        image_array,
        verbose=0
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
            f"2 classes. Received: {prediction.shape}"
        )

    probabilities = convert_to_probabilities(
        prediction[0]
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = PNEUMONIA_CLASS_MAP[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "normal_probability": float(
            probabilities[0]
        ),
        "pneumonia_probability": float(
            probabilities[1]
        ),
        "probabilities": probabilities
    }


# ============================================================
# ============================================================
# GRAD-CAM++ — GENERIC CNN IMPLEMENTATION
# ============================================================
#
# IMPORTANT:
#
# The pneumonia model saved in
# best_xception_pneumonia_model.keras is NOT required to contain
# a nested Xception model. Grad-CAM++ works directly with the
# actual computational graph of the loaded pneumonia model.
#
# This implementation therefore:
#   1. Searches the loaded pneumonia model for a usable 4-D
#      convolutional feature layer.
#   2. Builds the Grad-CAM graph directly from pneumonia_model.input.
#   3. Computes a numerically stable Grad-CAM++ approximation.
#   4. Does NOT assume a nested Xception object.
#
# This removes the crash:
# "The pneumonia model does not contain a nested Xception model."
# ============================================================


def _safe_rank_from_shape(shape):
    if shape is None:
        return None

    try:
        if shape.rank is not None:
            return int(shape.rank)
    except Exception:
        pass

    try:
        if hasattr(shape, "as_list"):
            return len(shape.as_list())
    except Exception:
        pass

    try:
        return len(shape)
    except Exception:
        return None


# ============================================================
# PRIMARY TENSOR HELPER
# ============================================================

def _primary_tensor(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _primary_tensor(value[0])
    return value


# ============================================================
# CHECK 4-D OUTPUT
# ============================================================

def _layer_has_4d_output(layer):
    try:
        output = _primary_tensor(layer.output)
        if output is None:
            return False
        return _safe_rank_from_shape(output.shape) == 4
    except Exception:
        return False


# ============================================================
# GRAD-CAM LAYER SCORE
# ============================================================

def _gradcam_layer_score(layer, index):
    class_name = getattr(layer.__class__, "__name__", "").lower()
    layer_name = getattr(layer, "name", "").lower()
    combined = class_name + " " + layer_name

    score = float(index)

    # Prefer convolutional feature layers.
    if "separableconv2d" in class_name:
        score += 6000.0
    elif "depthwiseconv2d" in class_name:
        score += 5500.0
    elif "conv2d" in class_name:
        score += 5000.0
    elif "conv" in class_name:
        score += 4000.0

    # Xception-style naming gets a small preference, but is NOT required.
    if "sepconv" in combined:
        score += 500.0

    # Prefer late feature blocks.
    match = re.search(r"block(\d+)", combined)
    if match:
        try:
            score += int(match.group(1)) * 100.0
        except Exception:
            pass

    # Avoid classification layers.
    if any(
        keyword in combined
        for keyword in (
            "flatten",
            "globalaveragepool",
            "globalmaxpool",
            "dense",
            "dropout",
            "softmax",
            "activation_softmax"
        )
    ):
        score -= 10000.0

    return score


# ============================================================
# FIND USABLE GRAD-CAM TARGET LAYER
# ============================================================

def find_gradcam_target_layer(model):
    """
    Find the best 4-D feature layer directly inside the loaded
    pneumonia model.

    We intentionally prefer direct layers of the loaded model because
    their tensors are guaranteed to belong to the model's input/output
    graph. No nested-Xception assumption is made.
    """

    try:
        layers = list(model.layers)
    except Exception:
        layers = []

    candidates = []

    for index, layer in enumerate(layers):
        if not _layer_has_4d_output(layer):
            continue

        class_name = getattr(layer.__class__, "__name__", "").lower()
        combined = (
            class_name
            + " "
            + getattr(layer, "name", "").lower()
        )

        # First choice: actual convolutional feature layers.
        if any(
            keyword in class_name
            for keyword in (
                "separableconv2d",
                "depthwiseconv2d",
                "conv2d",
                "conv"
            )
        ):
            score = _gradcam_layer_score(layer, index)
            candidates.append((score, index, layer))

    # If there are no direct convolution layers, allow any direct 4-D
    # activation layer. This keeps Grad-CAM usable with custom CNN blocks.
    if not candidates:
        for index, layer in enumerate(layers):
            if not _layer_has_4d_output(layer):
                continue

            combined = (
                getattr(layer.__class__, "__name__", "").lower()
                + " "
                + getattr(layer, "name", "").lower()
            )

            if any(
                keyword in combined
                for keyword in (
                    "pool",
                    "flatten",
                    "dense",
                    "dropout"
                )
            ):
                continue

            candidates.append((float(index), index, layer))

    if not candidates:
        raise RuntimeError(
            "No usable 4-D convolutional feature layer was found "
            "inside the loaded pneumonia model. Grad-CAM++ cannot "
            "be generated for this model architecture."
        )

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True
    )

    return candidates[0][2]


# ============================================================
# BUILD DIRECT GRAD-CAM MODEL
# ============================================================

def build_gradcam_model(model, target_layer):
    """
    Build a standard Grad-CAM model directly from the actual
    pneumonia model graph.
    """

    target_output = _primary_tensor(target_layer.output)

    if target_output is None:
        raise RuntimeError(
            "The selected Grad-CAM++ target layer has no usable output."
        )

    if _safe_rank_from_shape(target_output.shape) != 4:
        raise RuntimeError(
            "The selected Grad-CAM++ target layer must produce a "
            f"4-D feature map. Received: {target_output.shape}"
        )

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_output, model.output],
            name="PneuX_GradCAM_PlusPlus_Model"
        )
    except Exception as e:
        raise RuntimeError(
            "Could not connect the selected Grad-CAM++ feature layer "
            "to the loaded pneumonia model input/output graph.\n\n"
            f"Target layer: {getattr(target_layer, 'name', 'unknown')}\n"
            f"Original error: {e}"
        ) from e

    return grad_model


# ============================================================
# PREPARE GRAD-CAM COMPONENTS
# ============================================================

@st.cache_resource

def prepare_gradcam_components():
    """
    Prepare Grad-CAM++ using the actual loaded pneumonia model.

    No nested Xception model is required or searched for.
    """

    target_layer = find_gradcam_target_layer(
        pneumonia_model
    )

    grad_model = build_gradcam_model(
        pneumonia_model,
        target_layer
    )

    # --------------------------------------------------------
    # Architecture smoke test.
    # --------------------------------------------------------
    test_input = tf.zeros(
        [
            1,
            PNEUMONIA_IMAGE_SIZE[0],
            PNEUMONIA_IMAGE_SIZE[1],
            3
        ],
        dtype=tf.float32
    )

    try:
        test_features, test_prediction = grad_model(
            test_input,
            training=False
        )
    except Exception as e:
        raise RuntimeError(
            "The Grad-CAM++ model could not process a "
            "224x224 RGB image.\n\n"
            f"Target layer: {getattr(target_layer, 'name', 'unknown')}\n"
            f"Original error: {e}"
        ) from e

    if _safe_rank_from_shape(test_features.shape) != 4:
        raise RuntimeError(
            "The selected Grad-CAM++ feature layer does not produce "
            f"a 4-D tensor. Received: {test_features.shape}"
        )

    if _safe_rank_from_shape(test_prediction.shape) != 2:
        raise RuntimeError(
            "The pneumonia model prediction tensor must be 2-D. "
            f"Received: {test_prediction.shape}"
        )

    if test_prediction.shape[-1] != pneumonia_model.output_shape[-1]:
        raise RuntimeError(
            "The Grad-CAM++ prediction output does not match the "
            "loaded pneumonia model output.\n"
            f"Expected: {pneumonia_model.output_shape[-1]}\n"
            f"Received: {test_prediction.shape[-1]}"
        )

    return {
        "target_layer": target_layer,
        "grad_model": grad_model
    }


# ============================================================
# GRAD-CAM++ CALCULATION
# ============================================================

def calculate_gradcam_plus_plus(
    conv_features,
    gradients
):
    """
    Numerically stable Grad-CAM++ approximation.

    The implementation uses the first derivative and the standard
    Grad-CAM++ alpha approximation. This avoids fragile second- and
    third-order GradientTape calls that frequently become None with
    modern TensorFlow/Keras graphs.
    """

    conv_features = tf.cast(
        conv_features,
        tf.float32
    )

    gradients = tf.cast(
        gradients,
        tf.float32
    )

    epsilon = tf.constant(
        GRADCAM_EPSILON,
        dtype=tf.float32
    )

    gradients = tf.where(
        tf.math.is_finite(gradients),
        gradients,
        tf.zeros_like(gradients)
    )

    conv_features = tf.where(
        tf.math.is_finite(conv_features),
        conv_features,
        tf.zeros_like(conv_features)
    )

    positive_gradients = tf.maximum(
        gradients,
        0.0
    )

    gradients_squared = tf.square(
        gradients
    )

    gradients_cubed = (
        gradients_squared
        * gradients
    )

    spatial_sum = tf.reduce_sum(
        conv_features
        * gradients_cubed,
        axis=(1, 2),
        keepdims=True
    )

    denominator = (
        2.0 * gradients_squared
        + spatial_sum
    )

    denominator = tf.where(
        tf.abs(denominator) > epsilon,
        denominator,
        tf.ones_like(denominator) * epsilon
    )

    alpha = (
        gradients_squared
        / denominator
    )

    alpha = tf.where(
        tf.math.is_finite(alpha),
        alpha,
        tf.zeros_like(alpha)
    )

    weights = tf.reduce_sum(
        alpha * positive_gradients,
        axis=(1, 2)
    )

    weights = tf.where(
        tf.math.is_finite(weights),
        weights,
        tf.zeros_like(weights)
    )

    weighted_features = (
        conv_features
        * weights[:, tf.newaxis, tf.newaxis, :]
    )

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0.0
    )

    heatmap = heatmap[0]

    heatmap = tf.where(
        tf.math.is_finite(heatmap),
        heatmap,
        tf.zeros_like(heatmap)
    )

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > epsilon,
        heatmap / (maximum + epsilon),
        tf.zeros_like(heatmap)
    )

    heatmap = heatmap.numpy()

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return np.clip(
        heatmap,
        0.0,
        1.0
    )


# ============================================================
# GENERATE GRAD-CAM++
# ============================================================

def generate_gradcam_plus_plus(
    image,
    target_class_index=1
):

    components = prepare_gradcam_components()

    target_layer = components["target_layer"]
    grad_model = components["grad_model"]

    target_layer_name = getattr(
        target_layer,
        "name",
        "unknown"
    )

    # --------------------------------------------------------
    # Prepare image.
    # --------------------------------------------------------
    image_array = preprocess_image(
        image,
        PNEUMONIA_IMAGE_SIZE
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Direct gradient graph:
    #
    # image
    #   ↓
    # loaded pneumonia model
    #   ↓
    # target convolutional feature map
    #   ↓
    # pneumonia class score
    #
    # No nested Xception reconstruction is used.
    # --------------------------------------------------------
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)

        conv_features, predictions = grad_model(
            image_tensor,
            training=False
        )

        predictions = tf.convert_to_tensor(
            predictions
        )

        if _safe_rank_from_shape(predictions.shape) != 2:
            raise RuntimeError(
                "Pneumonia classifier returned an invalid prediction "
                f"shape during Grad-CAM++. Received: {predictions.shape}"
            )

        if target_class_index < 0 or target_class_index >= int(
            predictions.shape[-1]
        ):
            raise ValueError(
                "Invalid Grad-CAM++ target class index.\n"
                f"Target index: {target_class_index}\n"
                f"Available classes: {predictions.shape[-1]}"
            )

        class_score = predictions[
            :, target_class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_features
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM++ could not calculate gradients between the "
            "pneumonia prediction and the selected feature map.\n\n"
            f"Target layer: {target_layer_name}"
        )

    if _safe_rank_from_shape(conv_features.shape) != 4:
        raise RuntimeError(
            "Grad-CAM++ feature map is not 4-D.\n"
            f"Received: {conv_features.shape}\n"
            f"Target layer: {target_layer_name}"
        )

    # --------------------------------------------------------
    # Calculate Grad-CAM++.
    # --------------------------------------------------------
    heatmap = calculate_gradcam_plus_plus(
        conv_features,
        gradients
    )

    if heatmap.size == 0:
        raise RuntimeError(
            "Grad-CAM++ produced an empty heatmap."
        )

    if not np.isfinite(heatmap).any():
        raise RuntimeError(
            "Grad-CAM++ produced an invalid heatmap."
        )

    maximum = float(
        np.max(heatmap)
    )

    if maximum <= 1e-8:
        raise RuntimeError(
            "Grad-CAM++ produced an almost-zero localization map.\n\n"
            f"Target layer: {target_layer_name}\n\n"
            "The model returned no positive gradient signal for the "
            "selected feature layer."
        )

    heatmap /= (
        maximum
        + GRADCAM_EPSILON
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    return (
        heatmap,
        target_layer_name
    )


# ============================================================
# HEATMAP ENHANCEMENT
# ============================================================

# ============================================================

def enhance_gradcam_heatmap(
    heatmap
):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    positive_values = heatmap[
        heatmap > 0
    ]

    if positive_values.size < 10:

        return heatmap

    low_value = float(
        np.percentile(
            positive_values,
            GRADCAM_LOW_PERCENTILE
        )
    )

    high_value = float(
        np.percentile(
            positive_values,
            GRADCAM_HIGH_PERCENTILE
        )
    )

    maximum = float(
        np.max(heatmap)
    )

    if (
        high_value
        <=
        low_value
        +
        GRADCAM_EPSILON
    ):

        low_value = 0.0

        high_value = maximum

    enhanced = (
        heatmap
        -
        low_value
    ) / (
        high_value
        -
        low_value
        +
        GRADCAM_EPSILON
    )

    enhanced = np.clip(
        enhanced,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Gamma correction.
    # --------------------------------------------------------

    enhanced = np.power(
        enhanced,
        GRADCAM_GAMMA
    )

    # --------------------------------------------------------
    # Remove weak activation.
    # --------------------------------------------------------

    enhanced[
        enhanced
        <
        GRADCAM_MIN_ACTIVATION
    ] = 0.0

    # --------------------------------------------------------
    # Final normalization.
    # --------------------------------------------------------

    final_max = float(
        np.max(enhanced)
    )

    if final_max > GRADCAM_EPSILON:

        enhanced /= (
            final_max
            +
            GRADCAM_EPSILON
        )

    return np.clip(
        enhanced,
        0.0,
        1.0
    )


# ============================================================
# COLORIZE HEATMAP
# ============================================================

def colorize_gradcam_heatmap(
    heatmap
):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Blue → Cyan → Green → Yellow → Red
    # --------------------------------------------------------

    hue = (
        0.66
        *
        (
            1.0
            -
            heatmap
        )
    )

    saturation = np.ones_like(
        heatmap
    )

    value = np.ones_like(
        heatmap
    )

    h = hue * 6.0

    sector = np.floor(
        h
    ).astype(
        np.int32
    )

    fraction = (
        h
        -
        np.floor(h)
    )

    p = (
        value
        *
        (
            1.0
            -
            saturation
        )
    )

    q = (
        value
        *
        (
            1.0
            -
            saturation
            *
            fraction
        )
    )

    t = (
        value
        *
        (
            1.0
            -
            saturation
            *
            (
                1.0
                -
                fraction
            )
        )
    )

    r = np.zeros_like(
        heatmap
    )

    g = np.zeros_like(
        heatmap
    )

    b = np.zeros_like(
        heatmap
    )

    mask = sector == 0

    r[mask] = value[mask]
    g[mask] = t[mask]
    b[mask] = p[mask]

    mask = sector == 1

    r[mask] = q[mask]
    g[mask] = value[mask]
    b[mask] = p[mask]

    mask = sector == 2

    r[mask] = p[mask]
    g[mask] = value[mask]
    b[mask] = t[mask]

    mask = sector == 3

    r[mask] = p[mask]
    g[mask] = q[mask]
    b[mask] = value[mask]

    mask = sector == 4

    r[mask] = t[mask]
    g[mask] = p[mask]
    b[mask] = value[mask]

    mask = sector >= 5

    r[mask] = value[mask]
    g[mask] = p[mask]
    b[mask] = q[mask]

    rgb = np.stack(
        [
            r,
            g,
            b
        ],
        axis=-1
    )

    rgb = (
        np.clip(
            rgb,
            0.0,
            1.0
        )
        *
        255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb,
        mode="RGB"
    )


# ============================================================
# CREATE GRAD-CAM++ OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):

    original = (
        ImageOps.exif_transpose(
            image
        )
        .convert("RGB")
    )

    # --------------------------------------------------------
    # Enhance contrast.
    # --------------------------------------------------------

    enhanced_heatmap = (
        enhance_gradcam_heatmap(
            heatmap
        )
    )

    # --------------------------------------------------------
    # Standalone heatmap.
    # --------------------------------------------------------

    heatmap_image = (
        colorize_gradcam_heatmap(
            enhanced_heatmap
        )
    )

    # --------------------------------------------------------
    # Bicubic resize.
    # --------------------------------------------------------

    heatmap_image = (
        heatmap_image.resize(
            original.size,
            Image.Resampling.BICUBIC
        )
    )

    # --------------------------------------------------------
    # Smooth color map.
    # --------------------------------------------------------

    heatmap_image = (
        heatmap_image.filter(
            ImageFilter.GaussianBlur(
                radius=GRADCAM_BLUR_RADIUS
            )
        )
    )

    # --------------------------------------------------------
    # Resize activation mask using bicubic.
    # --------------------------------------------------------

    mask_image = Image.fromarray(
        (
            enhanced_heatmap
            *
            255.0
        ).astype(
            np.uint8
        ),
        mode="L"
    )

    mask_image = (
        mask_image.resize(
            original.size,
            Image.Resampling.BICUBIC
        )
    )

    mask_array = (
        np.asarray(
            mask_image,
            dtype=np.float32
        )
        /
        255.0
    )

    # --------------------------------------------------------
    # Alpha follows activation strength.
    # --------------------------------------------------------

    alpha_array = np.power(
        np.clip(
            mask_array,
            0.0,
            1.0
        ),
        0.85
    )

    alpha_array *= (
        GRADCAM_OVERLAY_ALPHA
    )

    # --------------------------------------------------------
    # Suppress weak background activation.
    # --------------------------------------------------------

    alpha_array[
        mask_array < 0.08
    ] = 0.0

    alpha_image = Image.fromarray(
        (
            np.clip(
                alpha_array,
                0.0,
                1.0
            )
            *
            255.0
        ).astype(
            np.uint8
        ),
        mode="L"
    )

    # --------------------------------------------------------
    # Transparent heatmap.
    # --------------------------------------------------------

    heatmap_rgba = (
        heatmap_image.convert(
            "RGBA"
        )
    )

    heatmap_rgba.putalpha(
        alpha_image
    )

    # --------------------------------------------------------
    # Blend with original X-ray.
    # --------------------------------------------------------

    original_rgba = (
        original.convert(
            "RGBA"
        )
    )

    overlay = Image.alpha_composite(
        original_rgba,
        heatmap_rgba
    )

    return (
        original,
        heatmap_image,
        overlay.convert("RGB")
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    modality_result,
    verifier_result,
    pneumonia_result=None,
    gradcam_overlay=None,
    gradcam_layer_name=None
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=14,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "PneuX-ModNet<br/>"
            "AI-Based Medical Image Analysis System",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Medical Image Analysis Report",
            subtitle_style
        )
    )

    report_time = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    story.append(
        Paragraph(
            f"<b>Analysis Date:</b> {report_time}",
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    report_image = RLImage(
        image_buffer,
        width=100 * mm,
        height=100 * mm
    )

    story.append(
        report_image
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    # --------------------------------------------------------
    # MODALITY
    # --------------------------------------------------------

    modality = modality_result[
        "class"
    ]

    modality_confidence = (
        modality_result[
            "confidence"
        ]
        *
        100
    )

    modality_data = [
        ["Parameter", "Result"],
        [
            "Detected Modality",
            modality
        ],
        [
            "Modality Confidence",
            f"{modality_confidence:.2f}%"
        ]
    ]

    story.append(
        Paragraph(
            "1. Medical Image Modality",
            heading_style
        )
    )

    modality_table = Table(
        modality_data,
        colWidths=[
            70 * mm,
            80 * mm
        ]
    )

    modality_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        modality_table
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    # --------------------------------------------------------
    # X-RAY VERIFICATION
    # --------------------------------------------------------

    if verifier_result is not None:

        xray_probability = (
            verifier_result[
                "xray_probability"
            ]
            *
            100
        )

        verifier_data = [
            ["Parameter", "Result"],
            [
                "X-ray Verification",
                (
                    "X-ray"
                    if verifier_result[
                        "is_xray"
                    ]
                    else
                    "Not X-ray"
                )
            ],
            [
                "X-ray Probability",
                f"{xray_probability:.2f}%"
            ]
        ]

        story.append(
            Paragraph(
                "2. X-ray Verification",
                heading_style
            )
        )

        verifier_table = Table(
            verifier_data,
            colWidths=[
                70 * mm,
                80 * mm
            ]
        )

        verifier_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    )
                ]
            )
        )

        story.append(
            verifier_table
        )

    # --------------------------------------------------------
    # PNEUMONIA
    # --------------------------------------------------------

    if pneumonia_result is not None:

        diagnosis = (
            pneumonia_result["class"]
        )

        confidence = (
            pneumonia_result["confidence"]
            *
            100
        )

        if diagnosis == "Pneumonia":

            probability = (
                pneumonia_result[
                    "pneumonia_probability"
                ]
                *
                100
            )

            pneumonia_data = [
                ["Parameter", "Result"],
                [
                    "Final Diagnosis",
                    diagnosis
                ],
                [
                    "Diagnosis Confidence",
                    f"{confidence:.2f}%"
                ],
                [
                    "Pneumonia Probability",
                    f"{probability:.2f}%"
                ]
            ]

        else:

            probability = (
                pneumonia_result[
                    "normal_probability"
                ]
                *
                100
            )

            pneumonia_data = [
                ["Parameter", "Result"],
                [
                    "Final Diagnosis",
                    diagnosis
                ],
                [
                    "Diagnosis Confidence",
                    f"{confidence:.2f}%"
                ],
                [
                    "Normal Probability",
                    f"{probability:.2f}%"
                ]
            ]

        story.append(
            Paragraph(
                "3. Pneumonia Detection",
                heading_style
            )
        )

        pneumonia_table = Table(
            pneumonia_data,
            colWidths=[
                70 * mm,
                80 * mm
            ]
        )

        pneumonia_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    )
                ]
            )
        )

        story.append(
            pneumonia_table
        )

        story.append(
            Spacer(
                1,
                12
            )
        )

        # ----------------------------------------------------
        # GRAD-CAM++
        # ----------------------------------------------------

        if (
            diagnosis == "Pneumonia"
            and
            gradcam_overlay is not None
        ):

            story.append(
                Paragraph(
                    "4. Pneumonia Localization "
                    "(Grad-CAM++)",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    "The highlighted region represents "
                    "areas of the chest X-ray associated "
                    "with the Pneumonia prediction.",
                    normal_style
                )
            )

            story.append(
                Spacer(
                    1,
                    8
                )
            )

            gradcam_buffer = io.BytesIO()

            gradcam_overlay.save(
                gradcam_buffer,
                format="PNG"
            )

            gradcam_buffer.seek(0)

            gradcam_report_image = RLImage(
                gradcam_buffer,
                width=140 * mm,
                height=140 * mm
            )

            story.append(
                gradcam_report_image
            )

            story.append(
                Spacer(
                    1,
                    8
                )
            )

            if gradcam_layer_name:

                story.append(
                    Paragraph(
                        f"<b>Grad-CAM++ Feature Layer:</b> "
                        f"{gradcam_layer_name}",
                        normal_style
                    )
                )

            story.append(
                Spacer(
                    1,
                    8
                )
            )

            story.append(
                Paragraph(
                    "<b>Localization Note:</b> "
                    "Grad-CAM++ is an explainability "
                    "technique that highlights image "
                    "regions associated with the model "
                    "prediction. It is not a pixel-level "
                    "clinical segmentation or a confirmed "
                    "boundary of disease.",
                    normal_style
                )
            )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This application is a "
            "research prototype and is not intended to "
            "provide clinical diagnosis or replace "
            "professional medical evaluation.",
            normal_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader(
    "Upload Medical Image"
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tif",
        "tiff"
    ],
    help=(
        "Upload a grayscale medical image "
        "such as CT, MRI, or X-ray."
    )
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )

        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        )

        image = ImageOps.exif_transpose(
            image
        )

        image.load()

    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()

    # ========================================================
    # DISPLAY UPLOADED IMAGE
    # ========================================================

    st.subheader(
        "Uploaded Medical Image"
    )

    # Keep the uploaded image from occupying the entire page.
    image_col1, image_col2, image_col3 = st.columns(
        [1, 2, 1]
    )

    with image_col2:

        st.image(
            image,
            caption="Uploaded Medical Image",
            width=500
        )

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "Check Your Image By Initiating AI Analysis",
        type="primary",
        use_container_width=True
    )

    if analyze:

        # ====================================================
        # STEP 1
        # ====================================================

        with st.spinner(
            "Validating image..."
        ):

            is_valid, validation_message = (
                validate_image(
                    image
                )
            )

        if not is_valid:

            st.error(
                validation_message
            )

            st.session_state.analysis_result = None

            st.session_state.pdf_report = None

            st.stop()

        st.success(
            "Image passed basic validation."
        )

        # ====================================================
        # STEP 2 — MODALITY
        # ====================================================

        with st.spinner(
            "Classifying medical image modality..."
        ):

            try:

                modality_result = (
                    predict_modality(
                        image
                    )
                )

            except Exception as e:

                st.error(
                    "Medical image modality classification failed."
                )

                st.exception(e)

                st.stop()

        modality = (
            modality_result["class"]
        )

        modality_confidence = (
            modality_result["confidence"]
            *
            100
        )

        # ====================================================
        # WEB INTERFACE:
        # CLASS ONLY
        # ====================================================

        st.markdown(
            "## Detected Medical Image Type"
        )

        st.markdown(
            f"### {modality}"
        )

        # ====================================================
        # CT / MRI STOP
        # ====================================================

        if modality in (
            "CT",
            "MRI"
        ):

            st.warning(
                f"This medical image was classified as "
                f"**{modality}**."
            )

            st.info(
                "Pneumonia detection is available only "
                "for chest X-ray images. Analysis has "
                "therefore stopped."
            )

            st.session_state.analysis_result = {
                "modality": modality,
                "modality_confidence": modality_confidence,
                "verifier": None,
                "pneumonia": None,
                "gradcam": None
            }

            st.session_state.pdf_report = (
                create_pdf_report(
                    image,
                    modality_result,
                    None,
                    None,
                    None,
                    None
                )
            )

            st.download_button(
                label="Download Modality Report (PDF)",
                data=st.session_state.pdf_report,
                file_name=(
                    "medical_image_modality_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

            st.stop()

        # ====================================================
        # STEP 3 — X-RAY VERIFICATION
        # ====================================================

        if modality == "X-ray":

            st.markdown(
                "## X-ray Verification"
            )

            with st.spinner(
                "Verifying chest X-ray image..."
            ):

                try:

                    verifier_result = (
                        predict_xray_verification(
                            image
                        )
                    )

                except Exception as e:

                    st.error(
                        "X-ray verification failed."
                    )

                    st.exception(e)

                    st.stop()

            # =================================================
            # NOT X-RAY
            # =================================================

            if not verifier_result["is_xray"]:

                st.error(
                    "The X-ray verifier did not "
                    "confirm this image as an X-ray."
                )

                st.info(
                    "Pneumonia detection has been stopped."
                )

                st.session_state.analysis_result = {
                    "modality": modality,
                    "modality_confidence": modality_confidence,
                    "verifier": verifier_result,
                    "pneumonia": None,
                    "gradcam": None
                }

                st.session_state.pdf_report = (
                    create_pdf_report(
                        image,
                        modality_result,
                        verifier_result,
                        None,
                        None,
                        None
                    )
                )

                st.download_button(
                    label=(
                        "Download X-ray Verification "
                        "Report (PDF)"
                    ),
                    data=st.session_state.pdf_report,
                    file_name=(
                        "xray_verification_report.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True
                )

                st.stop()

            # =================================================
            # STEP 4 — PNEUMONIA
            # =================================================

            st.success(
                "Chest X-ray verified successfully."
            )

            st.markdown(
                "## Pneumonia Detection"
            )

            with st.spinner(
                "Analyzing chest X-ray for pneumonia..."
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
                pneumonia_result["class"]
            )

            # =================================================
            # FINAL RESULT
            # =================================================

            if diagnosis == "Pneumonia":

                st.error(
                    f"### Final Result: {diagnosis}"
                )

            else:

                st.success(
                    f"### Final Result: {diagnosis}"
                )

            # =================================================
            # STEP 5 — GRAD-CAM++
            # =================================================

            gradcam_overlay = None

            gradcam_layer_name = None

            if diagnosis == "Pneumonia":

                st.markdown(
                    "## Pneumonia Localization"
                )

                with st.spinner(
                    "Generating Grad-CAM++ localization..."
                ):

                    try:

                        heatmap, gradcam_layer_name = (
                            generate_gradcam_plus_plus(
                                image,
                                target_class_index=1
                            )
                        )

                        (
                            original_gradcam_image,
                            heatmap_image,
                            gradcam_overlay
                        ) = create_gradcam_overlay(
                            image,
                            heatmap
                        )

                    except Exception as e:

                        st.error(
                            "Grad-CAM++ localization could "
                            "not be generated."
                        )

                        st.exception(e)

                        gradcam_overlay = None

                # =================================================
                # DISPLAY HEATMAP
                # =================================================

                if gradcam_overlay is not None:

                    st.success(
                        "Pneumonia localization generated "
                        "using Grad-CAM++."
                    )

                    gradcam_col1, gradcam_col2 = (
                        st.columns(2)
                    )

                    with gradcam_col1:

                        st.image(
                            heatmap_image,
                            caption=(
                                "Grad-CAM++ Activation Heatmap"
                            ),
                            use_container_width=True
                        )

                    with gradcam_col2:

                        st.image(
                            gradcam_overlay,
                            caption=(
                                "Grad-CAM++ Pneumonia "
                                "Localization"
                            ),
                            use_container_width=True
                        )

                    st.info(
                        "Red and yellow regions indicate "
                        "the strongest areas associated with "
                        "the Pneumonia prediction. The "
                        "visualization is an explainability "
                        "map, not a clinical segmentation."
                    )

            # =================================================
            # CREATE PDF
            # =================================================

            pdf_bytes = create_pdf_report(
                image,
                modality_result,
                verifier_result,
                pneumonia_result,
                gradcam_overlay,
                gradcam_layer_name
            )

            st.session_state.analysis_result = {
                "modality": modality,
                "modality_confidence": modality_confidence,
                "verifier": verifier_result,
                "pneumonia": pneumonia_result,
                "gradcam": gradcam_overlay,
                "gradcam_layer": gradcam_layer_name
            }

            st.session_state.pdf_report = (
                pdf_bytes
            )

            # =================================================
            # PDF DOWNLOAD
            # =================================================

            st.divider()

            st.subheader(
                "Analysis Report"
            )

            st.download_button(
                label="Download Final Report (PDF)",
                data=pdf_bytes,
                file_name=(
                    "pneumonia_detection_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )
