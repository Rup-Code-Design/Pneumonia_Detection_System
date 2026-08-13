import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


# ============================================================
# MODALITY MODEL BUILDER
# ============================================================
#
# PRETRAINED BACKBONE:
#     best_xray_verifier.keras
#
# PURPOSE:
#     Use the pretrained X-ray verifier as a feature extractor
#     and add a new 4-class medical-image modality classifier.
#
# CLASSES:
#
#     0 = CHEST_XRAY
#     1 = CT
#     2 = MRI
#     3 = OTHER
#
# IMPORTANT:
#
# The resulting modality classifier MUST be trained before
# using its weights in Streamlit.
#
# ============================================================


# ============================================================
# DEFAULT PATH
# ============================================================

DEFAULT_XRAY_MODEL_PATH = "best_xray_verifier.keras"


# ============================================================
# GELU
# ============================================================

def gelu(x):

    return tf.keras.activations.gelu(x)


# ============================================================
# BUILD MODALITY CLASSIFIER
# ============================================================

def build_modality_classifier(
    input_shape=(128, 128, 3),
    num_classes=4,
    xray_model_path=DEFAULT_XRAY_MODEL_PATH,
    freeze_backbone=True
):

    # ========================================================
    # CHECK X-RAY MODEL
    # ========================================================

    if not os.path.isfile(xray_model_path):

        raise FileNotFoundError(
            "Pretrained X-ray verifier was not found.\n"
            f"Expected location: {os.path.abspath(xray_model_path)}"
        )


    # ========================================================
    # LOAD PRETRAINED X-RAY VERIFIER
    # ========================================================

    print(
        "Loading pretrained X-ray verifier:"
    )

    print(
        os.path.abspath(
            xray_model_path
        )
    )


    try:

        xray_model = tf.keras.models.load_model(
            xray_model_path,
            compile=False
        )

    except Exception as e:

        raise RuntimeError(
            "Could not load best_xray_verifier.keras.\n"
            "Make sure this .keras file is a complete Keras model "
            "and that any custom layers/functions used during "
            "training are available.\n\n"
            f"Original error: {e}"
        )


    # ========================================================
    # CHECK INPUT SHAPE
    # ========================================================

    model_input_shape = xray_model.input_shape

    print(
        "X-ray verifier input shape:",
        model_input_shape
    )


    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # We use the actual input shape of the pretrained model.
    #
    # Do NOT force 224x224 if the X-ray verifier was trained
    # with 128x128.
    #
    # ========================================================

    actual_input_shape = model_input_shape[1:]


    if actual_input_shape is None:

        raise ValueError(
            "Unable to determine the input shape "
            "of best_xray_verifier.keras."
        )


    # ========================================================
    # VERIFY USER REQUESTED SHAPE
    # ========================================================

    if tuple(input_shape) != tuple(actual_input_shape):

        print(
            "WARNING:"
        )

        print(
            f"Requested input shape: {input_shape}"
        )

        print(
            f"X-ray model input shape: {actual_input_shape}"
        )

        print(
            "Using the X-ray model's actual input shape."
        )


    input_shape = tuple(actual_input_shape)


    # ========================================================
    # FREEZE / UNFREEZE BACKBONE
    # ========================================================

    xray_model.trainable = not freeze_backbone


    # ========================================================
    # GET FEATURE OUTPUT
    # ========================================================
    #
    # The X-ray verifier's final layer is assumed to be its
    # original X-ray / Non-X-ray classification layer.
    #
    # We remove that classification decision and use the
    # preceding layer as the feature representation.
    #
    # ========================================================

    if len(xray_model.layers) < 2:

        raise ValueError(
            "The X-ray verifier does not contain enough layers "
            "to create a feature extractor."
        )


    feature_layer = xray_model.layers[-2]


    feature_output = feature_layer.output


    print(
        "Feature layer:",
        feature_layer.name
    )

    print(
        "Feature shape:",
        feature_output.shape
    )


    # ========================================================
    # CREATE FEATURE EXTRACTOR
    # ========================================================

    feature_extractor = Model(
        inputs=xray_model.input,
        outputs=feature_output,
        name="XRay_Pretrained_Feature_Extractor"
    )


    # ========================================================
    # NEW MODALITY INPUT
    # ========================================================

    inputs = layers.Input(
        shape=input_shape,
        name="modality_input"
    )


    # ========================================================
    # EXTRACT PRETRAINED X-RAY FEATURES
    # ========================================================

    x = feature_extractor(
        inputs
    )


    # ========================================================
    # HANDLE FEATURE DIMENSION
    # ========================================================
    #
    # If the selected feature layer is still spatial
    # (4D tensor), apply global average pooling.
    #
    # If it is already a vector, keep it.
    #
    # ========================================================

    if len(x.shape) == 4:

        x = layers.GlobalAveragePooling2D(
            name="modality_feature_pooling"
        )(x)

    elif len(x.shape) == 3:

        x = layers.GlobalAveragePooling1D(
            name="modality_feature_pooling"
        )(x)


    # ========================================================
    # MODALITY CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        256,
        use_bias=False,
        name="modality_dense"
    )(x)


    x = layers.BatchNormalization(
        name="modality_batch_normalization"
    )(x)


    x = layers.Activation(
        gelu,
        name="modality_gelu"
    )(x)


    x = layers.Dropout(
        0.30,
        name="modality_dropout"
    )(x)


    # ========================================================
    # 4-CLASS OUTPUT
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="modality_probability"
    )(x)


    # ========================================================
    # FINAL MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Medical_Image_Modality_Classifier"
    )


    # ========================================================
    # PRINT MODEL INFORMATION
    # ========================================================

    print(
        "\n=================================================="
    )

    print(
        "Medical Image Modality Classifier"
    )

    print(
        "=================================================="
    )

    print(
        "Input shape:",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )

    print(
        "Classes:",
        num_classes
    )

    print(
        "0 = CHEST_XRAY"
    )

    print(
        "1 = CT"
    )

    print(
        "2 = MRI"
    )

    print(
        "3 = OTHER"
    )

    print(
        "Backbone frozen:",
        freeze_backbone
    )

    print(
        "==================================================\n"
    )


    return model
