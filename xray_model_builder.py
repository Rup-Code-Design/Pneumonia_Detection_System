# ============================================================
# xray_model_builder.py
#
# Medical Image Modality Classification Model
#
# Classes:
#
#   0 = X-RAY
#   1 = CT
#   2 = MRI
#
# IMPORTANT:
#
# This is a 3-class classifier.
# It MUST be trained using X-ray, CT and MRI images.
#
# Existing 2-class weights such as:
#
#   best_xray_verifier.weights.h5
#
# are NOT compatible with this 3-class model.
#
# Train and save a NEW weight file, for example:
#
#   best_medical_modality_verifier.weights.h5
# ============================================================

import tensorflow as tf

from tensorflow.keras import layers, Model, Input


# ============================================================
# CLASS DEFINITIONS
# ============================================================

MODALITY_CLASSES = [
    "X-RAY",
    "CT",
    "MRI"
]

NUM_CLASSES = 3


# ============================================================
# SE ATTENTION BLOCK
# ============================================================

def se_block(x, reduction=16):

    channels = x.shape[-1]

    if channels is None:

        raise ValueError(
            "Channel dimension must be defined "
            "for SE attention."
        )

    se = layers.GlobalAveragePooling2D()(
        x
    )

    se = layers.Dense(
        max(
            channels // reduction,
            4
        ),
        activation="gelu"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid"
    )(se)

    se = layers.Reshape(
        (1, 1, channels)
    )(se)

    x = layers.Multiply()(
        [x, se]
    )

    return x


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

def conv_block(
    x,
    filters,
    kernel_size=3,
    strides=1
):

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Activation(
        "gelu"
    )(x)

    return x


# ============================================================
# XCEPTION-STYLE BLOCK
# ============================================================

def xception_block(
    x,
    filters
):

    shortcut = x

    # --------------------------------------------------------
    # First separable convolution
    # --------------------------------------------------------

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Activation(
        "gelu"
    )(x)

    # --------------------------------------------------------
    # Second separable convolution
    # --------------------------------------------------------

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    # --------------------------------------------------------
    # Shortcut projection
    # --------------------------------------------------------

    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    # --------------------------------------------------------
    # Residual addition
    # --------------------------------------------------------

    x = layers.Add()(
        [x, shortcut]
    )

    x = layers.Activation(
        "gelu"
    )(x)

    # --------------------------------------------------------
    # SE attention
    # --------------------------------------------------------

    x = se_block(
        x
    )

    return x


# ============================================================
# RESIDUAL BLOCK
# ============================================================

def residual_block(
    x,
    filters
):

    shortcut = x

    # --------------------------------------------------------
    # First convolution
    # --------------------------------------------------------

    x = conv_block(
        x,
        filters,
        kernel_size=3
    )

    # --------------------------------------------------------
    # Second convolution
    # --------------------------------------------------------

    x = conv_block(
        x,
        filters,
        kernel_size=3
    )

    # --------------------------------------------------------
    # Shortcut projection
    # --------------------------------------------------------

    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    # --------------------------------------------------------
    # Residual addition
    # --------------------------------------------------------

    x = layers.Add()(
        [x, shortcut]
    )

    x = layers.Activation(
        "gelu"
    )(x)

    return x


# ============================================================
# BUILD MEDICAL MODALITY CLASSIFIER
# ============================================================

def build_xray_classifier(
    input_shape=(128, 128, 3)
):

    inputs = Input(
        shape=input_shape,
        name="medical_image_input"
    )

    # ========================================================
    # STEM
    # ========================================================

    x = conv_block(
        inputs,
        filters=16,
        kernel_size=3
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # XCEPTION-STYLE BLOCK
    # ========================================================

    x = xception_block(
        x,
        filters=32
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # RESIDUAL BLOCK 1
    # ========================================================

    x = residual_block(
        x,
        filters=64
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # RESIDUAL BLOCK 2
    # ========================================================

    x = residual_block(
        x,
        filters=128
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # RESIDUAL BLOCK 3
    # ========================================================

    x = residual_block(
        x,
        filters=128
    )

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling2D()(
        x
    )

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Dense(
        128,
        activation="gelu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    # ========================================================
    # THREE-CLASS OUTPUT
    # ========================================================
    #
    # 0 = X-RAY
    # 1 = CT
    # 2 = MRI
    #
    # ========================================================

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="modality_probability"
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Medical_Modality_Verifier"
    )

    return model
