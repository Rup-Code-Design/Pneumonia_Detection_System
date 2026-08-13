# ============================================================
# modality_model_builder.py
#
# Medical Image Modality Classifier
#
# Classes:
#   0 = CHEST_XRAY
#   1 = CT
#   2 = MRI
#
# Input:
#   224 x 224 x 3
#
# Output:
#   3-class softmax
# ============================================================

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


# ============================================================
# GELU ACTIVATION
# ============================================================

def gelu(x):
    return tf.keras.activations.gelu(x)


# ============================================================
# SE ATTENTION BLOCK
# ============================================================

def se_block(
    inputs,
    reduction=16,
    name="se"
):

    channels = inputs.shape[-1]

    if channels is None:
        raise ValueError(
            "SE block requires a known channel dimension."
        )

    reduced_channels = max(
        channels // reduction,
        1
    )

    x = layers.GlobalAveragePooling2D(
        name=f"{name}_gap"
    )(inputs)

    x = layers.Reshape(
        (1, 1, channels),
        name=f"{name}_reshape"
    )(x)

    x = layers.Dense(
        reduced_channels,
        activation="relu",
        name=f"{name}_reduce"
    )(x)

    x = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{name}_expand"
    )(x)

    x = layers.Multiply(
        name=f"{name}_scale"
    )([
        inputs,
        x
    ])

    return x


# ============================================================
# CONVOLUTIONAL BLOCK
# ============================================================

def conv_block(
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    name="conv_block"
):

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv"
    )(inputs)

    x = layers.BatchNormalization(
        name=f"{name}_bn"
    )(x)

    x = layers.Activation(
        gelu,
        name=f"{name}_gelu"
    )(x)

    return x


# ============================================================
# SEPARABLE CONVOLUTIONAL BLOCK
# ============================================================

def separable_conv_block(
    inputs,
    filters,
    strides=1,
    name="sep_block"
):

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv"
    )(inputs)

    x = layers.BatchNormalization(
        name=f"{name}_bn"
    )(x)

    x = layers.Activation(
        gelu,
        name=f"{name}_gelu"
    )(x)

    return x


# ============================================================
# RESIDUAL BLOCK
# ============================================================

def residual_block(
    inputs,
    filters,
    strides=1,
    name="residual"
):

    shortcut = inputs

    # --------------------------------------------------------
    # Main path
    # --------------------------------------------------------

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv1"
    )(inputs)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.Activation(
        gelu,
        name=f"{name}_gelu1"
    )(x)

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    # --------------------------------------------------------
    # Shortcut projection
    # --------------------------------------------------------

    if (
        inputs.shape[-1] != filters
        or strides != 1
    ):

        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_shortcut_conv"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_shortcut_bn"
        )(shortcut)

    # --------------------------------------------------------
    # Residual addition
    # --------------------------------------------------------

    x = layers.Add(
        name=f"{name}_add"
    )([
        x,
        shortcut
    ])

    x = layers.Activation(
        gelu,
        name=f"{name}_output"
    )(x)

    return x


# ============================================================
# BUILD MODALITY CLASSIFIER
# ============================================================

def build_modality_classifier(
    input_shape=(224, 224, 3),
    num_classes=3
):

    # ========================================================
    # VALIDATION
    # ========================================================

    if num_classes != 3:

        raise ValueError(
            "This modality classifier requires exactly "
            "3 classes:\n"
            "0 = CHEST_XRAY\n"
            "1 = CT\n"
            "2 = MRI"
        )

    if len(input_shape) != 3:

        raise ValueError(
            "input_shape must be "
            "(height, width, channels)."
        )

    if input_shape[-1] != 3:

        raise ValueError(
            "The model requires 3 input channels."
        )


    # ========================================================
    # INPUT
    # ========================================================

    inputs = layers.Input(
        shape=input_shape,
        name="image_input"
    )


    # ========================================================
    # INITIAL CONVOLUTION
    #
    # IMPORTANT:
    #
    # Your saved weights previously showed:
    #
    # (3, 3, 3, 32)
    #
    # Therefore the first Conv2D uses 32 filters.
    # ========================================================

    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv2d"
    )(inputs)

    x = layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = layers.Activation(
        gelu,
        name="activation"
    )(x)


    # ========================================================
    # FEATURE BLOCK 1
    # ========================================================

    x = layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name="separable_conv2d"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization_1"
    )(x)

    x = layers.Activation(
        gelu,
        name="activation_1"
    )(x)


    # ========================================================
    # FEATURE BLOCK 2
    # ========================================================

    x = layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="separable_conv2d_1"
    )(x)

    x = layers
