# ============================================================
# modality_model_builder.py
#
# 4-Class Medical Image Modality Classifier
#
# Classes:
#   0 = CHEST_XRAY
#   1 = CT
#   2 = MRI
#   3 = OTHER
#
# IMPORTANT:
# This architecture is intended to match the modality
# classifier weights used by the Streamlit application.
# ============================================================

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers


# ============================================================
# SE BLOCK
# ============================================================

def se_block(
    inputs,
    reduction=16,
    name="se"
):

    channels = inputs.shape[-1]

    if channels is None:
        raise ValueError(
            "Input channel dimension must be defined."
        )

    # Squeeze
    x = layers.GlobalAveragePooling2D(
        name=f"{name}_gap"
    )(inputs)

    x = layers.Reshape(
        (1, 1, channels),
        name=f"{name}_reshape"
    )(x)

    # Excitation
    x = layers.Dense(
        max(channels // reduction, 1),
        activation="relu",
        name=f"{name}_dense1"
    )(x)

    x = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{name}_dense2"
    )(x)

    return layers.Multiply(
        name=f"{name}_multiply"
    )([
        inputs,
        x
    ])


# ============================================================
# CONVOLUTIONAL BLOCK
# ============================================================

def conv_block(
    inputs,
    filters,
    stride=1,
    name="conv_block"
):

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv"
    )(inputs)

    x = layers.BatchNormalization(
        name=f"{name}_bn"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name=f"{name}_gelu"
    )(x)

    return x


# ============================================================
# DEPTHWISE-SEPARABLE BLOCK
# ============================================================

def separable_block(
    inputs,
    filters,
    stride=1,
    name="sep_block"
):

    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv"
    )(inputs)

    x = layers.BatchNormalization(
        name=f"{name}_bn"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name=f"{name}_gelu"
    )(x)

    return x


# ============================================================
# RESIDUAL BLOCK
# ============================================================

def residual_block(
    inputs,
    filters,
    stride=1,
    name="residual"
):

    shortcut = inputs

    # Main path
    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv1"
    )(inputs)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name=f"{name}_gelu1"
    )(x)

    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    # Shortcut projection when dimensions differ
    if (
        inputs.shape[-1] != filters
        or stride != 1
    ):

        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name}_shortcut_conv"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_shortcut_bn"
        )(shortcut)

    x = layers.Add(
        name=f"{name}_add"
    )([
        x,
        shortcut
    ])

    x = layers.Activation(
        tf.keras.activations.gelu,
        name=f"{name}_output"
    )(x)

    return x


# ============================================================
# BUILD MODALITY CLASSIFIER
# ============================================================

def build_modality_classifier(
    input_shape=(224, 224, 3),
    num_classes=4
):

    # --------------------------------------------------------
    # Validate number of classes
    # --------------------------------------------------------

    if num_classes != 4:

        raise ValueError(
            "This modality classifier requires "
            "exactly 4 classes:\n"
            "0 = CHEST_XRAY\n"
            "1 = CT\n"
            "2 = MRI\n"
            "3 = OTHER"
        )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=input_shape,
        name="image_input"
    )


    # ========================================================
    # INITIAL FEATURE EXTRACTION
    # ========================================================

    # IMPORTANT:
    # The modality weights show:
    #
    # Conv2D kernel =
    # (3, 3, 3, 32)
    #
    # Therefore the first convolution MUST have 32 filters.
    #

    x = layers.Conv2D(
        32,
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
        tf.keras.activations.gelu,
        name="activation"
    )(x)


    # ========================================================
    # FEATURE BLOCK 1
    # ========================================================

    x = separable_block(
        x,
        64,
        stride=1,
        name="separable_conv2d"
    )


    # ========================================================
    # FEATURE BLOCK 2
    # ========================================================

    x = separable_block(
        x,
        128,
        stride=2,
        name="separable_conv2d_1"
    )


    # ========================================================
    # RESIDUAL BLOCK
    # ========================================================

    x = residual_block(
        x,
        128,
        stride=1,
        name="residual_block"
    )


    # ========================================================
    # SE ATTENTION
    # ========================================================

    x = se_block(
        x,
        reduction=16,
        name="se_attention"
    )


    # ========================================================
    # HIGH-LEVEL FEATURES
    # ========================================================

    x = layers.Conv2D(
        256,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv2d_1"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization_2"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name="activation_1"
    )(x)


    x = layers.Conv2D(
        512,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv2d_2"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization_4"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name="activation_2"
    )(x)


    # ========================================================
    # GLOBAL FEATURES
    # ========================================================

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        256,
        activation=tf.keras.activations.gelu,
        name="dense"
    )(x)


    x = layers.Dropout(
        0.30,
        name="dropout"
    )(x)


    x = layers.Dense(
        128,
        activation=tf.keras.activations.gelu,
        name="dense_1"
    )(x)


    x = layers.Dropout(
        0.20,
        name="dropout_1"
    )(x)


    # ========================================================
    # MODALITY CLASSIFIER HEAD
    # ========================================================

    modality_features = layers.Dense(
        128,
        activation=tf.keras.activations.gelu,
        name="modality_dense"
    )(x)


    modality_features = layers.BatchNormalization(
        name="modality_batch_normalization"
    )(modality_features)


    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="modality_probability"
    )(modality_features)


    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Medical_Image_Modality_Classifier"
    )


    return model
