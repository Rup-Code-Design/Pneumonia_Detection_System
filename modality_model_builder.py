import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


# ============================================================
# MODALITY MODEL BUILDER
# ============================================================
#
# Classes:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
# 3 = OTHER
#
# Input:
# (224, 224, 3)
#
# Output:
# 4-class softmax
#
# ============================================================


# ============================================================
# GELU ACTIVATION
# ============================================================

def gelu(x):

    return tf.keras.activations.gelu(
        x
    )


# ============================================================
# CONVOLUTIONAL BLOCK
# ============================================================

def conv_block(
    x,
    filters,
    kernel_size=3,
    stride=1
):

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        gelu
    )(x)

    return x


# ============================================================
# DEPTHWISE SEPARABLE CONVOLUTION BLOCK
# ============================================================

def depthwise_block(
    x,
    filters,
    stride=1
):

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        gelu
    )(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        gelu
    )(x)

    return x


# ============================================================
# SE ATTENTION BLOCK
# ============================================================

def se_block(
    x,
    reduction=16
):

    channels = x.shape[-1]

    if channels is None:
        raise ValueError(
            "Channel dimension cannot be None."
        )

    # Global average pooling

    se = layers.GlobalAveragePooling2D()(
        x
    )

    # Excitation

    se = layers.Dense(
        max(
            channels // reduction,
            8
        ),
        activation="relu"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid"
    )(se)

    # Reshape

    se = layers.Reshape(
        (1, 1, channels)
    )(se)

    # Channel recalibration

    x = layers.Multiply()(
        [x, se]
    )

    return x


# ============================================================
# RESIDUAL BLOCK
# ============================================================

def residual_block(
    x,
    filters,
    stride=1
):

    shortcut = x

    # Main path

    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        gelu
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        strides=1,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    # Shortcut projection when necessary

    if (
        shortcut.shape[-1] != filters
        or stride != 1
    ):

        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    # Add residual connection

    x = layers.Add()(
        [x, shortcut]
    )

    x = layers.Activation(
        gelu
    )(x)

    return x


# ============================================================
# BUILD MODALITY CLASSIFIER
# ============================================================

def build_modality_classifier(
    input_shape=(224, 224, 3),
    num_classes=4
):

    # ========================================================
    # INPUT
    # ========================================================

    inputs = layers.Input(
        shape=input_shape,
        name="modality_input"
    )


    # ========================================================
    # INITIAL FEATURE EXTRACTION
    # ========================================================

    x = conv_block(
        inputs,
        filters=32,
        kernel_size=3,
        stride=2
    )


    x = conv_block(
        x,
        filters=64,
        kernel_size=3,
        stride=1
    )


    # ========================================================
    # RESIDUAL FEATURE EXTRACTION
    # ========================================================

    x = residual_block(
        x,
        filters=64,
        stride=1
    )


    x = residual_block(
        x,
        filters=128,
        stride=2
    )


    # ========================================================
    # SE ATTENTION
    # ========================================================

    x = se_block(
        x
    )


    # ========================================================
    # DEPTHWISE FEATURE EXTRACTION
    # ========================================================

    x = depthwise_block(
        x,
        filters=128,
        stride=1
    )


    x = depthwise_block(
        x,
        filters=256,
        stride=2
    )


    # ========================================================
    # SE ATTENTION
    # ========================================================

    x = se_block(
        x
    )


    # ========================================================
    # HIGH-LEVEL FEATURES
    # ========================================================

    x = residual_block(
        x,
        filters=256,
        stride=1
    )


    x = residual_block(
        x,
        filters=512,
        stride=2
    )


    # ========================================================
    # FINAL SE ATTENTION
    # ========================================================

    x = se_block(
        x
    )


    # ========================================================
    # GLOBAL FEATURE REPRESENTATION
    # ========================================================

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        256,
        use_bias=False,
        name="modality_dense"
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Activation(
        gelu
    )(x)

    x = layers.Dropout(
        0.30
    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="modality_probability"
    )(x)


    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Medical_Image_Modality_Classifier"
    )


    return model
