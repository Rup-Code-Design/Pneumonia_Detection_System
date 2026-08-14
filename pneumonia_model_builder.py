# ============================================================
# pneumonia_model_builder.py
# PROPOSED PNEUMONIA MODEL
#
# Architecture:
# Xception-style blocks
# + SE Attention
# + Residual Block
# + GAP + GMP Feature Fusion
# + GELU Classifier
#
# CLASSIFICATION:
# 0 = Normal
# 1 = Pneumonia
# ============================================================

import tensorflow as tf

from tensorflow.keras import (
    layers,
    Model,
    Input
)


# ============================================================
# CLASS CONFIGURATION
# ============================================================

NUM_CLASSES = 2

CLASS_NAMES = [
    "Normal",
    "Pneumonia"
]


# ============================================================
# SE ATTENTION
# ============================================================

def se_block(
    x,
    reduction=16
):

    channels = x.shape[-1]

    if channels is None:
        raise ValueError(
            "The number of channels must be known "
            "before applying SE attention."
        )

    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    se = layers.GlobalAveragePooling2D(
        name="se_global_average_pooling"
    )(x)

    # --------------------------------------------------------
    # Bottleneck
    # --------------------------------------------------------

    se = layers.Dense(
        max(
            channels // reduction,
            4
        ),
        activation="gelu",
        name="se_dense_1"
    )(se)

    # --------------------------------------------------------
    # Channel Attention
    # --------------------------------------------------------

    se = layers.Dense(
        channels,
        activation="sigmoid",
        name="se_dense_2"
    )(se)

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    se = layers.Reshape(
        (1, 1, channels),
        name="se_reshape"
    )(se)

    # --------------------------------------------------------
    # Channel-wise multiplication
    # --------------------------------------------------------

    x = layers.Multiply(
        name="se_multiply"
    )(
        [
            x,
            se
        ]
    )

    return x


# ============================================================
# XCEPTION BLOCK
# ============================================================

def xception_block(
    x,
    filters,
    stride=1
):

    shortcut = x

    # --------------------------------------------------------
    # SEPARABLE CONVOLUTION 1
    # --------------------------------------------------------

    x = layers.SeparableConv2D(
        filters,
        3,
        strides=stride,
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
    # SEPARABLE CONVOLUTION 2
    # --------------------------------------------------------

    x = layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    # --------------------------------------------------------
    # SHORTCUT PROJECTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESIDUAL ADDITION
    # --------------------------------------------------------

    x = layers.Add()(
        [
            x,
            shortcut
        ]
    )

    x = layers.Activation(
        "gelu"
    )(x)

    # --------------------------------------------------------
    # SE ATTENTION
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
    filters,
    stride=1
):

    shortcut = x

    # --------------------------------------------------------
    # CONVOLUTION 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
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
    # CONVOLUTION 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    # --------------------------------------------------------
    # SHORTCUT PROJECTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESIDUAL ADDITION
    # --------------------------------------------------------

    x = layers.Add()(
        [
            x,
            shortcut
        ]
    )

    x = layers.Activation(
        "gelu"
    )(x)

    return x


# ============================================================
# PROPOSED PNEUMONIA MODEL
# ============================================================

def build_model(
    input_shape=(224, 224, 3),
    num_classes=NUM_CLASSES
):

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if num_classes != 2:

        raise ValueError(
            "This pneumonia model is designed "
            "for exactly 2 classes:\n"
            "0 = Normal\n"
            "1 = Pneumonia"
        )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    inputs = Input(
        shape=input_shape,
        name="pneumonia_input"
    )

    # ========================================================
    # XCEPTION STAGE 1
    # ========================================================

    x = xception_block(
        inputs,
        64
    )

    x = layers.MaxPooling2D(
        2
    )(x)

    # ========================================================
    # XCEPTION STAGE 2
    # ========================================================

    x = xception_block(
        x,
        128
    )

    x = layers.MaxPooling2D(
        2
    )(x)

    # ========================================================
    # RESIDUAL STAGE
    # ========================================================

    x = residual_block(
        x,
        256
    )

    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    gap = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # ========================================================
    # GLOBAL MAX POOLING
    # ========================================================

    gmp = layers.GlobalMaxPooling2D(
        name="global_max_pooling"
    )(x)

    # ========================================================
    # GAP + GMP FEATURE FUSION
    # ========================================================

    x = layers.Concatenate(
        name="gap_gmp_fusion"
    )(
        [
            gap,
            gmp
        ]
    )

    # ========================================================
    # BATCH NORMALIZATION
    # ========================================================

    x = layers.BatchNormalization(
        name="classifier_batch_normalization"
    )(x)

    # ========================================================
    # DENSE CLASSIFIER
    # ========================================================

    x = layers.Dense(
        128,
        activation="gelu",
        name="classifier_dense"
    )(x)

    # ========================================================
    # DROPOUT
    # ========================================================

    x = layers.Dropout(
        0.3,
        name="classifier_dropout"
    )(x)

    # ========================================================
    # FINAL TWO-CLASS OUTPUT
    #
    # IMPORTANT:
    #
    # 0 = Normal
    # 1 = Pneumonia
    #
    # The training dataset MUST use the same mapping.
    # ========================================================

    outputs = layers.Dense(
        2,
        activation="softmax",
        name="pneumonia_probability"
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Xception_SE_Residual_Pneumonia"
    )

    return model
