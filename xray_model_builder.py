# ============================================================
# xray_model_builder.py
# 3-Class Medical Image Verifier
#
# 0 = X-RAY
# 1 = CT
# 2 = MRI
# ============================================================

import tensorflow as tf

from tensorflow.keras import (
    layers,
    Model,
    Input
)


# ============================================================
# SE ATTENTION
# ============================================================

def se_block(
    x,
    reduction=16
):

    channels = x.shape[-1]

    channels = int(channels)

    se = layers.GlobalAveragePooling2D()(x)

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

    x = layers.BatchNormalization()(x)

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

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "gelu"
    )(x)

    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

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

    x = layers.Add()(
        [x, shortcut]
    )

    x = layers.Activation(
        "gelu"
    )(x)

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

    x = conv_block(
        x,
        filters
    )

    x = conv_block(
        x,
        filters
    )

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

    x = layers.Add()(
        [x, shortcut]
    )

    x = layers.Activation(
        "gelu"
    )(x)

    return x


# ============================================================
# BUILD 3-CLASS VERIFIER
# ============================================================

def build_xray_classifier(
    input_shape=(128, 128, 3),
    num_classes=3
):

    inputs = Input(
        shape=input_shape,
        name="input_image"
    )

    # --------------------------------------------------------
    # STEM
    # --------------------------------------------------------

    x = conv_block(
        inputs,
        16
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # XCEPTION BLOCK
    # --------------------------------------------------------

    x = xception_block(
        x,
        32
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # RESIDUAL BLOCK 1
    # --------------------------------------------------------

    x = residual_block(
        x,
        64
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # RESIDUAL BLOCK 2
    # --------------------------------------------------------

    x = residual_block(
        x,
        128
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # --------------------------------------------------------
    # RESIDUAL BLOCK 3
    # --------------------------------------------------------

    x = residual_block(
        x,
        128
    )

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        128,
        activation="gelu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    # --------------------------------------------------------
    # THREE CLASS OUTPUT
    #
    # 0 = X-RAY
    # 1 = CT
    # 2 = MRI
    # --------------------------------------------------------

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="medical_image_classification"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="XRay_CT_MRI_Verifier"
    )

    return model
