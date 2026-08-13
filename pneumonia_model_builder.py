# ============================================================
# pneumonia_model_builder.py
# PROPOSED PNEUMONIA MODEL
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
    # SHORTCUT
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
    # RESIDUAL ADD
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
    # SE
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
    # SHORTCUT
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
    # ADD
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
# PROPOSED MODEL
# ============================================================

def build_model(
    input_shape=(224, 224, 3),
    num_classes=2
):

    inputs = Input(
        shape=input_shape
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
    # GAP + GMP FUSION
    # ========================================================

    gap = layers.GlobalAveragePooling2D()(
        x
    )

    gmp = layers.GlobalMaxPooling2D()(
        x
    )

    x = layers.Concatenate()(
        [
            gap,
            gmp
        ]
    )

    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = layers.BatchNormalization()(
        x
    )

    x = layers.Dense(
        128,
        activation="gelu"
    )(x)

    x = layers.Dropout(
        0.3
    )(x)

    # ========================================================
    # TWO CLASSES
    #
    # 0 = Normal
    # 1 = Pneumonia
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="pneumonia_probability"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="Xception_SE_Residual_Pneumonia"
    )

    return model
