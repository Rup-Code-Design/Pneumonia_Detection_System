# ============================================================
# pneumonia_model_builder.py
# Pneumonia Detection Model Architecture
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model, Input


# ============================================================
# SE ATTENTION BLOCK
# ============================================================

def se_block(x, reduction=16):

    channels = x.shape[-1]

    se = layers.GlobalAveragePooling2D()(x)

    se = layers.Dense(
        max(channels // reduction, 4),
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
# XCEPTION BLOCK
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

    x = layers.BatchNormalization()(x)

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

    x = layers.BatchNormalization()(x)

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

    x = se_block(x)

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
# BUILD PNEUMONIA MODEL
# ============================================================

def build_model(
    input_shape=(224, 224, 3)
):

    inputs = Input(
        shape=input_shape,
        name="input_image"
    )

    # ========================================================
    # STEM
    # ========================================================

    x = conv_block(
        inputs,
        16,
        kernel_size=3
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # XCEPTION BLOCK
    # ========================================================

    x = xception_block(
        x,
        32
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # RESIDUAL BLOCK 1
    # ========================================================

    x = residual_block(
        x,
        64
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # RESIDUAL BLOCK 2
    # ========================================================

    x = residual_block(
        x,
        128
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)

    # ========================================================
    # RESIDUAL BLOCK 3
    # ========================================================

    x = residual_block(
        x,
        128
    )

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        256,
        activation="gelu"
    )(x)

    x = layers.Dropout(
        0.3
    )(x)

    # ========================================================
    # BINARY PNEUMONIA OUTPUT
    #
    # 0 = Normal
    # 1 = Pneumonia
    # ========================================================

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="pneumonia_probability"
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Pneumonia_Detection_Model"
    )

    return model
```
