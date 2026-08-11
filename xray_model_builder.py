
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
# XCEPTION-STYLE BLOCK
# ============================================================

def xception_block(
    x,
    filters
):

    shortcut = x


    # Depthwise separable convolution

    x = layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "gelu"
    )(x)


    # Second separable convolution

    x = layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)


    # Shortcut projection

    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )


    # Residual addition

    x = layers.Add()(
        [x, shortcut]
    )

    x = layers.Activation(
        "gelu"
    )(x)


    # SE attention

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
            filters,
            1,
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
# BUILD X-RAY CLASSIFIER
# ============================================================

def build_xray_classifier(
    input_shape=(128, 128, 3)
):

    inputs = Input(
        shape=input_shape
    )


    # --------------------------------------------------------
    # STEM
    # --------------------------------------------------------

    x = conv_block(
        inputs,
        16,
        3
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
    # RESIDUAL BLOCK
    # --------------------------------------------------------

    x = residual_block(
        x,
        64
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # RESIDUAL BLOCK
    # --------------------------------------------------------

    x = residual_block(
        x,
        128
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)


    # --------------------------------------------------------
    # FINAL RESIDUAL BLOCK
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
    # BINARY OUTPUT
    #
    # 0 = Chest X-ray
    # 1 = Non-X-ray
    # --------------------------------------------------------

    outputs = layers.Dense(
        2,
        activation="softmax",
        dtype="float32"
    )(x)


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="XRay_Verifier"
    )


    return model
