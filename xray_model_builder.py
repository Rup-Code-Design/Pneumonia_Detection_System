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

    if channels is None:

        raise ValueError(
            "Channel dimension is undefined."
        )

    channels = int(channels)


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
# CONVOLUTION BLOCK
# ============================================================

def conv_block(
    x,
    filters,
    kernel_size=3
):

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
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


    x = layers.SeparableConv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(
        x
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
        [
            x,
            shortcut
        ]
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
# 3-CLASS MEDICAL IMAGE VERIFIER
#
# 0 = X-RAY
# 1 = CT
# 2 = MRI
# ============================================================

def build_xray_classifier(
    input_shape=(128, 128, 3),
    num_classes=3
):

    inputs = Input(
        shape=input_shape,
        name="medical_image"
    )


    # ========================================================
    # STEM
    # ========================================================

    x = conv_block(
        inputs,
        16
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
    # RESIDUAL 64
    # ========================================================

    x = residual_block(
        x,
        64
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)


    # ========================================================
    # RESIDUAL 128
    # ========================================================

    x = residual_block(
        x,
        128
    )

    x = layers.MaxPooling2D(
        pool_size=2
    )(x)


    # ========================================================
    # FINAL RESIDUAL
    # ========================================================

    x = residual_block(
        x,
        128
    )


    # ========================================================
    # GAP + GMP
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
    # CLASSIFICATION HEAD
    # ========================================================

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
    # 3-CLASS OUTPUT
    # ========================================================

    outputs = layers.Dense(
        3,
        activation="softmax",
        dtype="float32",
        name="medical_image_classification"
    )(x)


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="XRay_Verifier_3Class"
    )


    return model
