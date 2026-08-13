# ============================================================
# modality_model_builder.py
#
# Medical Image Modality Classifier
#
# Classes:
#   0 = CHEST_XRAY
#   1 = CT
#   2 = MRI
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_modality_classifier(
    input_shape=(224, 224, 3),
    num_classes=3
):

    if num_classes != 3:
        raise ValueError(
            "Modality classifier requires exactly 3 classes:\n"
            "0 = CHEST_XRAY\n"
            "1 = CT\n"
            "2 = MRI"
        )

    inputs = layers.Input(
        shape=input_shape,
        name="image_input"
    )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    x = layers.Conv2D(
        32,
        (3, 3),
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

    x = layers.SeparableConv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
        name="separable_conv2d"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization_1"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name="activation_1"
    )(x)

    x = layers.SeparableConv2D(
        128,
        (3, 3),
        strides=2,
        padding="same",
        use_bias=False,
        name="separable_conv2d_1"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization_2"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name="activation_2"
    )(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        strides=2,
        padding="same",
        use_bias=False,
        name="conv2d_1"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization_3"
    )(x)

    x = layers.Activation(
        tf.keras.activations.gelu,
        name="activation_3"
    )(x)

    # ========================================================
    # SE ATTENTION
    # ========================================================

    channels = int(x.shape[-1])

    se = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)

    se = layers.Dense(
        max(channels // 16, 1),
        activation="relu",
        name="dense"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid",
        name="dense_1"
    )(se)

    se = layers.Reshape(
        (1, 1, channels),
        name="reshape"
    )(se)

    x = layers.Multiply(
        name="multiply"
    )([x, se])

    # ========================================================
    # FINAL FEATURE BLOCK
    # ========================================================

    x = layers.Conv2D(
        512,
        (3, 3),
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
        name="activation_4"
    )(x)

    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d_1"
    )(x)

    x = layers.Dense(
        256,
        activation=tf.keras.activations.gelu,
        name="dense_2"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    x = layers.Dense(
        128,
        activation=tf.keras.activations.gelu,
        name="modality_dense"
    )(x)

    x = layers.BatchNormalization(
        name="modality_batch_normalization"
    )(x)

    outputs = layers.Dense(
        3,
        activation="softmax",
        name="modality_probability"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Medical_Image_Modality_Classifier"
    )

    return model
