# ============================================================
# preprocessing.py
# SHARED X-RAY PREPROCESSING
# ============================================================

import cv2
import numpy as np
from PIL import Image


# ============================================================
# TARGET IMAGE SIZE
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# X-RAY PREPROCESSING
# ============================================================

def preprocess_xray(
    image,
    target_size=IMAGE_SIZE,
    use_clahe=True
):
    """
    Preprocess a chest X-ray.

    Pipeline:
        1. Convert to grayscale
        2. Optional CLAHE
        3. Preserve aspect ratio
        4. Resize
        5. Pad to target size
        6. Normalize to [0, 1]
        7. Convert grayscale to 3 channels

    Returns:
        NumPy array with shape:
        (height, width, 3)
    """

    # --------------------------------------------------------
    # PIL IMAGE
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        image = image.convert("L")

        image = np.asarray(
            image,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # NUMPY IMAGE
    # --------------------------------------------------------

    else:

        image = np.asarray(
            image
        )

        if image.ndim == 3:

            if image.shape[-1] == 3:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2GRAY
                )

            elif image.shape[-1] == 4:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2GRAY
                )

        image = image.astype(
            np.uint8
        )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if use_clahe:

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        image = clahe.apply(
            image
        )

    # --------------------------------------------------------
    # TARGET SIZE
    # --------------------------------------------------------

    target_height = int(
        target_size[0]
    )

    target_width = int(
        target_size[1]
    )

    # --------------------------------------------------------
    # ORIGINAL SIZE
    # --------------------------------------------------------

    height, width = image.shape

    # --------------------------------------------------------
    # PRESERVE ASPECT RATIO
    # --------------------------------------------------------

    scale = min(
        target_width / width,
        target_height / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # PAD
    # --------------------------------------------------------

    canvas = np.zeros(
        (
            target_height,
            target_width
        ),
        dtype=np.uint8
    )

    y_offset = (
        target_height - new_height
    ) // 2

    x_offset = (
        target_width - new_width
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = image

    image = canvas

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # GRAYSCALE -> RGB
    # --------------------------------------------------------

    image = np.stack(
        [
            image,
            image,
            image
        ],
        axis=-1
    )

    return image


# ============================================================
# PREPROCESS WITH BATCH DIMENSION
# ============================================================

def preprocess_xray_batch(
    image,
    target_size=IMAGE_SIZE,
    use_clahe=True
):

    processed = preprocess_xray(
        image,
        target_size,
        use_clahe
    )

    return np.expand_dims(
        processed,
        axis=0
    )
