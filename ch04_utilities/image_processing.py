import numpy as np


def convert_to_grayscale(I, normalized=False):
    """
    Convert an image (from imageio.imread) to grayscale.
    Handles grayscale, RGB, RGBA, and palette-expanded RGB.

    [inputs]
    I : np.ndarray
        Input image loaded by imageio.imread()

    normalized : bool (default=False)
        If True, output is normalized to the range [0, 1].

    [output]
    gray : np.ndarray
        2D grayscale image
    """

    # -------------------------------------------------
    # If already grayscale (H, W), return float version
    # -------------------------------------------------
    if I.ndim == 2:
        gray = I.astype(np.float64)

    # --------------------------------
    # If multi-channel image (H, W, C)
    # --------------------------------
    elif I.ndim == 3:
        # Keep only first 3 channels (drop alpha or extras)
        RGB = I[..., :3].astype(np.float64)

        # Convert to grayscale using luminance formula
        gray = np.dot(RGB, [0.299, 0.587, 0.114])

    else:
        raise ValueError(f"Unsupported image shape: {I.shape}")

    # --------------------------------
    # Normalize to [0, 1] if requested
    # --------------------------------
    if normalized:
        min_val = gray.min()
        max_val = gray.max()

        # Avoid division by zero
        if max_val > min_val:
            gray = (gray - min_val) / (max_val - min_val)
        else:
            gray = np.zeros_like(gray)

    return gray

