import numpy as np
from PIL import Image


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


def load_grayscale_as_amplitude(path: str, N: int, invert: bool = False) -> np.ndarray:
    """
    Load an image, convert to grayscale, resize to NxN, and normalize to [0,1].

    [input]
        path : str
            Path to input image.
        N : int
            Output grid size (NxN).
        invert : bool
            If True, use (1 - normalized_image). Helpful for some test targets.

    [output]
        amp : (N,N) float64
            Real-valued amplitude mask in [0,1].
    """
    img = Image.open(path).convert("L").resize((N, N), Image.Resampling.LANCZOS)
    amp = np.asarray(img, dtype=np.float64)

    # Normalize to [0,1]
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()

    # Optional inversion
    if invert:
        amp = 1.0 - amp

    return amp


def make_xy_grid(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create a centered coordinate grid for a square computational window.

    [inputs]
        N : int
            Number of samples along x and y (N x N).
        L : float
            Physical side length of the window (meters).

    [output]
        x : (N,) float64
            1D x coordinates (meters).
        xx, yy : (N,N) float64
            Meshgrids of x and y (meters).
        dx : float
            Sample spacing (meters).
    """
    dx = L / N
    x = (np.arange(N) - N // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return x, xx, yy, dx


def center_crop(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    H, W = arr.shape
    h, w = out_shape
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return arr[y0:y0+h, x0:x0+w]