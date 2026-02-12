from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
from typing import Iterable, Optional, Sequence, Tuple, Union


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


def extract_field_points(
    img: np.ndarray,
    box_size: int = 13,
    border_size: int = 3,
    border_color: Optional[Sequence[Union[str, Sequence[int]]]] = ("blue", "green", "yellow"),
    resize: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3 field cutouts (field0, field7, field10) from `img` and return a marked copy.

    [inputs]
        img:
            Input image. Supports HxW (grayscale) or HxWxC (color), dtype uint8/float/etc.
        box_size:
            Size of the square cutouts in pixels (box_size x box_size).
        border_size:
            Border thickness in pixels for solid borders, or dash "stroke width" for dashed.
        border_color:
            If not None: a sequence of 3 colors (for field0, field7, field10).
              Each can be a string name ("blue","green","yellow", etc.) or a BGR/RGB tuple/list.
              Strings are mapped to BGR for OpenCV drawing.
            If None: all boxes are drawn as dashed/broken RED.
        resize:
            If None: each extracted field cutout remains box_size x box_size.
            If not None: each cutout is resized to resize x resize.

    [outputs]
        box_marks : same size as `img`, with 3 rectangular box marks (no fill)
        field0    : cutout centered at image center
        field7    : cutout centered at 0.7*r along the line from center -> upper-right corner
        field10   : cutout whose upper-right corner coincides with image upper-right corner

    Notes on coordinates:
        - Image origin is top-left. x increases to the right, y increases downward.
        - "Upper-right corner" is (W-1, 0).
        - field7 center is computed using the Euclidean distance r between center and upper-right,
          then placed at 0.7*r along that direction (i.e., 70% of the way from center to upper-right).

    Behavior near borders:
        If a requested cutout extends outside the image, it is zero-padded to keep size box_size x box_size.

    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray")
    if img.ndim not in (2, 3):
        raise ValueError("img must be HxW or HxWxC")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")
    if border_size <= 0:
        raise ValueError("border_size must be > 0")
    if resize is not None and resize <= 0:
        raise ValueError("resize must be None or > 0")

    H, W = img.shape[:2]
    half = box_size // 2

    # ---- internal functions -------------------------------------------------------------

    def _to_uint8_view_preserve(im: np.ndarray) -> np.ndarray:
        """
        Convert to uint8 for drawing while preserving appearance.
        - No min-max normalization (avoids contrast/color shift).
        """
        if im.dtype == np.uint8:
            return im.copy()

        if np.issubdtype(im.dtype, np.floating):
            imf = np.nan_to_num(im.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mx = float(imf.max()) if imf.size else 0.0
            # If it's likely in [0,1], scale up
            if mx <= 1.0 + 1e-6:
                imf *= 255.0
            imf = np.clip(imf, 0.0, 255.0)
            return (imf + 0.5).astype(np.uint8)

        if np.issubdtype(im.dtype, np.integer):
            info = np.iinfo(im.dtype)
            imf = im.astype(np.float32) / float(info.max) * 255.0
            imf = np.clip(imf, 0.0, 255.0)
            return (imf + 0.5).astype(np.uint8)

        imf = np.clip(im.astype(np.float32), 0.0, 255.0)
        return (imf + 0.5).astype(np.uint8)

    def _ensure_color3_for_drawing(im_u8: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Ensure BGR 3-channel for OpenCV drawing.
        Returns (im_bgr, order_tag) where order_tag is 'GRAY' or 'RGB'.
        Since you load via imageio, color inputs are RGB -> convert to BGR for cv2.
        """
        if im_u8.ndim == 2:
            return cv2.cvtColor(im_u8, cv2.COLOR_GRAY2BGR), "GRAY"

        if im_u8.shape[2] == 1:
            return cv2.cvtColor(im_u8, cv2.COLOR_GRAY2BGR), "GRAY"

        if im_u8.shape[2] >= 3:
            im_rgb = im_u8[:, :, :3].copy()          # imageio => RGB
            im_bgr = im_rgb[:, :, ::-1].copy()       # RGB -> BGR for cv2
            return im_bgr, "RGB"

        raise ValueError("Unsupported channel count")

    def _color_to_bgr(c: Union[str, Sequence[int]]) -> Tuple[int, int, int]:
        """Map a color spec to BGR tuple (OpenCV)."""
        if isinstance(c, str):
            name = c.strip().lower()
            table = {
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "red": (0, 0, 255),
                "cyan": (255, 255, 0),
                "magenta": (255, 0, 255),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "orange": (0, 165, 255),
                "purple": (128, 0, 128),
            }
            if name not in table:
                raise ValueError(f"Unknown color name '{c}'. Use BGR tuple or a known name.")
            return table[name]
        if isinstance(c, (list, tuple)) and len(c) == 3:
            b, g, r = c
            return (int(b), int(g), int(r))
        raise TypeError("Color must be a string (e.g. 'blue') or a length-3 tuple/list (B,G,R).")

    def _extract_cutout(center_xy: Tuple[float, float]) -> np.ndarray:
        """Extract a box_size x box_size cutout centered at (cx,cy). Pads with zeros if needed."""
        cx, cy = center_xy
        cx_i = int(round(cx))
        cy_i = int(round(cy))

        x0 = cx_i - half
        y0 = cy_i - half
        x1 = x0 + box_size
        y1 = y0 + box_size

        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(W, x1)
        iy1 = min(H, y1)

        if img.ndim == 2:
            out = np.zeros((box_size, box_size), dtype=img.dtype)
        else:
            out = np.zeros((box_size, box_size, img.shape[2]), dtype=img.dtype)

        ox0 = ix0 - x0
        oy0 = iy0 - y0
        ox1 = ox0 + (ix1 - ix0)
        oy1 = oy0 + (iy1 - iy0)

        out[oy0:oy1, ox0:ox1, ...] = img[iy0:iy1, ix0:ix1, ...]
        return out

    def _resize_cutout(cut: np.ndarray) -> np.ndarray:
        if resize is None:
            return cut
        if cut.ndim == 2:
            return cv2.resize(cut, (resize, resize), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(cut, (resize, resize), interpolation=cv2.INTER_AREA)

    def _solid_rect(im_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int, bgr: Tuple[int, int, int]) -> None:
        cv2.rectangle(im_bgr, (x0, y0), (x1, y1), bgr, thickness=border_size, lineType=cv2.LINE_AA)

    def _rect_from_center(center_xy: Tuple[float, float]) -> Tuple[int, int, int, int]:
        cx, cy = center_xy
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        x0 = cx_i - half
        y0 = cy_i - half
        x1 = x0 + box_size - 1
        y1 = y0 + box_size - 1
        return x0, y0, x1, y1

    def _rect_for_field10() -> Tuple[int, int, int, int]:
        x1 = W - 1
        y0 = 0
        x0 = x1 - (box_size - 1)
        y1 = y0 + (box_size - 1)
        return x0, y0, x1, y1

    # ---- compute field centers ------------------------------------------------

    cx0 = (W - 1) / 2.0
    cy0 = (H - 1) / 2.0

    urx = float(W - 1)
    ury = 0.0

    vx = urx - cx0
    vy = ury - cy0
    r = float(np.hypot(vx, vy))
    if r == 0.0:
        cx7, cy7 = cx0, cy0
    else:
        cx7 = cx0 + 0.7 * vx
        cy7 = cy0 + 0.7 * vy

    x0_10, y0_10, x1_10, y1_10 = _rect_for_field10()
    cx10 = (x0_10 + x1_10) / 2.0
    cy10 = (y0_10 + y1_10) / 2.0

    # ---- extract cutouts ------------------------------------------------------

    field0 = _resize_cutout(_extract_cutout((cx0, cy0)))
    field7 = _resize_cutout(_extract_cutout((cx7, cy7)))
    field10 = _resize_cutout(_extract_cutout((cx10, cy10)))

    # ---- draw box marks (FIXED: preserve RGB appearance) ----------------------

    draw_u8 = _to_uint8_view_preserve(img)
    draw_bgr, order_tag = _ensure_color3_for_drawing(draw_u8)

    box_marks_bgr = draw_bgr.copy()

    x0_0, y0_0, x1_0, y1_0 = _rect_from_center((cx0, cy0))
    x0_7, y0_7, x1_7, y1_7 = _rect_from_center((cx7, cy7))
    # field10: (x0_10, y0_10, x1_10, y1_10) already computed

    if border_color is None:
        red = (0, 0, 255)
        _solid_rect(box_marks_bgr, x0_0, y0_0, x1_0, y1_0, red)
        _solid_rect(box_marks_bgr, x0_7, y0_7, x1_7, y1_7, red)
        _solid_rect(box_marks_bgr, x0_10, y0_10, x1_10, y1_10, red)
    else:
        if len(border_color) != 3:
            raise ValueError("border_color must be a sequence of 3 colors (field0, field7, field10) or None.")
        c0 = _color_to_bgr(border_color[0])
        c7 = _color_to_bgr(border_color[1])
        c10 = _color_to_bgr(border_color[2])
        _solid_rect(box_marks_bgr, x0_0, y0_0, x1_0, y1_0, c0)
        _solid_rect(box_marks_bgr, x0_7, y0_7, x1_7, y1_7, c7)
        _solid_rect(box_marks_bgr, x0_10, y0_10, x1_10, y1_10, c10)

    # Convert back to RGB if input was RGB
    if order_tag == "RGB":
        box_marks = box_marks_bgr[:, :, ::-1].copy()  # BGR -> RGB
    else:
        box_marks = box_marks_bgr

    return box_marks, field0, field7, field10


def generate_checker_board(N: int, n_row_blocks: int) -> np.ndarray:
    """
    Create an uint8 NxN checkerboard image.

    Notes:
    - There are n_row_blocks blocks across (horizontal) and n_row_blocks blocks down (vertical).
    - Top-left block is WHITE, then alternates BLACK/WHITE.
    - If N is not divisible by n_row_blocks, the last blocks will be slightly larger
      because integer division + remainder is handled via index mapping.

    [output]
        2D uint8 array with values 0 (black) and 255 (white).
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not isinstance(n_row_blocks, int) or n_row_blocks <= 0:
        raise ValueError("n_row_blocks must be a positive integer.")
    if n_row_blocks > N:
        raise ValueError("n_row_blocks cannot be larger than N (would create <1-pixel blocks).")

    # Map each pixel to a block index along y and x: 0..n_row_blocks-1
    # This handles non-divisible N cleanly (blocks differ by at most 1 pixel).
    y_block = (np.arange(N) * n_row_blocks) // N
    x_block = (np.arange(N) * n_row_blocks) // N

    # Parity pattern: (y_block + x_block) even -> white (top-left is white), odd -> black
    pattern = (y_block[:, None] + x_block[None, :]) % 2  # 0 or 1

    # Convert to uint8 image: 0->255 (white), 1->0 (black)
    img = np.where(pattern == 0, 255, 0).astype(np.uint8)
    return img


def pad_field(U: np.ndarray, n_pad_pixels: int) -> np.ndarray:
    """
    Zero-pad a 2D complex field symmetrically.

    [inputs]
        U : (N,N) complex ndarray
            Input complex field.
        n_pad_pixels : int
            Number of pixels to pad on EACH side (top/bottom/left/right).
            Example: N=2048, n_pad_pixels=256 -> padded size = 2560.

    [output]
        U_padded : (N+2p, N+2p) complex ndarray
            Zero-padded field.
    """
    if n_pad_pixels <= 0:
        return U

    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError("pad_field expects a 2D array.")
    if U.shape[0] != U.shape[1]:
        raise ValueError("pad_field expects a square array (N x N).")

    p = int(n_pad_pixels)
    # Pad with zeros (vacuum outside the original support).
    return np.pad(U, pad_width=((p, p), (p, p)), mode="constant", constant_values=0.0)


def unpad_field(U_padded: np.ndarray, n_pad_pixels: int) -> np.ndarray:
    """
    Remove symmetric padding previously applied by pad_field.

    [input]
        U_padded : (N+2p, N+2p) complex ndarray
            Padded field.
        n_pad_pixels : int
            Padding in pixels on each side.

    [output]
        U : (N,N) complex ndarray
            Cropped (unpadded) field.
    """
    if n_pad_pixels <= 0:
        return U_padded

    U_padded = np.asarray(U_padded)
    if U_padded.ndim != 2:
        raise ValueError("unpad_field expects a 2D array.")
    if U_padded.shape[0] != U_padded.shape[1]:
        raise ValueError("unpad_field expects a square array (N x N).")

    p = int(n_pad_pixels)
    if 2 * p >= U_padded.shape[0]:
        raise ValueError("n_pad_pixels is too large for the current array size.")

    return U_padded[p:-p, p:-p]
