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

    def _to_uint8_view(im: np.ndarray) -> np.ndarray:
        """For drawing only: convert to uint8 0..255 if needed, without changing geometry."""
        if im.dtype == np.uint8:
            return im.copy()
        # Robust-ish normalization for drawing:
        imf = im.astype(np.float32)
        mn, mx = float(np.nanmin(imf)), float(np.nanmax(imf))
        if mx > mn:
            imf = (imf - mn) / (mx - mn)
        imf = np.clip(imf, 0.0, 1.0)
        return (imf * 255.0 + 0.5).astype(np.uint8)

    def _ensure_color3(im: np.ndarray) -> np.ndarray:
        """Ensure BGR 3-channel for drawing with colored borders."""
        if im.ndim == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 1:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] >= 3:
            # Keep first 3 channels
            return im[:, :, :3].copy()
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
        # sequence like (B,G,R) or (R,G,B). We'll assume BGR if length 3.
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

        # Compute intersection with image bounds
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
        # cv2.resize expects (W,H)
        if cut.ndim == 2:
            return cv2.resize(cut, (resize, resize), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(cut, (resize, resize), interpolation=cv2.INTER_AREA)

    def _solid_rect(im_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int, bgr: Tuple[int, int, int]) -> None:
        """Draw a solid rectangle (no fill)."""
        cv2.rectangle(im_bgr, (x0, y0), (x1, y1), bgr, thickness=border_size, lineType=cv2.LINE_AA)

    def _dashed_rect(
        im_bgr: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        bgr: Tuple[int, int, int],
        dash_len: int = 8,
        gap_len: int = 6,
    ) -> None:
        """Draw a dashed/broken-line rectangle."""
        # Top & bottom edges
        def _dash_line(p0: Tuple[int, int], p1: Tuple[int, int]) -> None:
            xA, yA = p0
            xB, yB = p1
            length = int(round(np.hypot(xB - xA, yB - yA)))
            if length <= 0:
                return
            dx = (xB - xA) / length
            dy = (yB - yA) / length
            s = 0
            while s < length:
                e = min(length, s + dash_len)
                xs = int(round(xA + dx * s))
                ys = int(round(yA + dy * s))
                xe = int(round(xA + dx * e))
                ye = int(round(yA + dy * e))
                cv2.line(im_bgr, (xs, ys), (xe, ye), bgr, thickness=border_size, lineType=cv2.LINE_AA)
                s += dash_len + gap_len

        _dash_line((x0, y0), (x1, y0))  # top
        _dash_line((x0, y1), (x1, y1))  # bottom
        _dash_line((x0, y0), (x0, y1))  # left
        _dash_line((x1, y0), (x1, y1))  # right

    def _rect_from_center(center_xy: Tuple[float, float]) -> Tuple[int, int, int, int]:
        """Return rectangle bounds (x0,y0,x1,y1) inclusive-ish for drawing."""
        cx, cy = center_xy
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        x0 = cx_i - half
        y0 = cy_i - half
        x1 = x0 + box_size - 1
        y1 = y0 + box_size - 1
        return x0, y0, x1, y1

    def _rect_for_field10() -> Tuple[int, int, int, int]:
        """field10: upper-right corner of cutout coincides with image upper-right."""
        x1 = W - 1
        y0 = 0
        x0 = x1 - (box_size - 1)
        y1 = y0 + (box_size - 1)
        return x0, y0, x1, y1

    # ---- compute field centers ------------------------------------------------

    # field0: image center
    cx0 = (W - 1) / 2.0
    cy0 = (H - 1) / 2.0

    # upper-right point
    urx = float(W - 1)
    ury = 0.0

    # vector from center to upper-right and r = its length
    vx = urx - cx0
    vy = ury - cy0
    r = float(np.hypot(vx, vy))
    if r == 0.0:
        # degenerate 1x1 image
        cx7, cy7 = cx0, cy0
    else:
        # field7: 0.7*r along direction to upper-right
        cx7 = cx0 + 0.7 * vx
        cy7 = cy0 + 0.7 * vy

    # field10 center: based on cutout whose upper-right corner matches image upper-right
    x0_10, y0_10, x1_10, y1_10 = _rect_for_field10()
    cx10 = (x0_10 + x1_10) / 2.0
    cy10 = (y0_10 + y1_10) / 2.0

    # ---- extract cutouts ------------------------------------------------------

    field0 = _resize_cutout(_extract_cutout((cx0, cy0)))
    field7 = _resize_cutout(_extract_cutout((cx7, cy7)))
    field10 = _resize_cutout(_extract_cutout((cx10, cy10)))

    # ---- draw box marks -------------------------------------------------------

    draw_base = _ensure_color3(_to_uint8_view(img))
    box_marks = draw_base.copy()

    # Rect bounds
    x0_0, y0_0, x1_0, y1_0 = _rect_from_center((cx0, cy0))
    x0_7, y0_7, x1_7, y1_7 = _rect_from_center((cx7, cy7))
    # field10 already computed
    # x0_10, y0_10, x1_10, y1_10

    if border_color is None:
        red = (0, 0, 255)
        _dashed_rect(box_marks, x0_0, y0_0, x1_0, y1_0, red)
        _dashed_rect(box_marks, x0_7, y0_7, x1_7, y1_7, red)
        _dashed_rect(box_marks, x0_10, y0_10, x1_10, y1_10, red)
    else:
        if len(border_color) != 3:
            raise ValueError("border_color must be a sequence of 3 colors (field0, field7, field10) or None.")
        c0 = _color_to_bgr(border_color[0])
        c7 = _color_to_bgr(border_color[1])
        c10 = _color_to_bgr(border_color[2])
        _solid_rect(box_marks, x0_0, y0_0, x1_0, y1_0, c0)
        _solid_rect(box_marks, x0_7, y0_7, x1_7, y1_7, c7)
        _solid_rect(box_marks, x0_10, y0_10, x1_10, y1_10, c10)

    return box_marks, field0, field7,


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