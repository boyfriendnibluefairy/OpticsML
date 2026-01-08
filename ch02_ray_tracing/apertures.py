import numpy as np

def circular_pupil(xx: np.ndarray, yy: np.ndarray, radius: float) -> np.ndarray:
    """
    Binary circular aperture:
        P(x,y) = 1 inside radius, 0 outside

    [inputs]
        xx, yy : (N,N)
            Spatial coordinates (meters).
        radius : float
            Aperture radius (meters).

    [output]
        P : (N,N) float64
            Pupil mask.
    """
    return ((xx**2 + yy**2) <= radius**2).astype(np.float64)


def circular_hard_aperture(size, relative_radius):
    """
    Create a centered circular hard aperture mask.

    [inputs]
        size : int or tuple[int, int]
            Output array size. If int, creates a square (size, size).
            If tuple, interpreted as (rows, cols).
        relative_radius : float
            Radius of the circular opening relative to half of the *smaller*
            array dimension. Typical range: [0, 1].
            - 1.0 means radius = min(rows, cols)/2 (touches the nearest edge).
            - 0.5 means radius = min(rows, cols)/4, etc.

    [output]
        aperture : np.ndarray
            2D array of shape (rows, cols), dtype np.float64.
            Opaque region: 0.0, Opening: 1.0.
    """
    # Validate / normalize size
    if isinstance(size, int):
        rows, cols = size, size
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        rows, cols = int(size[0]), int(size[1])
    else:
        raise TypeError("size must be an int or a tuple/list of length 2 (rows, cols).")

    if rows <= 0 or cols <= 0:
        raise ValueError("size dimensions must be positive.")

    # Validate relative_radius
    rr = float(relative_radius)
    if rr < 0.0:
        raise ValueError("relative_radius must be >= 0.")
    # Allow rr > 1.0 (it just saturates to all-ones), but clamp for safety/clarity.
    rr = min(rr, 1.0)

    # Coordinate grid centered at the array center
    # Using pixel-centered coordinates: indices map to [-0.5*(N-1), ..., +0.5*(N-1)]
    y = np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0
    x = np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Physical radius in pixels, relative to half of the smaller dimension
    r_max = min(rows, cols) / 2.0
    r = rr * r_max

    # Build mask
    aperture = ((X * X + Y * Y) <= (r * r)).astype(np.float64)
    return aperture
