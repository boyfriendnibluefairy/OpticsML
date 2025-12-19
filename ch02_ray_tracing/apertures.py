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