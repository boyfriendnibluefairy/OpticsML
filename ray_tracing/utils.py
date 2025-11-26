# for create_square_aperture()
import numpy as np

# create square aperture object
def create_square_aperture(delta, M, square_pixels):
    """
    Create a 2D square aperture on an M x M grid.

    [input]
    delta : float
        Sampling interval in the aperture plane (meters).
        (Included for physical interpretation; not used numerically here.)
    M : int
        Number of samples in x and y (grid size). Result is M x M.
    square_pixels : int
        Side length of the square aperture in pixels
        (e.g., 22 in Poon & Liu's example).

    output
    OB : np.ndarray (float)
        2D real array with ones inside the square aperture and zeros elsewhere.
    """

    # Initialize an M x M array of zeros (opaque everywhere)
    OB = np.zeros((M, M), dtype=np.float64)

    # Center index where the square aperture will be placed
    start = (M - square_pixels) // 2
    end = start + square_pixels  # Python slicing: end index is exclusive

    # Set central square region to 1 (transparent aperture)
    OB[start:end, start:end] = 1.0

    return OB