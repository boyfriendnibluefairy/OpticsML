import numpy as np
import torch
from typing import List, Tuple

def generate_object_points(
    M: int,
    z_obj: float = -200.0,      # [mm]
    L: float = 6.0,             # [mm]
    pixel_pitch: float = 3.0e-6 # [m]
) -> List[Tuple[float, float, float]]:
    """
    Generate object-plane sampling points representing pixel centers.

    Each object point corresponds to the center of one pixel in an M×M grid,
    centered at (0, 0, z_obj).

    [inputs]
        M : int
            Object grid size. M=1024 means 1024×1024 object pixels.

        z_obj : float
            z-position of the object plane [mm].
            Default: -200 mm.

        L : float
            Physical side length of the object grid [mm].
            Default: 6.0 mm.

        pixel_pitch : float
            Pixel pitch [meters].
            Default: 3.0e-6 m.

    [output]
        object_points : list of (x0, y0, z0)
            Each tuple is the center coordinate of an object pixel in mm.

    Usage:
        object_points = generate_object_points(M=1024)
        for (x0, y0, z0) in object_points:
            ...
    """

    if M <= 0:
        raise ValueError("M must be a positive integer.")

    # Convert pixel pitch from meters to millimeters
    pixel_pitch_mm = pixel_pitch * 1e3

    # If L is inconsistent with M and pixel_pitch, L takes priority
    # and pixel centers are uniformly distributed across [-L/2, L/2]
    dx = L / M

    # Coordinate vector for pixel centers
    coords = (np.arange(M) - (M - 1) / 2.0) * dx

    # Build 2D grid of object points
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    # Flatten and attach z-coordinate
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = np.full_like(x_flat, z_obj, dtype=np.float64)

    # Pack into list of tuples
    object_points = list(zip(
        x_flat.astype(float),
        y_flat.astype(float),
        z_flat.astype(float)
    ))

    return object_points
