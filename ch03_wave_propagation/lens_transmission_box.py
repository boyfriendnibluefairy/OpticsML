import numpy as np
import torch
from typing import Tuple, Optional

### this function is called by generate_object_point_rays_v2()
def pupil_points_unit_disk(pattern: str, n_rays: int, nx: int, ny: int) -> np.ndarray:
    """
    Generates pupil sample points on a *unit* disk, shape (N,2), where x,y in [-1,1] and x^2+y^2<=1.
        for cartesian pattern: this applies (nx,ny) grid then masks to unit disk
        for fibonacci/random pattern: this applies n_rays samples
    """
    pattern_l = pattern.lower()

    if pattern_l == "cartesian":
        x = np.linspace(-1.0, 1.0, int(nx))
        y = np.linspace(-1.0, 1.0, int(ny))
        xx, yy = np.meshgrid(x, y, indexing="xy")
        mask = (xx * xx + yy * yy) <= 1.0
        pts = np.stack([xx[mask], yy[mask]], axis=1)
        return pts

    if n_rays <= 0:
        raise ValueError("n_rays must be > 0 for fibonacci/random patterns.")

    if pattern_l == "fibonacci":
        k = np.arange(n_rays, dtype=np.float64)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        r = np.sqrt((k + 0.5) / n_rays)
        theta = k * golden_angle
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        return np.stack([px, py], axis=1)

    if pattern_l == "random":
        r = np.sqrt(np.random.rand(n_rays))
        theta = 2.0 * np.pi * np.random.rand(n_rays)
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        return np.stack([px, py], axis=1)

    raise ValueError("pattern must be 'cartesian', 'fibonacci', or 'random'.")