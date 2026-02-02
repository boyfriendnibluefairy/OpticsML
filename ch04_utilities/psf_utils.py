"""
    The functions in psf_utils.py are used by psf/get_fft_psf_v18()
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
### for zero padding
import torch.nn.functional as F
try:
    from typing import Literal  # Python 3.8+
except ImportError:
    # For Python 3.7 or older environments
    from typing_extensions import Literal
### for view() interpolation
from scipy.ndimage import zoom
### if log_enabled is True for view()
from matplotlib.colors import LogNorm
### for _coerced_psf2d()
import math
### for correct_tilt_v2()
from numpy.typing import NDArray
### used by display_fft_psf() to set ticks similar to Zemax OpticStudio
from matplotlib.ticker import FixedLocator, FuncFormatter


def calculate_sphere_from_image_chief_ray(
        image_chief_ray:torch.Tensor,
        z_exit_pupil: float | int | torch.Tensor
        ) -> tuple[float, float, float, float]:
    """Calculate sphere radius and center.
    GMO:
       The center is the intersection of chief ray on the image plane.
       The radius is assumed to be the path from center to exit pupil.

    [inputs]
    image_chief_ray : torch.Tensor
        Ray tensor with columns [x, y, z, X, Y, Z, wvln, opl, total_OPL].
        Accepts shape (9,), (1,9), or (N,9) but must represent exactly one chief ray.
    pupil_z : float | int | torch.Tensor
        z-position of the pupil center (same units as ray coordinates, e.g., mm).

    [output]
    (x, y, z, R) : tuple[float, float, float, float]
        Center of the reference sphere (at the chief ray intercept point) and radius.
    """
    if image_chief_ray.dim() == 1:
        if image_chief_ray.numel() != 9:
            raise ValueError("Expected a 9-element chief ray vector.")
        ray = image_chief_ray.view(1, 9)
    elif image_chief_ray.dim() == 2:
        if image_chief_ray.size(1) != 9:
            raise ValueError("Expected ray tensor with 9 columns.")
        if image_chief_ray.size(0) != 1:
            raise ValueError("Chief ray must be traced alone (got batch > 1).")
        ray = image_chief_ray
    else:
        raise ValueError("image_chief_ray must be a 1D or 2D tensor.")

    # Extract intersection point at the image plane
    x, y, z = ray[0, 0], ray[0, 1], ray[0, 2]

    # Ensure pupil_z is a tensor on the same device/dtype
    pupil_z_t = torch.as_tensor(z_exit_pupil, device=ray.device, dtype=ray.dtype)

    # Reference sphere radius: distance from pupil center (0,0,pupil_z) to (x,y,z)
    R = torch.sqrt(x**2 + y**2 + (z - pupil_z_t)**2)

    # Return plain Python floats
    return (float(x.item()), float(y.item()), float(z.item()), float(R.item()))
