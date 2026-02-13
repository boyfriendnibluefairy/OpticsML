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


def opticstudio_effective_sampling(requested_num_rays: int) -> int:
    """
    Converts the sampling grid size at the exit pupil plane to a lower approximate
    effective pupil sampling size. The logic behind this is to re-adjust the sampling grid size
    at the exit pupil plane so that the PSF calculated at image plane do not lose its resolution
    when FFT is performed on the exit pupil plane. When the grid size at the exit pupil is
    increased while the diameter is fixed, many optical rays "fall out" of that diameter.
    Therefore, we re-adjust from requested_num_rays to a smaller value so that more rays
    will be sampled again at the exit pupil. Please see the reference below:

    https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/FFT_PSF.html
    """
    eff = int(np.floor(32 * (2 ** ((np.log2(requested_num_rays) - 5) / 2))))
    return max(32, min(requested_num_rays, eff))



def display_fft_psf(
    fft_psf_real_t: torch.Tensor,
    threshold: float = 0.25,          # fraction of peak (0..1); e.g., 0.25 = 25% of max
    num_points: int = 128,            # interpolation size for display
    working_FNO: float = 4.0,
    grid_size: int = 128,
    num_rays: int = 128,
    wavelength: float = 0.550,        # micrometers
    log_enabled: bool = True,
    center_crop_extent: float | None = None,
    figsize=(5, 4),
):
    """
    Visualize a PSF (Torch tensor) computed on a square FFT grid.
    [input]
        fft_psf_real : torch.Tensor - converted to numpy.ndarray
        threshold : float - The threshold value for determining non-zero
                elements in the PSF matrix. Default is 0.25.
        num_points (int, optional): The number of points used for
                interpolating the PSF for smoother visualization. Defaults to 128.
        working_FNO : float - effective F-number of the optical system
        grid_size : int - number of pixels along one axis, assume that PSF is square
        num_rays : int - pupil sampling density
        wavelength : float - wavelength of the source
        log_enabled : bool - if True, extreme values of PSF are handled by log() function

    [output]
    fig, ax : matplotlib Figure and Axes
    """
    # ---- robust input handling (fix for shape == () errors) ----
    psf_np = _coerce_to_psf2d(fft_psf_real_t)
    H, W = psf_np.shape
    original_size = (H, W)

    if psf_np.ndim != 2:
        raise ValueError(f"PSF must be 2D; got shape {psf_np.shape}")
    # If complex sneaks in, use the real intensity
    if np.iscomplexobj(psf_np):
        psf_np = psf_np.real

    H, W = psf_np.shape
    original_size = (H, W)

    # ---- find peak and a reasonable crop window ----
    peak_flat = np.argmax(psf_np)
    peak_y, peak_x = np.unravel_index(peak_flat, psf_np.shape)
    peak_val = psf_np[peak_y, peak_x]

    # Build a mask using a relative threshold of the peak
    # If absolute thresholding is wanted, change (threshold * peak_val) -> threshold
    mask = psf_np >= (threshold * peak_val)
    nz = np.argwhere(mask)

    if nz.size > 0:
        (min_y, min_x), (max_y, max_x) = nz.min(axis=0), nz.max(axis=0)
        # Make it square and centered on the peak
        size_y = max_y - min_y + 1
        size_x = max_x - min_x + 1
        size = max(size_y, size_x)

        # Center square around the peak
        half = size // 2
        min_y = int(max(0, peak_y - half))
        max_y = int(min(H, min_y + size))
        min_x = int(max(0, peak_x - half))
        max_x = int(min(W, min_x + size))

        # Re-adjust start if end hit the boundary
        min_y = max(0, max_y - size)
        min_x = max(0, max_x - size)
    else:
        # Fallback: centered square around the peak with half the smaller dimension
        size = min(H, W) // 2
        half = size // 2
        min_y = max(0, peak_y - half)
        max_y = min(H, min_y + size)
        min_x = max(0, peak_x - half)
        max_x = min(W, min_x + size)
        min_y = max(0, max_y - size)
        min_x = max(0, max_x - size)

    psf_crop = psf_np[min_y:max_y, min_x:max_x]

    # ---- optional warning for very heavy interpolation ----
    oversampling_factor = num_points / max(1, psf_crop.shape[0])
    if oversampling_factor > 3:
        print(
            f"[display_fft_psf] High view oversampling factor "
            f"({oversampling_factor:.2f}). Visuals may be overly smooth."
        )

    # ---- physical extent (μm) for axes ----
    x_extent, y_extent = get_psf_units(
        image=psf_crop,
        working_FNO=working_FNO,
        grid_size=grid_size,
        num_rays=num_rays,
        wavelength=wavelength,
    )
    extent = [-x_extent / 2, x_extent / 2, -y_extent / 2, y_extent / 2]
    print(f"extent = {extent}")
    print(f"grid_size = {grid_size}")
    print(f"num_rays = {num_rays}")

    # ---- interpolation for smooth display ----
    #psf_smooth_np = interpolate_psf(psf_crop, n=num_points)  # GMO
    if center_crop_extent is not None: # GMO
        psf_smooth_np = interpolate_psf(psf_crop, n=num_points*8)
    else:
        psf_smooth_np = interpolate_psf(psf_crop, n=num_points)

    # ---- optional physical center crop in micrometers ----
    if center_crop_extent is not None:
        center_crop_extent = float(center_crop_extent)
        if center_crop_extent <= 0:
            raise ValueError("center_crop_extent must be > 0 (micrometers) or None.")
        if center_crop_extent > min(x_extent, y_extent):
            # You can choose: clamp, warn, or raise. Clamping is usually friendliest.
            center_crop_extent = min(x_extent, y_extent)

        Hs, Ws = psf_smooth_np.shape

        # Convert requested physical span (µm) -> pixels in the smoothed grid
        um_per_pix_x = x_extent / Ws
        um_per_pix_y = y_extent / Hs
        crop_w = int(round(center_crop_extent / um_per_pix_x))
        crop_h = int(round(center_crop_extent / um_per_pix_y))

        crop_w = max(1, min(Ws, crop_w))
        crop_h = max(1, min(Hs, crop_h))

        # center crop (around the middle of the *current displayed window*)
        cx = Ws // 2
        cy = Hs // 2
        x0 = max(0, cx - crop_w // 2)
        x1 = min(Ws, x0 + crop_w)
        y0 = max(0, cy - crop_h // 2)
        y1 = min(Hs, y0 + crop_h)

        # re-shift if we hit edges
        x0 = max(0, x1 - crop_w)
        y0 = max(0, y1 - crop_h)

        psf_smooth_np = psf_smooth_np[y0:y1, x0:x1]

        # update extent to the requested physical crop
        extent = [
            -center_crop_extent / 2,
            +center_crop_extent / 2,
            -center_crop_extent / 2,
            +center_crop_extent / 2,
        ]

    # ---- log scale (safe) ----
    norm = LogNorm() if log_enabled else None
    if log_enabled:
        # Replace nonpositive with smallest positive value (if any); else leave as-is
        if np.any(psf_smooth_np <= 0):
            pos = psf_smooth_np[psf_smooth_np > 0]
            if pos.size > 0:
                min_pos = float(pos.min())
                psf_smooth_np = np.where(psf_smooth_np <= 0, min_pos, psf_smooth_np)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(psf_smooth_np, origin="lower", extent=extent, norm=norm, cmap='jet')
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    #ax.set_title("FFT PSF")

    ### GMO: Set ticks similar to Zemax OpticStudio
    xticks = [extent[0], 0.0, extent[1]]
    yticks = [extent[2], 0.0, extent[3]]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))

    def _fmt(v, _pos=None):
        if abs(v) < 1e-12: v = 0.0
        return f"{v:.3g}"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Relative Intensity", rotation=270, labelpad=15)

    plt.show()
    return fig, ax


def create_pupil(N: int, d: float):
    """
    Generate a circular pupil mask on an NxN Cartesian grid.

    [inputs]
    N : int - Number of grid samples per dimension (NxN grid).
    d : float - Physical diameter of the computational window.

    [output]
    p : (N, N) np.ndarray - Circular pupil mask (1 inside unit radius, 0 outside).
    x : (N, N) np.ndarray - x-coordinate grid.
    rsq : (N, N) np.ndarray - Squared radial coordinate grid (x^2 + y^2).
    """
    r = d / 2.0                         # compute half-width of computational window
    x_line = np.linspace(-r, r, N + 1)  # generate 1D coordinate vector from -r to +r (N+1 points)
    x_line = x_line[:N]                 # keep first N samples to avoid "+r + del_x" later
    x = np.tile(x_line, (N, 1))    # create 2D x-coordinate grid by repeating row vector N times
    xsq = x * x                         # compute x^2 at each grid point
    rsq = xsq + xsq.T                   # compute r^2 = x^2 + y^2 using transpose to represent y-grid
    p = rsq <= 1.0                      # create circular pupil mask: 1 inside unit circle, 0 outside

    # return mask as float (0.0 / 1.0), plus coordinate grid and r^2
    return p.astype(np.float64), x, rsq
