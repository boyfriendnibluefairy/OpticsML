import numpy as np
import torch
from typing import Dict, Literal, Optional, Tuple, Union
from ch02_ray_tracing.apertures import *
from ch04_utilities.image_processing import *


def thin_lens_transmittance(xx: np.ndarray, yy: np.ndarray, wavelength: float, f: float) -> np.ndarray:
    """
    Thin-lens phase transmittance:
        T(x,y) = exp(-i*pi/(lambda*f) * (x^2 + y^2))

    [inputs]
        xx, yy : (N,N)
            Spatial coordinates (meters).
        wavelength : float
            Wavelength (meters).
        f : float
            Lens focal length (meters).

    [output]
        T : (N,N) complex128
            Lens complex transmittance.
    """
    return np.exp(-1j * np.pi * (xx**2 + yy**2) / (wavelength * f))


def apply_thin_lens(U: np.ndarray, wavelength: float, L: float, f: float, pupil_radius: float | None = None) -> np.ndarray:
    """
    Apply a thin lens (phase-only) and optional circular pupil to a complex field U.

    [inputs]
        U : (N,N) complex
            Input complex field at the lens plane.
        wavelength : float
            Wavelength (meters).
        L : float
            Physical side length (meters) of current plane.
        f : float
            Focal length (meters).
        pupil_radius : float | None
            If provided, applies a circular aperture of this radius (meters).

    [output]
        U_out : (N,N) complex128
            Field after lens and pupil.
    """
    N = U.shape[0]
    _, xx, yy, _ = make_xy_grid(N, L)

    # Thin-lens phase
    T = thin_lens_transmittance(xx, yy, wavelength, f)
    U_out = U * T

    # Optional pupil stop
    if pupil_radius is not None:
        P = circular_pupil(xx, yy, pupil_radius)
        U_out *= P

    return U_out.astype(np.complex128)


def calculate_sphere_from_image_chief_ray(
    chief_ray: Union[torch.Tensor, np.ndarray],
    z_exit_pupil: float,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute a reference sphere (center + radius) using a traced chief ray.

    Notes:
        The sphere center is chosen at the *chief ray position* (typically at the image plane).
        The sphere radius is the distance from that center to the chief ray intersection point
              on the exit pupil plane z = z_exit_pupil.

        This makes the reference sphere pass through the chief ray point on the pupil plane,
        and it approximates the "ideal" reference wave that would converge to the image point.

    [inputs]
        chief_ray : torch.Tensor or np.ndarray, shape (9,)
            Ray in surf3Draytrace_v2 format:
                [x, y, z, X, Y, Z, wvln, opl, total_opl]
            IMPORTANT: For best meaning, chief_ray should be the chief ray *at the image plane*
            (or at the plane you treat as the wave's center of curvature).

        z_exit_pupil : float
            The z-coordinate of the exit pupil plane.

        eps : float
            Small value to avoid division by zero.

    [output]
        dictionary:
            xc, yc, zc : sphere center coordinates
            R          : sphere radius
    """

    # ---- Convert numpy input to torch if necessary (keeps downstream math consistent).
    if isinstance(chief_ray, np.ndarray):
        chief_ray = torch.from_numpy(chief_ray)

    # ---- Ensure floating precision is high enough for geometric computations.
    chief_ray = chief_ray.to(torch.float64)

    # ---- Sanity-check input length (we expect exactly 9 values for a single ray).
    if chief_ray.ndim != 1 or chief_ray.numel() < 9:
        raise ValueError("chief_ray must be a 1D array/tensor with at least 9 elements.")

    # ---- Unpack chief ray position (x, y, z) at its current plane (ideally image plane).
    x0 = chief_ray[0]  # chief x-position
    y0 = chief_ray[1]  # chief y-position
    z0 = chief_ray[2]  # chief z-position

    # ---- Unpack chief ray direction (X, Y, Z) at its current plane.
    X0 = chief_ray[3]  # chief direction x-component
    Y0 = chief_ray[4]  # chief direction y-component
    Z0 = chief_ray[5]  # chief direction z-component

    # ---- We need to intersect the chief ray with the exit pupil plane at z = z_exit_pupil.
    #      Parametric ray: r(t) = (x0, y0, z0) + t*(X0, Y0, Z0)
    #      Solve for z_exit_pupil = z0 + t*Z0  =>  t = (z_exit_pupil - z0)/Z0
    if torch.abs(Z0) <= eps:
        raise ValueError("Chief ray Z-direction is ~0; cannot intersect with a z-plane.")

    t = (z_exit_pupil - z0) / Z0  # intersection parameter along the chief ray

    # ---- Compute the chief ray intersection point with the exit pupil plane.
    xp = x0 + t * X0  # pupil-plane x of chief
    yp = y0 + t * Y0  # pupil-plane y of chief
    zp = torch.tensor(float(z_exit_pupil), dtype=torch.float64)  # pupil-plane z (constant)

    # ---- Define the sphere center at the chief ray position (x0,y0,z0).
    #      This corresponds to a wave converging (or diverging) from that point.
    xc = x0
    yc = y0
    zc = z0

    # ---- Sphere radius is distance from sphere center to the pupil-plane chief point.
    #      This ensures the sphere passes through the pupil-plane chief point.
    R = torch.sqrt((xp - xc) ** 2 + (yp - yc) ** 2 + (zp - zc) ** 2)

    # ---- If R is extremely small, the geometry is degenerate (avoid division by zero later).
    if R <= eps:
        raise ValueError("Computed sphere radius is ~0; check chief ray plane and z_exit_pupil.")

    # ---- Return as plain Python floats for easy downstream metadata use.
    return {
        "xc": float(xc.item()),
        "yc": float(yc.item()),
        "zc": float(zc.item()),
        "R": float(R.item()),
    }