import numpy as np


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


def apply_lens(U: np.ndarray, wavelength: float, L: float, f: float, pupil_radius: float | None = None) -> np.ndarray:
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