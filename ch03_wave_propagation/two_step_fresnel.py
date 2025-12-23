import numpy as np
from ch04_utilities.image_processing import *

def two_step_propagate(U1: np.ndarray, wavelength: float, z: float, L1: float, scale_factor: float) -> tuple[np.ndarray, float]:
    """
    Two-step or Scaled Fresnel propagation.

    [inputs]
        U1 : (N,N) complex
            Input complex field.
        wavelength : float
            Wavelength (meters).
        z : float
            Propagation distance (meters).
        L1 : float
            Input plane physical side length (meters).
        scale : float
            Output scaling factor, L2 = scale * L1.

    [output]
        U2 : (N,N) complex128
            Output complex field on a scaled window.
        L2 : float
            Output plane physical side length (meters).
    """
    # Basic constants
    k = 2.0 * np.pi / wavelength
    N = U1.shape[0]
    if U1.shape[0] != U1.shape[1]:
        raise ValueError("U1 must be square (N x N) for this simplified script.")

    # Define input and output window sizes
    L2 = scale_factor * L1

    # Build input coordinates (x1,y1) and output coordinates (x2,y2)
    _, xx1, yy1, dx1 = make_xy_grid(N, L1)
    _, xx2, yy2, dx2 = make_xy_grid(N, L2)

    # Quadratic phase factors ("chirps") in source and destination planes
    # These appear in Fresnel diffraction as exp(i*k/(2z) * r^2).
    Q1 = np.exp(1j * k / (2.0 * z) * (xx1**2 + yy1**2))
    Q2 = np.exp(1j * k / (2.0 * z) * (xx2**2 + yy2**2))

    # Multiply the input by the source chirp
    U_pre = U1 * Q1

    # Compute Fourier transform with centered convention:
    #   ifftshift -> FFT -> fftshift
    # Multiply by dx^2 to approximate integral scaling.
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_pre))) * (dx1 * dx1)

    # Fresnel prefactor:
    #   exp(i*k*z) / (i*lambda*z)
    # Multiply by destination chirp.
    U2 = (np.exp(1j * k * z) / (1j * wavelength * z)) * Q2 * F

    return U2.astype(np.complex128), L2