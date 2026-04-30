"""
    file : asm_variants_toolbox.py
    description:    This module compiles private and protected functions for
                    angular spectrum methods (asm) and its variants.
"""
import numpy as np

### ======================================================
### PRIVATE OR PROTECTED FUNCTIONS FOR conventional_asm()
### ======================================================

def asm_max_propagation_distance(delta_x, wavelength, M, M0):
    """
    Compute the maximum alias-free propagation distance z_max
    based on Poon & Liu (2014), Eq. (4.27).

    [inputs]
    delta_x : float
        Pixel pitch (meters).

    wavelength : float
        Wavelength (meters).

    M : int
        Total grid size (number of samples).

    M0 : int
        Nonzero object support size (number of samples).

    [output]
    z_max : float
        Maximum propagation distance (meters).
    """

    # Check validity condition
    if wavelength >= 2 * delta_x:
        raise ValueError("Invalid: wavelength must be < 2 * delta_x")

    # Compute square root term
    sqrt_term = np.sqrt(4 * delta_x**2 - wavelength**2)

    # Compute z_max
    z_max = ((M - M0) * delta_x / (2 * wavelength)) * sqrt_term

    return z_max


def is_asm_valid(z, delta_x, wavelength, M, M0):
    """
    Check if a given propagation distance satisfies ASM sampling condition.

    [output]
        True if valid, False otherwise.

    Sample Usage:
    if not is_asm_valid(z, dx, wavelength, M, M0):
        print("⚠️ ASM is at risk of aliasing → switch to Fresnel or scaled ASM")
    """
    z_max = asm_max_propagation_distance(delta_x, wavelength, M, M0)
    return z <= z_max


def required_M_for_z(z, delta_x, wavelength, M0):
    """
    Compute the minimum grid size M required to support
    propagation distance z without aliasing.

    [input]
    z : float
        Desired propagation distance (meters)

    [output]
    M_required : float
        Minimum required grid size
    """

    if wavelength >= 2 * delta_x:
        raise ValueError("Invalid: wavelength must be < 2 * delta_x")

    sqrt_term = np.sqrt(4 * delta_x**2 - wavelength**2)

    M_required = M0 + (2 * wavelength * z) / (delta_x * sqrt_term)

    return M_required


def required_padding(delta_x, wavelength, z):
    """
    Compute required padding (M - M0) based on simplified form.

    Approximation valid when delta_x >> wavelength.
    """
    return (wavelength * z) / (delta_x**2)


def check_tf_sampling(M, delta_x_mm, wvln_mm, z_mm):
    """
    Evaluate Voelz–Roggemann 2009 sampling condition for TF (ASM).
    Prints the results on the terminal

    [inputs]
        M           : padded grid size
        delta_x_mm  : pixel pitch [mm]
        wvln_mm     : wavelength [mm]
        z_mm        : propagation distance [mm]
    """

    # Computational side length (IMPORTANT: use padded grid)
    L_mm = M * delta_x_mm

    # Sampling factor
    s = (wvln_mm * z_mm) / (delta_x_mm * L_mm)

    # Ideal pixel pitch
    delta_x_ideal = np.sqrt((wvln_mm * z_mm) / M)

    print("==== Voelz–Roggemann 2009 Sampling Conditions ====")
    print(f"M (padded size)        = {M}")
    print(f"Δx (pixel pitch)       = {delta_x_mm:.6e} mm")
    print(f"λ (wavelength)         = {wvln_mm:.6e} mm")
    print(f"z (distance)           = {z_mm:.6e} mm")
    print(f"L (side length)        = {L_mm:.6e} mm")
    print(f"Sampling factor s      = {s:.6f}")
    print(f"Ideal Δx               = {delta_x_ideal:.6e} mm")
    print("")

    if np.isclose(s, 1.0, atol=1e-3):
        print("✔ TF is IDEALLY sampled")
        print("   → Best numerical accuracy")
    elif s < 1:
        print("✔ TF is OVERSAMPLED")
        print("   → Safe (no TF aliasing)")
        print("   → Possible bandwidth limitation")
    else:
        print("❌ TF is UNDERSAMPLED")
        print("   → Dangerous: aliasing in transfer function")
        print("   → Expect spiky / nonphysical artifacts")

    print("========================================")
