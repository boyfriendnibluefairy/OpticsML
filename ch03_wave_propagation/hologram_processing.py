import numpy as np
import matplotlib.pyplot as plt

def suppress_dc_via_average_intensity(I_holo):
    """
    Suppress the DC term of a hologram using average intensity subtraction

    See Eq. 3.27–3.28 of
    Schnars, Ulf, and Werner PO Jüptner. "Digital recording and numerical reconstruction
    of holograms." Measurement science and technology 13, no. 9 (2002): R85.

    [inputs]
        I_holo : np.ndarray
            Hologram intensity (non-negative real values)

    [output]
        I_dc_norm : np.ndarray
            DC-suppressed and normalized hologram intensity in [0,1]
    """

    # 1) Normalize hologram
    I_norm = I_holo / I_holo.max()

    # 2) Compute average intensity (DC term)
    Im = I_norm.mean()

    # 3) Subtract DC term (Eq. 3.27–3.28)
    I_dc = I_holo - Im

    # 4) Shift so minimum = 0 (to avoid negatives)
    I_dc = I_dc - I_dc.min()

    # 5) Normalize to [0, 1]
    I_dc_norm = I_dc / I_dc.max()

    return I_dc_norm


def suppress_dc_via_high_pass_filter(I_holo, filter_radius, comparison_plot=False):
    """
    Suppress the DC term by blocking low spatial frequencies in the Fourier domain.

    [inputs]
        I_holo : np.ndarray
            2D hologram intensity image.
        filter_radius : float
            Percentage (0–1) of the radius of the opaque mask in the freq domain.
            Example: filter_radius = 0.3 → block central 30% (in radius).
        comparison_plot : bool
            If True, display a comparison figure of (before, after) suppression.

    [output]
        I_hp_norm : np.ndarray
            High-pass filtered hologram intensity normalized to [0, 1].
    """

    # ------------------------------------------
    # FFT of hologram (centered)
    # ------------------------------------------
    F = np.fft.fftshift(np.fft.fft2(I_holo))

    # ------------------------------------------
    # Circular high-pass mask
    # ------------------------------------------
    M, N = I_holo.shape
    y, x = np.ogrid[:M, :N]
    cy, cx = M // 2, N // 2

    max_radius = min(M, N) // 2               # half the image size
    r_cut = filter_radius * max_radius        # radius in pixels

    dist = np.sqrt((y - cy)**2 + (x - cx)**2)

    mask = np.ones((M, N))
    mask[dist <= r_cut] = 0.0                 # opaque disk → removes DC + low freq

    # ------------------------------------------
    # Apply high-pass filter
    # ------------------------------------------
    F_hp = F * mask

    # ------------------------------------------
    # Inverse FFT → filtered hologram
    # ------------------------------------------
    I_hp = np.fft.ifft2(np.fft.ifftshift(F_hp))
    I_hp = np.abs(I_hp)

    # ------------------------------------------
    # Normalize to [0, 1]
    # ------------------------------------------
    I_hp -= I_hp.min()
    I_hp_norm = I_hp / I_hp.max()

    # ------------------------------------------
    # Optional comparison plot
    # ------------------------------------------
    if comparison_plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(I_holo / I_holo.max(), cmap='gray')
        axs[0].set_title("Original Hologram")
        axs[0].axis('off')

        axs[1].imshow(I_hp_norm, cmap='gray')
        axs[1].set_title(f"High-Pass DC Suppressed (radius={filter_radius})")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return I_hp_norm


def suppress_dc_via_band_pass_filter(I_holo, R1, R2, comparison_plot=False):
    """
    Suppress selected spatial-frequency band by applying an opaque 'donut' mask
    in the Fourier domain.

    [inputs]
        I_holo : np.ndarray
            2D hologram intensity image.
        R1 : float
            Inner radius of the opaque donut, as a fraction of half the image size (0–1).
            This defines the radius of the central *hole* (region that still passes).
        R2 : float
            Outer radius of the opaque donut, as a fraction of half the image size (0–1).
            Frequencies with radius R1 <= r <= R2 are blocked.
        comparison_plot : bool
            If True, display a comparison figure before/after filtering.

    [output]
        I_bp_norm : np.ndarray
            Donut-filtered hologram intensity normalized to [0, 1].
    """

    # Safety checks
    if not (0.0 <= R1 <= 1.0 and 0.0 <= R2 <= 1.0):
        raise ValueError("R1 and R2 must be in [0, 1].")
    if R1 >= R2:
        raise ValueError("R1 must be strictly less than R2.")

    # ------------------------------------------
    # FFT of hologram (centered)
    # ------------------------------------------
    F = np.fft.fftshift(np.fft.fft2(I_holo))

    # ------------------------------------------
    #    Build opaque donut mask
    #    mask = 0 for R1 <= r <= R2 (blocked band)
    #    mask = 1 elsewhere (center + outer region pass)
    # ------------------------------------------
    M, N = I_holo.shape
    y, x = np.ogrid[:M, :N]
    cy, cx = M // 2, N // 2

    max_radius = min(M, N) // 2          # half the image size in pixels
    r1_px = R1 * max_radius
    r2_px = R2 * max_radius

    dist = np.sqrt((y - cy)**2 + (x - cx)**2)

    mask = np.ones((M, N))
    # Opaque donut: block frequencies in the ring [r1_px, r2_px]
    mask[(dist >= r1_px) & (dist <= r2_px)] = 0.0

    # ------------------------------------------
    # Apply donut (band-stop) filter in freq domain
    # ------------------------------------------
    F_bp = F * mask

    # ------------------------------------------
    # Inverse FFT → filtered hologram
    # ------------------------------------------
    I_bp = np.fft.ifft2(np.fft.ifftshift(F_bp))
    I_bp = np.abs(I_bp)

    # ------------------------------------------
    # Normalize to [0, 1]
    # ------------------------------------------
    I_bp -= I_bp.min()
    I_bp_norm = I_bp / I_bp.max()

    # ------------------------------------------
    # Optional comparison plot
    # ------------------------------------------
    if comparison_plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(I_holo / I_holo.max(), cmap='gray')
        axs[0].set_title("Original Hologram")
        axs[0].axis('off')

        axs[1].imshow(I_bp_norm, cmap='gray')
        axs[1].set_title(f"Donut Filtered (R1={R1}, R2={R2})")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return I_bp_norm
