import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def _display_np_core(U_np,
                     intensity=True,
                     phase=False,
                     title="Intensity and Phase",
                     filename="hologram.png",
                     cmap="gray"):
    """
    Internal function that plots or saves figures from both real and complex NumPy Arrays.

    [inputs]
        U = 2D complex wave field, it can be real-valued or complex-valued, ndarray or PyTorch tensor
        intensity : Boolean = if True, the function will display intensity distribution of U
        phase : Boolean = if True, the function will display phase distribution of U
        title :  str = if not None, the function will include the title in the display
        filename : str = if not None, the function will also save the display image in the
                         same directory of the calling script. filename is the file name of the saved image.
    """
    if not intensity and not phase:
        raise ValueError("At least one of `intensity` or `phase` must be True.")

    # Ensure 2D
    if U_np.ndim != 2:
        raise ValueError("U must be a 2D array.")

    # Prepare data
    I = None
    Phi = None

    if intensity:
        # Intensity = |U|^2 even for real-valued fields
        I = np.abs(U_np) ** 2

        # Min-Max Normalization (common for visualisation)
        eps = 1e-12
        I_norm = (I - I.min()) / (I.max() - I.min() + eps)
        I = I_norm

        # # Log-normalized intensity (for huge dynamic range like diffraction pattern)
        # I_log = np.log10(I + 1e-12)
        # I_log_norm = (I_log - I_log.min()) / (I_log.max() - I_log.min())
        # I = I_log_norm

    if phase:
        # Phase in [-pi, pi]
        Phi = np.angle(U_np)

    # Create figure
    if intensity and phase:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axI, axP = axes

        imI = axI.imshow(I, cmap=cmap)
        axI.set_title("Intensity")
        axI.set_xlabel("x (pixels)")
        axI.set_ylabel("y (pixels)")
        fig.colorbar(imI, ax=axI, fraction=0.046, pad=0.04)

        imP = axP.imshow(Phi, cmap=cmap)
        axP.set_title("Phase")
        axP.set_xlabel("x (pixels)")
        axP.set_ylabel("y (pixels)")
        fig.colorbar(imP, ax=axP, fraction=0.046, pad=0.04)

        if title is not None:
            fig.suptitle(title)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        if intensity:
            im = ax.imshow(I, cmap=cmap)
            ax.set_title("Intensity" if title is None else title)
        else:  # phase only
            im = ax.imshow(Phi, cmap=cmap)
            ax.set_title("Phase" if title is None else title)

        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save if requested
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def display_real_tensor(U,
                        intensity=True,
                        phase=False,
                        title=None,         # title="hologram"
                        filename=None,      # filename="hologram.png"
                        cmap="gray"):
    """
    Display a real-valued PyTorch tensor U.

    [inputs]
        U = 2D real-valued torch.Tensor
        intensity : Boolean = if True, the function will display intensity distribution of U
        phase : Boolean = if True, the function will display phase distribution of U
        title :  str = if not None, the function will include the title in the display
        filename : str = if not None, the function will also save the display image in the
                         same directory of the calling script. filename is the file name of the saved image.
    """
    if torch is None:
        raise ImportError("PyTorch is not available, but display_real_tensor was called.")

    if not isinstance(U, torch.Tensor):
        raise TypeError("U must be a torch.Tensor for display_real_tensor().")

    if torch.is_complex(U):
        raise TypeError("display_real_tensor() expects a real-valued tensor.")

    U_np = U.detach().cpu().numpy().astype(np.float64)
    _display_np_core(U_np, intensity=intensity, phase=phase,
                     title=title, filename=filename, cmap=cmap)


def display_complex_tensor(U,
                           intensity=True,
                           phase=False,
                           title=None,
                           filename=None,
                           cmap="gray"):
    """
    Display a complex-valued PyTorch tensor U.

    U: 2D complex-valued torch.Tensor
    """
    if torch is None:
        raise ImportError("PyTorch is not available, but display_complex_tensor was called.")

    if not isinstance(U, torch.Tensor):
        raise TypeError("U must be a torch.Tensor for display_complex_tensor().")

    if not torch.is_complex(U):
        raise TypeError("display_complex_tensor() expects a complex-valued tensor.")

    U_np = U.detach().cpu().numpy()
    _display_np_core(U_np, intensity=intensity, phase=phase,
                     title=title, filename=filename, cmap=cmap)


def display_real_np(U,
                    intensity=True,
                    phase=False,
                    title=None,
                    filename=None,
                    cmap="gray"):
    """
    Display a real-valued NumPy array U.

    [inputs]
        U = 2D real-valued np.ndarray
        intensity : Boolean = if True, the function will display intensity distribution of U
        phase : Boolean = if True, the function will display phase distribution of U
        title :  str = if not None, the function will include the title in the display
        filename : str = if not None, the function will also save the display image in the
                         same directory of the calling script. filename is the file name of the saved image.
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a NumPy ndarray for display_real_np().")

    if np.iscomplexobj(U):
        raise TypeError("display_real_np() expects a real-valued array.")

    U_np = U.astype(np.float64)
    _display_np_core(U_np, intensity=intensity, phase=phase,
                     title=title, filename=filename, cmap=cmap)


def display_complex_np(U,
                       intensity=True,
                       phase=False,
                       title=None,
                       filename=None,
                       cmap="gray"):
    """
    Display a complex-valued NumPy array U.

    [inputs]
        U = 2D complex-valued np.ndarray
        intensity : Boolean = if True, the function will display intensity distribution of U
        phase : Boolean = if True, the function will display phase distribution of U
        title :  str = if not None, the function will include the title in the display
        filename : str = if not None, the function will also save the display image in the
                         same directory of the calling script. filename is the file name of the saved image.
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a NumPy ndarray for display_complex_np().")

    if not np.iscomplexobj(U):
        raise TypeError("display_complex_np() expects a complex-valued array.")

    U_np = U
    _display_np_core(U_np, intensity=intensity, phase=phase,
                     title=title, filename=filename, cmap=cmap)


def display_thin_lens(
    f,
    L=6.0e-3,
    aperture_radius=3e-3,
    strip_labels=False,
    N=1024,
    wvln=633e-9,
    intensity=True,
    phase=False,
):
    """
    Display intensity and/or phase of a thin-lens transmission function.

    [inputs]
        f : float
            Thin-lens focal length [meters].
        L : float, default 6.0e-3
            Physical side length of the computational window [meters].
        aperture_radius : float or None, default 3e-3
            Physical aperture radius [meters]. If None, no aperture is applied
            (pure phase over the full grid).
        strip_labels : bool, default False
            If True: remove axis ticks, titles, units, and colorbars.
        N : int, default 1024
            Grid size (NxN samples).
        wvln : float, default 633e-9
            Wavelength [meters].
        intensity : bool, default True
            If True, show intensity |t(x,y)|^2.
        phase : bool, default False
            If True, show phase angle(t(x,y)).

    Behavior
    --------
    - intensity=True,  phase=False  -> show intensity only (default)
    - intensity=False, phase=True   -> show phase only
    - intensity=True,  phase=True   -> show both side by side
    - intensity=False, phase=False  -> raises ValueError
    """
    # -----------------------------
    # Basic validation
    # -----------------------------
    if f == 0:
        raise ValueError("f must be non-zero (meters).")
    if L <= 0:
        raise ValueError("L must be > 0 (meters).")
    if N is None or int(N) < 2:
        raise ValueError("N must be an integer >= 2.")
    N = int(N)

    if wvln is None or float(wvln) <= 0:
        raise ValueError("wvln must be > 0 (meters).")
    wvln = float(wvln)

    if aperture_radius is not None and float(aperture_radius) <= 0:
        raise ValueError("aperture_radius must be > 0 (meters) or None.")

    if (not intensity) and (not phase):
        raise ValueError("At least one of intensity or phase must be True.")

    # -----------------------------
    # Spatial grid
    # -----------------------------
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    y = np.linspace(-L / 2, L / 2, N, endpoint=False)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    r2 = XX**2 + YY**2

    # -----------------------------
    # Thin-lens phase
    # t(x,y) = A(x,y) * exp( -i*pi/(lambda*f) * (x^2+y^2) )
    # -----------------------------
    phi = -(np.pi / (wvln * f)) * r2

    # -----------------------------
    # Aperture amplitude
    # -----------------------------
    if aperture_radius is None:
        A = np.ones_like(phi, dtype=np.float64)
    else:
        A = (np.sqrt(r2) <= float(aperture_radius)).astype(np.float64)

    # -----------------------------
    # Complex transmission
    # -----------------------------
    t = A * np.exp(1j * phi)

    # Precompute requested outputs
    intensity_map = np.abs(t) ** 2 if intensity else None
    phase_map = np.angle(t) if phase else None

    # Mask phase outside aperture for cleaner visualization (only if aperture exists)
    if phase and (aperture_radius is not None):
        phase_map = phase_map.copy()
        phase_map[A == 0] = np.nan

    extent = (-L / 2, L / 2, -L / 2, L / 2)

    # -----------------------------
    # Plotting layout
    # -----------------------------
    n_panels = int(bool(intensity)) + int(bool(phase))

    if n_panels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        axes = [ax]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes = [ax1, ax2]

    # -----------------------------
    # Render requested panels
    # -----------------------------
    rendered = {}  # store artists for optional colorbars
    panel_idx = 0

    if intensity:
        axI = axes[panel_idx]
        imI = axI.imshow(
            intensity_map,
            extent=extent,
            origin="lower",
            interpolation="nearest",
        )
        axI.set_aspect("equal")
        rendered["intensity"] = (axI, imI)
        panel_idx += 1

    if phase:
        axP = axes[panel_idx]
        imP = axP.imshow(
            phase_map,
            extent=extent,
            origin="lower",
            interpolation="nearest",
        )
        axP.set_aspect("equal")
        rendered["phase"] = (axP, imP)

    # -----------------------------
    # Labels / titles / colorbars
    # -----------------------------
    if not strip_labels:
        if "intensity" in rendered:
            axI, imI = rendered["intensity"]
            axI.set_title("Thin-lens intensity |t(x,y)|²")
            axI.set_xlabel("x [m]")
            axI.set_ylabel("y [m]")
            cbarI = fig.colorbar(imI, ax=axI, fraction=0.046, pad=0.04)
            cbarI.set_label("|t|²")

        if "phase" in rendered:
            axP, imP = rendered["phase"]
            axP.set_title("Thin-lens phase ∠t(x,y)")
            axP.set_xlabel("x [m]")
            axP.set_ylabel("y [m]")
            cbarP = fig.colorbar(imP, ax=axP, fraction=0.046, pad=0.04)
            cbarP.set_label("Phase [rad]")
    else:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()

    # return {
    #     "x": x,
    #     "y": y,
    #     "t": t,
    #     "intensity": intensity_map,
    #     "phase": phase_map if phase else np.angle(t),  # always return a valid phase array
    #     "figure": fig,
    #     "axes": axes,
    # }