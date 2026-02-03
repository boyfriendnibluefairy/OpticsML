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


def display_field_sub_images(
    box_marks: np.ndarray,
    field0: np.ndarray,
    field7: np.ndarray,
    field10: np.ndarray,
    border_size: int = 3,
    border_color: Optional[Sequence[Union[str, Sequence[int]]]] = ("blue", "green", "yellow"),
    window_name: str = "Field sub-images",
    wait: bool = True,
    margin_inch: float = 1.0,          # <-- 1 inch margin from screen edges (default)
    allow_upscale: bool = False,       # <-- set True if you want to enlarge composite to fill space
) -> np.ndarray:
    """
    Display a single composite window:

        [ box_marks | field10 ]
        [          | field7  ]
        [          | field0  ]

    - Left column: box_marks (kept as-is for display, except type/channel conversions).
    - Right column: a vertical stack of (field10, field7, field0).
      These three are resized so their *combined* height matches the displayed height of box_marks.
    - Borders (optional) are drawn around the three right-column images:
        field10 = yellow, field7 = green, field0 = blue (based on sample border_color).
      If border_color is None -> no borders.

    [output]
        composite_bgr: The composed BGR uint8 image shown in the window.
                      (Useful if you want to save it with cv2.imwrite.)
    """
    if border_size < 0:
        raise ValueError("border_size must be >= 0")
    if margin_inch < 0:
        raise ValueError("margin_inch must be >= 0")

    # ----------------------------
    # internal functions
    # ----------------------------
    def _to_uint8(im: np.ndarray) -> np.ndarray:
        if im.dtype == np.uint8:
            return im
        imf = im.astype(np.float32)
        mn, mx = float(np.nanmin(imf)), float(np.nanmax(imf))
        if mx > mn:
            imf = (imf - mn) / (mx - mn)
        imf = np.clip(imf, 0.0, 1.0)
        return (imf * 255.0 + 0.5).astype(np.uint8)

    def _to_bgr(im: np.ndarray) -> np.ndarray:
        im = _to_uint8(im)
        if im.ndim == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 1:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        im3 = im[:, :, :3]

        # imageio => RGB, but OpenCV expects BGR
        return im3[:, :, ::-1].copy()  # RGB -> BGR

    def _color_to_bgr(c: Union[str, Sequence[int]]) -> Tuple[int, int, int]:
        if isinstance(c, str):
            name = c.strip().lower()
            table = {
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "red": (0, 0, 255),
                "cyan": (255, 255, 0),
                "magenta": (255, 0, 255),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "orange": (0, 165, 255),
                "purple": (128, 0, 128),
            }
            if name not in table:
                raise ValueError(f"Unknown color name '{c}'. Use a BGR tuple or a known name.")
            return table[name]
        if isinstance(c, (list, tuple)) and len(c) == 3:
            b, g, r = c
            return (int(b), int(g), int(r))
        raise TypeError("Color must be a string or a length-3 tuple/list.")

    def _add_border(im_bgr: np.ndarray, color_bgr: Tuple[int, int, int], t: int) -> np.ndarray:
        if t <= 0:
            return im_bgr
        return cv2.copyMakeBorder(im_bgr, t, t, t, t, borderType=cv2.BORDER_CONSTANT, value=color_bgr)

    def _resize_keep_aspect_to_width(im_bgr: np.ndarray, target_w: int) -> np.ndarray:
        h, w = im_bgr.shape[:2]
        if w == target_w:
            return im_bgr
        if w <= 0:
            raise ValueError("Invalid image width")
        scale = target_w / float(w)
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(im_bgr, (target_w, new_h), interpolation=cv2.INTER_AREA)

    def _resize_to_hw(im_bgr: np.ndarray, target_h: int, target_w: int, downscale_only: bool = False) -> np.ndarray:
        h, w = im_bgr.shape[:2]
        if downscale_only and (target_w >= w and target_h >= h):
            return im_bgr
        interp = cv2.INTER_AREA if (target_w < w or target_h < h) else cv2.INTER_LINEAR
        return cv2.resize(im_bgr, (target_w, target_h), interpolation=interp)

    def _get_screen_w_h_and_ppi() -> Tuple[int, int, float]:
        """
        Uses tkinter to get screen size and pixels-per-inch (so we can enforce 1 inch margins).
        """
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = int(root.winfo_screenwidth())
        screen_h = int(root.winfo_screenheight())
        # pixels per inch (more robust than guessing 96)
        ppi = float(root.winfo_fpixels("1i"))
        root.destroy()
        return screen_w, screen_h, ppi

    # ----------------------------
    # normalize inputs for display
    # ----------------------------
    left = _to_bgr(box_marks)
    H_left, W_left = left.shape[:2]

    f0 = _to_bgr(field0)
    f7 = _to_bgr(field7)
    f10 = _to_bgr(field10)

    # Borders for right-column tiles
    if border_color is None:
        f10_b, f7_b, f0_b = f10, f7, f0
    else:
        if len(border_color) != 3:
            raise ValueError("border_color must be a sequence of 3 colors (field0, field7, field10) or None.")

        c0 = _color_to_bgr(border_color[0])   # field0 (blue in sample)
        c7 = _color_to_bgr(border_color[1])   # field7 (green in sample)
        c10 = _color_to_bgr(border_color[2])  # field10 (yellow in sample)

        f10_b = _add_border(f10, c10, border_size)
        f7_b  = _add_border(f7,  c7,  border_size)
        f0_b  = _add_border(f0,  c0,  border_size)

    # ----------------------------
    # Keep the right column internally consistent (as you had it)
    # ----------------------------
    widths = [f10_b.shape[1], f7_b.shape[1], f0_b.shape[1]]
    target_w0 = int(max(widths))

    f10_w = _resize_keep_aspect_to_width(f10_b, target_w0)
    f7_w  = _resize_keep_aspect_to_width(f7_b,  target_w0)
    f0_w  = _resize_keep_aspect_to_width(f0_b,  target_w0)

    stack_h0 = f10_w.shape[0] + f7_w.shape[0] + f0_w.shape[0]
    if stack_h0 <= 0:
        raise ValueError("Invalid stack height from field images.")

    scale = H_left / float(stack_h0)
    target_w = max(1, int(round(target_w0 * scale)))

    h10 = max(1, int(round(f10_w.shape[0] * scale)))
    h7  = max(1, int(round(f7_w.shape[0]  * scale)))
    h0  = max(1, H_left - (h10 + h7))

    f10_final = _resize_to_hw(f10_w, h10, target_w)
    f7_final  = _resize_to_hw(f7_w,  h7,  target_w)
    f0_final  = _resize_to_hw(f0_w,  h0,  target_w)

    right = np.vstack([f10_final, f7_final, f0_final])

    if right.shape[0] != H_left:
        right = _resize_to_hw(right, H_left, right.shape[1])

    composite = np.hstack([left, right])

    # ----------------------------
    # NEW: fit composite into a centered window with >= 1 inch margins
    # ----------------------------
    screen_w, screen_h, ppi = _get_screen_w_h_and_ppi()
    margin_px = int(round(ppi * margin_inch))

    avail_w = max(1, screen_w - 2 * margin_px)
    avail_h = max(1, screen_h - 2 * margin_px)

    comp_h, comp_w = composite.shape[:2]

    # Scale composite to fit inside available area
    fit_scale = min(avail_w / float(comp_w), avail_h / float(comp_h))
    if not allow_upscale:
        fit_scale = min(1.0, fit_scale)

    disp_w = max(1, int(round(comp_w * fit_scale)))
    disp_h = max(1, int(round(comp_h * fit_scale)))

    composite_disp = _resize_to_hw(composite, disp_h, disp_w)

    # Center window on screen (and still maintain margins)
    x = max(margin_px, (screen_w - disp_w) // 2)
    y = max(margin_px, (screen_h - disp_h) // 2)

    # Create a resizable window, set its size and position, then show
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.moveWindow(window_name, x, y)

    cv2.imshow(window_name, composite_disp)

    if wait:
        cv2.waitKey(0)
        #cv2.destroyWindow(window_name) # GMO
        cv2.destroyAllWindows()

    return composite_disp


def display_psf_and_pupil_func(
    psf,
    pupil,
    psf_cmap="viridis",
    pupil_view="intensity",          # "intensity" or "phase"
    pupil_cmap_intensity="gray",
    pupil_cmap_phase="twilight",
):
    """
    Display PSF (left) and padded pupil (right) in one window.

    Left panel (PSF):
      - Shows PSF with a colorbar
      - Axes ticks labeled ONLY at:
            -((N/2) - 0.5), 0, +((N/2) + 0.5)
        where N is the PSF array size (after squeeze).

    Right panel (Pupil):
      - pupil_view="intensity" -> displays |pupil|^2
      - pupil_view="phase"     -> displays angle(pupil) in [-pi, pi]
      - Includes a colorbar

    [inputs]
    psf : np.ndarray or torch.Tensor
        2D PSF intensity array.
    pupil : np.ndarray or torch.Tensor
        2D complex pupil array (often zero-padded pupil).
    psf_cmap : str
        Matplotlib colormap for PSF.
    pupil_view : str
        "intensity" or "phase".
    pupil_cmap_intensity : str
        Colormap for pupil intensity display.
    pupil_cmap_phase : str
        Colormap for pupil phase display.
    """

    # ----------------------------------------
    # internal function: torch -> numpy safely
    # ----------------------------------------
    def _to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    psf_np = np.squeeze(_to_numpy(psf))
    pupil_np = np.squeeze(_to_numpy(pupil))

    if psf_np.ndim != 2:
        raise ValueError(f"psf must be 2D after squeeze; got shape {psf_np.shape}")
    if pupil_np.ndim != 2:
        raise ValueError(f"pupil must be 2D after squeeze; got shape {pupil_np.shape}")
    if psf_np.shape[0] != psf_np.shape[1]:
        raise ValueError(f"psf must be square; got shape {psf_np.shape}")

    # ------------------------------------------
    # PSF axis labeling (similar to OpticStudio)
    # ------------------------------------------
    N = int(psf_np.shape[0])
    neg_label = -((N / 2) - 0.5)
    pos_label = +((N / 2) + 0.5)
    extent_psf = [neg_label, pos_label, neg_label, pos_label]

    # -----------------------------
    # Choose pupil display
    # -----------------------------
    pupil_view_l = pupil_view.lower()
    if pupil_view_l == "intensity":
        pupil_img = np.abs(pupil_np) ** 2
        pupil_title = "Padded pupil |U|^2"
        pupil_cmap = pupil_cmap_intensity
        pupil_vmin, pupil_vmax = None, None
        pupil_cbar_label = "Intensity |U|^2"
    elif pupil_view_l == "phase":
        pupil_img = np.angle(pupil_np)
        pupil_title = "Padded pupil phase ∠U (rad)"
        pupil_cmap = pupil_cmap_phase
        pupil_vmin, pupil_vmax = -np.pi, np.pi
        pupil_cbar_label = "Phase (rad)"
    else:
        raise ValueError("pupil_view must be 'intensity' or 'phase'.")

    # -----------------------------
    # Plot: one window, 2 panels
    # -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: PSF
    im0 = ax[0].imshow(
        psf_np,
        origin="lower",
        extent=extent_psf,
        cmap=psf_cmap,
    )
    ax[0].set_title("FFT PSF")
    ax[0].set_xlabel("x (samples)")
    ax[0].set_ylabel("y (samples)")

    ax[0].set_xticks([neg_label, 0.0, pos_label])
    ax[0].set_yticks([neg_label, 0.0, pos_label])
    ax[0].set_xticklabels([f"{neg_label:g}", "0", f"{pos_label:g}"])
    ax[0].set_yticklabels([f"{neg_label:g}", "0", f"{pos_label:g}"])

    cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Intensity (normalized)")

    # Right: Pupil
    im1 = ax[1].imshow(
        pupil_img,
        origin="lower",
        cmap=pupil_cmap,
        vmin=pupil_vmin,
        vmax=pupil_vmax,
    )
    ax[1].set_title(pupil_title)
    ax[1].set_xlabel("x (pixels)")
    ax[1].set_ylabel("y (pixels)")

    cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    cbar1.set_label(pupil_cbar_label)

    plt.show()
