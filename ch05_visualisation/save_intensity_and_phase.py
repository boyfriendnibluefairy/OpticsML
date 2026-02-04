import numpy as np
import matplotlib.pyplot as plt
import torch
import os

"""
    Example usage:
    The code below means to save the complex wavefield U2 one step outside ch05_visualisation folder,
    create a temporary folder called "temporary_output_folder", and then save the png image inside that
    new folder. The term "temporary" means you have to manually delete the folder to avoid overwriting
    your old outputs. The term "{z:04.1f}" means I have to suffix the filename with the value of z but
    it must be attached with zeros at the beginning with zeros until the 4 significant figures are reached.
    The ".1f" value means round off decimal values to 1 decimal place.

    save_complex_np(U=U2, filename=f"../temporary_output_folder/holo_{z:04.1f}.png")
"""


def _save_np_core(U_np,
                  filename,
                  intensity=True,
                  phase=False,
                  title="Intensity and Phase",
                  cmap="gray",
                  strip_labels=False):
    """
    Internal function that saves figures from both real and complex NumPy arrays.
    Includes a terminal message showing where the file was saved.
    If the directory where the file to be saved does not exist, it will automatically create such folder.

    [inputs]
        U_np        : 2D complex wave field, real-valued or complex-valued ndarray
        filename    : str, path where image will be saved (directories auto-created)
        intensity   : if True, save intensity distribution
        phase       : if True, save phase distribution
        title       : figure title
        cmap        : matplotlib colormap
        strip_labels: if True, remove axes, ticks, borders, and colorbars.
                      NOTE: If title is not None, the title will still be rendered.
    """

    def _resolve_path(path):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:  # __file__ undefined in interactive environments
            base_dir = os.getcwd()

        if os.path.isabs(path):
            return path

        rel = path.lstrip("/\\")
        return os.path.join(base_dir, rel)

    if not intensity and not phase:
        raise ValueError("At least one of `intensity` or `phase` must be True.")

    if U_np.ndim != 2:
        raise ValueError("U must be a 2D array.")

    # Compute intensity and/or phase
    I = np.abs(U_np) ** 2 if intensity else None

    # Min-Max Normalization (common for visualization)
    if intensity:
        eps = 1e-12
        I_norm = (I - I.min()) / (I.max() - I.min() + eps)
        I = I_norm

    Phi = np.angle(U_np) if phase else None

    # Resolve path and ensure directory exists (do once)
    full_path = _resolve_path(filename)
    out_dir = os.path.dirname(full_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Option A: Normal scientific figure (strip_labels=False)
    # ----------------------------------------------------------------------
    if not strip_labels:
        if intensity and phase:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axI, axP = axes

            imI = axI.imshow(I, cmap=cmap)
            imP = axP.imshow(Phi, cmap=cmap)

            axI.set_title("Intensity")
            axP.set_title("Phase")

            fig.colorbar(imI, ax=axI, fraction=0.046, pad=0.04)
            fig.colorbar(imP, ax=axP, fraction=0.046, pad=0.04)

            if title is not None:
                fig.suptitle(title)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            data = I if intensity else Phi
            label = "Intensity" if intensity else "Phase"
            im = ax.imshow(data, cmap=cmap)

            ax.set_title(label if title is None else title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Image saved successfully at: {full_path}")
        return

    # ----------------------------------------------------------------------
    # Option B: Borderless export (strip_labels=True)
    #   - Still render title if title is explicitly given (title is not None)
    # ----------------------------------------------------------------------
    data = I if intensity else Phi

    # Create a figure and fill it with an axes occupying almost all space.
    # If a title exists, reserve a small top band for it.
    fig = plt.figure()

    has_title = (title is not None) and (str(title).strip() != "")

    if has_title:
        # Reserve some space at the top for the suptitle
        ax = fig.add_axes([0, 0, 1, 0.94])  # leave ~6% height for title
    else:
        ax = fig.add_axes([0, 0, 1, 1])     # full-bleed

    ax.imshow(data, cmap=cmap)

    # Remove everything visual from axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("")  # ensure no axes title

    if has_title:
        fig.suptitle(title)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(full_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(full_path, dpi=300, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    print(f"[INFO] Borderless image saved successfully at: {full_path}")



def save_real_tensor(U,
                     filename,
                     intensity=True,
                     phase=False,
                     title=None,
                     cmap="gray",
                     strip_labels=False):
    """
    Save a real-valued PyTorch tensor U as an image file.

    [inputs]
        U        : 2D real-valued torch.Tensor
        filename : str, required. See _save_np_core() docstring for path handling.
        intensity : Boolean = if True, save intensity distribution of U
        phase     : Boolean = if True, save phase distribution of U
        title     : str = if not None, the function will include the title in the saved figure
        cmap      : str = matplotlib colormap name
    """
    if torch is None:
        raise ImportError("PyTorch is not available, but save_real_tensor was called.")

    if not isinstance(U, torch.Tensor):
        raise TypeError("U must be a torch.Tensor for save_real_tensor().")

    if torch.is_complex(U):
        raise TypeError("save_real_tensor() expects a real-valued tensor.")

    U_np = U.detach().cpu().numpy().astype(np.float64)
    _save_np_core(U_np,
                  filename=filename,
                  intensity=intensity,
                  phase=phase,
                  title=title,
                  cmap=cmap,
                  strip_labels=strip_labels)


def save_complex_tensor(U,
                        filename,
                        intensity=True,
                        phase=False,
                        title=None,
                        cmap="gray",
                        strip_labels=False):
    """
    Save a complex-valued PyTorch tensor U as an image file.

    [inputs]
        U        : 2D complex-valued torch.Tensor
        filename : str, required. See _save_np_core() docstring for path handling.
        intensity : Boolean = if True, save intensity distribution of U
        phase     : Boolean = if True, save phase distribution of U
        title     : str = if not None, the function will include the title in the saved figure
        cmap      : str = matplotlib colormap name
    """
    if torch is None:
        raise ImportError("PyTorch is not available, but save_complex_tensor was called.")

    if not isinstance(U, torch.Tensor):
        raise TypeError("U must be a torch.Tensor for save_complex_tensor().")

    if not torch.is_complex(U):
        raise TypeError("save_complex_tensor() expects a complex-valued tensor.")

    U_np = U.detach().cpu().numpy()
    _save_np_core(U_np,
                  filename=filename,
                  intensity=intensity,
                  phase=phase,
                  title=title,
                  cmap=cmap,
                  strip_labels=strip_labels)


def save_real_np(U,
                 filename,
                 intensity=True,
                 phase=False,
                 title=None,
                 cmap="gray",
                 strip_labels=False):
    """
    Save a real-valued NumPy array U as an image file.

    [inputs]
        U        : 2D real-valued np.ndarray
        filename : str, required. See _save_np_core() docstring for path handling.
        intensity : Boolean = if True, save intensity distribution of U
        phase     : Boolean = if True, save phase distribution of U
        title     : str = if not None, the function will include the title in the saved figure
        cmap      : str = matplotlib colormap name
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a NumPy ndarray for save_real_np().")

    if np.iscomplexobj(U):
        raise TypeError("save_real_np() expects a real-valued array.")

    U_np = U.astype(np.float64)
    _save_np_core(U_np,
                  filename=filename,
                  intensity=intensity,
                  phase=phase,
                  title=title,
                  cmap=cmap,
                  strip_labels=strip_labels)


def save_complex_np(U,
                    filename,
                    intensity=True,
                    phase=False,
                    title=None,
                    cmap="gray",
                    strip_labels=False):
    """
    Save a complex-valued NumPy array U as an image file.

    [inputs]
        U        : 2D complex-valued np.ndarray
        filename : str, required. See _save_np_core() docstring for path handling.
        intensity : Boolean = if True, save intensity distribution of U
        phase     : Boolean = if True, save phase distribution of U
        title     : str = if not None, the function will include the title in the saved figure
        cmap      : str = matplotlib colormap name
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("U must be a NumPy ndarray for save_complex_np().")

    if not np.iscomplexobj(U):
        raise TypeError("save_complex_np() expects a complex-valued array.")

    U_np = U
    _save_np_core(U_np,
                  filename=filename,
                  intensity=intensity,
                  phase=phase,
                  title=title,
                  cmap=cmap,
                  strip_labels=strip_labels)


def save_rgb01(
    rgb01: np.ndarray,
    filename: str = "rgb_image_on_sensor_plane.png",
    title: str = "RGB",
    strip_labels: bool = True,
) -> None:
    """
    Save an RGB image in [0,1] as a PNG file.

    [inputs]
        rgb01 : np.ndarray
            RGB image of shape (H, W, 3), values expected in [0,1].
        filename : str
            Output filename (saved in the directory of the calling script).
        title : str
            Title to use if strip_labels=False.
        strip_labels : bool
            If True, save ONLY the RGB image (no axes, ticks, borders, title).
            If False, include title, axes, ticks, etc.
    """
    import os
    import inspect
    import matplotlib.pyplot as plt
    import numpy as np

    # Clamp to valid display range
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    # ------------------------------------------------------------
    # Determine directory of the *calling* script
    # ------------------------------------------------------------
    try:
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        out_dir = os.path.dirname(os.path.abspath(caller_file))
    except Exception:
        # Fallback (e.g. interactive / notebook)
        out_dir = os.getcwd()

    out_path = os.path.join(out_dir, filename)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    ax.imshow(rgb01)

    if strip_labels:
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
    else:
        ax.set_title(title)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)

    plt.close(fig)

    print(f"[save_rgb01] Saved RGB image to: {out_path}")


def save_field_sub_images(
    box_marks: np.ndarray,
    field0: np.ndarray,
    field7: np.ndarray,
    field10: np.ndarray,
    border_size: int = 3,
    border_color: Optional[Sequence[Union[str, Sequence[int]]]] = ("blue", "green", "yellow"),
    filename: str = "composite_image.png",
    margin_inch: float = 1.0,
    allow_upscale: bool = False,
) -> np.ndarray:
    """
    Builds the same composite_disp as display_field_sub_images(),
    but saves it to disk instead of displaying it.

    The image is saved in the same directory as the *calling script*.

    [output]
    composite_disp : np.ndarray
        The final image written to disk (uint8, BGR).
    """

    if border_size < 0:
        raise ValueError("border_size must be >= 0")
    if margin_inch < 0:
        raise ValueError("margin_inch must be >= 0")

    # --------------------------------------------------
    # Internal functions
    # --------------------------------------------------
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
            name = c.strip().lower()
            if name not in table:
                raise ValueError(f"Unknown color '{c}'")
            return table[name]
        return tuple(int(v) for v in c)

    def _add_border(im: np.ndarray, color, t: int) -> np.ndarray:
        if t <= 0:
            return im
        return cv2.copyMakeBorder(im, t, t, t, t, cv2.BORDER_CONSTANT, value=color)

    def _resize(im, h, w):
        interp = cv2.INTER_AREA if (h < im.shape[0] or w < im.shape[1]) else cv2.INTER_LINEAR
        return cv2.resize(im, (w, h), interpolation=interp)

    def _get_screen_info():
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        ppi = root.winfo_fpixels("1i")
        root.destroy()
        return w, h, ppi

    # --------------------------------------------------------------
    # Build composite image containing original image and sub-images
    # --------------------------------------------------------------
    left = _to_bgr(box_marks)
    H_left = left.shape[0]

    f0 = _to_bgr(field0)
    f7 = _to_bgr(field7)
    f10 = _to_bgr(field10)

    if border_color is not None:
        c0 = _color_to_bgr(border_color[0])
        c7 = _color_to_bgr(border_color[1])
        c10 = _color_to_bgr(border_color[2])
        f0 = _add_border(f0, c0, border_size)
        f7 = _add_border(f7, c7, border_size)
        f10 = _add_border(f10, c10, border_size)

    widths = [f0.shape[1], f7.shape[1], f10.shape[1]]
    target_w = max(widths)

    def _resize_w(im):
        scale = target_w / im.shape[1]
        return _resize(im, int(round(im.shape[0] * scale)), target_w)

    f10 = _resize_w(f10)
    f7 = _resize_w(f7)
    f0 = _resize_w(f0)

    stack_h = f10.shape[0] + f7.shape[0] + f0.shape[0]
    scale = H_left / stack_h

    h10 = int(round(f10.shape[0] * scale))
    h7  = int(round(f7.shape[0] * scale))
    h0  = H_left - h10 - h7

    right = np.vstack([
        _resize(f10, h10, int(round(target_w * scale))),
        _resize(f7,  h7,  int(round(target_w * scale))),
        _resize(f0,  h0,  int(round(target_w * scale))),
    ])

    composite = np.hstack([left, right])

    # --------------------
    # Fit to screen margin
    # --------------------
    screen_w, screen_h, ppi = _get_screen_info()
    margin_px = int(round(ppi * margin_inch))

    avail_w = screen_w - 2 * margin_px
    avail_h = screen_h - 2 * margin_px

    scale_fit = min(avail_w / composite.shape[1], avail_h / composite.shape[0])
    if not allow_upscale:
        scale_fit = min(1.0, scale_fit)

    composite_disp = _resize(
        composite,
        int(round(composite.shape[0] * scale_fit)),
        int(round(composite.shape[1] * scale_fit)),
    )

    # --------------------------------------------------
    # Save next to calling script
    # --------------------------------------------------
    caller_file = inspect.stack()[1].filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    save_path = os.path.join(caller_dir, filename)

    cv2.imwrite(save_path, composite_disp)

    return composite_disp
