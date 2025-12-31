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