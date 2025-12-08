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
                  cmap="gray"):
    """
    Internal function that saves figures from both real and complex NumPy arrays.
    Includes a terminal message showing where the file was saved.
    If the directory where the file to be saved does not exist, it will automatically create such folder.

    [inputs]
        U_np     : 2D complex wave field, it can be real-valued or complex-valued ndarray
        filename : str, required.
                   If it's just a file name (e.g., "hologram.png"), the image is saved
                   in the same directory as the calling script.
                   If it contains directories (e.g., "output/after_lens/hologram.png"
                   or "\\output\\after_lens\\hologram.png"), those directories are
                   created (relative to the calling script directory) before saving.
        intensity : Boolean = if True, save intensity distribution of U
        phase     : Boolean = if True, save phase distribution of U
        title     : str = if not None, the saved figure will include the title
        cmap      : str = matplotlib colormap name
    """
    if not intensity and not phase:
        raise ValueError("At least one of `intensity` or `phase` must be True.")

    # Ensure input is 2D
    if U_np.ndim != 2:
        raise ValueError("U must be a 2D array.")

    # placeholders for data
    I = None
    Phi = None

    if intensity:
        I = np.abs(U_np) ** 2

    if phase:
        Phi = np.angle(U_np)

    # Create figure
    if intensity and phase:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axI, axP = axes

        imI = axI.imshow(I, cmap=cmap)
        axI.set_title("Intensity")
        fig.colorbar(imI, ax=axI, fraction=0.046, pad=0.04)

        imP = axP.imshow(Phi, cmap=cmap)
        axP.set_title("Phase")
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

    # Resolve output path relative to calling script directory
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    if os.path.isabs(filename):
        full_path = filename
    else:
        rel_path = filename.lstrip("/\\")
        full_path = os.path.join(base_dir, rel_path)\

    # Ensure directories exist
    out_dir = os.path.dirname(full_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save image
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ✅ Terminal message
    #print(f"[INFO] Image saved successfully at:\n       {full_path}\n")
    print(f"[INFO] Image saved successfully at: {full_path} ")


def save_real_tensor(U,
                     filename,
                     intensity=True,
                     phase=False,
                     title=None,
                     cmap="gray"):
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
                  cmap=cmap)


def save_complex_tensor(U,
                        filename,
                        intensity=True,
                        phase=False,
                        title=None,
                        cmap="gray"):
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
                  cmap=cmap)


def save_real_np(U,
                 filename,
                 intensity=True,
                 phase=False,
                 title=None,
                 cmap="gray"):
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
                  cmap=cmap)


def save_complex_np(U,
                    filename,
                    intensity=True,
                    phase=False,
                    title=None,
                    cmap="gray"):
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
                  cmap=cmap)