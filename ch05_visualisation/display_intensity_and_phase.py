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