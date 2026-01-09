import numpy as np
import torch

def convert_numpy_to_tensor(U):
    """
    Convert a NumPy ndarray into a PyTorch tensor.

    [inputs]
        U : np.ndarray
            Input NumPy array. Can be real or complex, any dimensionality.

    [output]
        tensor : torch.Tensor
            PyTorch tensor with matching dtype and shape.
            - Real arrays -> torch.float64 / torch.float32 (depending on NumPy dtype)
            - Complex arrays -> torch.complex128 / torch.complex64

    Notes:
        The returned tensor is always on CPU. Move explicitly if needed:
        tensor = convert_numpy_to_tensor(U).to("cuda")
        If gradients are needed later:
        tensor = convert_numpy_to_tensor(U).requires_grad_(True)
    """
    if not isinstance(U, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")

    # torch.from_numpy preserves dtype and shares memory (when possible)
    tensor = torch.from_numpy(U)

    return tensor


def convert_tensor_to_numpy(U):
    """
    Convert a PyTorch tensor into a NumPy ndarray.

    [inputs]
        U : torch.Tensor
            Input PyTorch tensor. Can be real or complex, any dimensionality,
            and may reside on CPU or GPU.

    [output]
        array : np.ndarray
            NumPy ndarray with matching shape and dtype.
            - Real tensors -> np.float32 / np.float64
            - Complex tensors -> np.complex64 / np.complex128
    """
    if not isinstance(U, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    # Detach from autograd, move to CPU if necessary, then convert
    # .detach() prevents unwanted gradient tracking.
    # .cpu() ensures compatibility with NumPy.
    array = U.detach().cpu().numpy()

    return array

