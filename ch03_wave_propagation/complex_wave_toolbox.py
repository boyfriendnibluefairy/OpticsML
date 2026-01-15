import numpy as np
import torch.nn.functional as F
import torch

def resample_complex_wave_to_shape(Uc: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """
    Bilinear-resample a complex 2D array to out_shape using torch interpolate.
    This function allows to resample complex pupil function to match a certain wave grid.
    Resamples real and imag parts separately.

    [inputs]
        Uc: (H, W) complex np.ndarray
        out_shape: (H2, W2)
    """
    if Uc.ndim != 2:
        raise ValueError("Uc must be 2D.")
    H2, W2 = out_shape

    Ur = torch.from_numpy(np.real(Uc)).unsqueeze(0).unsqueeze(0).to(torch.float32)  # (1,1,H,W)
    Ui = torch.from_numpy(np.imag(Uc)).unsqueeze(0).unsqueeze(0).to(torch.float32)

    Ur2 = F.interpolate(Ur, size=(H2, W2), mode="bilinear", align_corners=False)
    Ui2 = F.interpolate(Ui, size=(H2, W2), mode="bilinear", align_corners=False)

    Ur2 = Ur2.squeeze().cpu().numpy().astype(np.float64)
    Ui2 = Ui2.squeeze().cpu().numpy().astype(np.float64)

    return (Ur2 + 1j * Ui2).astype(np.complex128)
