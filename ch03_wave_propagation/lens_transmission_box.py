import numpy as np
import torch
from typing import Tuple, Optional

### this function is called by generate_object_point_rays_v2()
def pupil_points_unit_disk(pattern: str, n_rays: int, nx: int, ny: int) -> np.ndarray:
    """
    Generates pupil sample points on a *unit* disk, shape (N,2), where x,y in [-1,1] and x^2+y^2<=1.
        for cartesian pattern: this applies (nx,ny) grid then masks to unit disk
        for fibonacci/random pattern: this applies n_rays samples
    """
    pattern_l = pattern.lower()

    if pattern_l == "cartesian":
        x = np.linspace(-1.0, 1.0, int(nx))
        y = np.linspace(-1.0, 1.0, int(ny))
        xx, yy = np.meshgrid(x, y, indexing="xy")
        mask = (xx * xx + yy * yy) <= 1.0
        pts = np.stack([xx[mask], yy[mask]], axis=1)
        return pts

    if n_rays <= 0:
        raise ValueError("n_rays must be > 0 for fibonacci/random patterns.")

    if pattern_l == "fibonacci":
        k = np.arange(n_rays, dtype=np.float64)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        r = np.sqrt((k + 0.5) / n_rays)
        theta = k * golden_angle
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        return np.stack([px, py], axis=1)

    if pattern_l == "random":
        r = np.sqrt(np.random.rand(n_rays))
        theta = 2.0 * np.pi * np.random.rand(n_rays)
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        return np.stack([px, py], axis=1)

    raise ValueError("pattern must be 'cartesian', 'fibonacci', or 'random'.")


### called by ray_based_lens_transmission_v3()
def bin_sum(contrib: torch.Tensor, N:int, flat_idx:torch.Tensor) -> torch.Tensor:
    """
    Coherently adds ray contributions onto the pupil grid. Physically,
    this converts a discrete ray representation into a sampled wavefront
    on the exit pupil.

    This function performs the core operation:
        U[pixel] = Σ (complex phasors of rays landing in that pixel)

    [inputs]
        contrib : torch.Tensor, shape (Nrays,)
            Complex contribution of each ray (amplitude × exp(i·phase)).
        N : int = grid size
        flat_idx : list of int = indices of flattended phase distribution

    [output]
        U : torch.Tensor, shape: (N, N) or (grid_size, grid_size)
            Complex pupil-plane field after coherent summation.
    """

    # Allocate a flattened complex pupil grid. Length = grid_size × grid_size
    # Each element corresponds to one pupil pixel.
    #
    # complex128 is used because phase-accurate wavefront reconstruction
    # is sensitive to numerical precision.
    U = torch.zeros(N * N, dtype=torch.complex128, device=contrib.device)

    # Scatter-add ray contributions into the flattened grid.
    #
    # For each ray k:
    #   U[flat_idx[k]] += contrib[k]
    #
    # If multiple rays map to the same pupil pixel, scatter_add_
    # correctly accumulates (sums) their complex phasors.
    #
    # This is the key operation that preserves phase coherence.
    U.scatter_add_(0, flat_idx, contrib)

    # Reshape the flattened grid back into a 2D pupil array.
    return U.view(N, N)