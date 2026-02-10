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


def generate_object_point_rays_v2(
        object_point_mm: Tuple[float, float, float],
        system_params: "SimpleNamespace"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (skew_rays, chief_ray) for ONE object point.
    Rays start at (x0,y0,z0) and are aimed to intersect the entrance pupil plane z=0 at pupil sample points.

    Notes for debugging:
        -> Output format is compatible with surf3Draytrace_v2 => [x,y,z,X,Y,Z,wvln,opl1,total_opl].
        -> positions in mm
        -> directions are unitless (normalized)
        -> wvln stored in mm
    """
    x0, y0, z0 = object_point_mm

    ### extract necessary variables from system_params
    wvln_mm = float(system_params.wvln) * 1e3           # wavelength: meters -> mm (633e-9 m = 0.000633 mm)
    pattern = system_params.pattern
    nx, ny = system_params.N, system_params.N
    n_rays = nx*ny
    EPD_mm = system_params.EPD_mm
    opl1 = system_params.opl1
    total_opl1 = system_params.total_opl1
    device = system_params.device
    side_length = system_params.L0

    ### generate entrance pupil points based on desired sampling pattern
    unit_pts = pupil_points_unit_disk(pattern=pattern, n_rays=n_rays, nx=nx, ny=ny)
    pupil_xy = unit_pts * (EPD_mm / 2.0)  # mm
    px = pupil_xy[:, 0]
    py = pupil_xy[:, 1]
    pz = np.zeros_like(px)  # for now, we assume that entrance pupil plane is at z=0

    ### build direction vectors from object point to pupil point
    dx = px - x0
    dy = py - y0
    dz = pz - z0
    norm = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
    X = dx / norm
    Y = dy / norm
    Z = dz / norm

    ### generate SKEW RAYS from all pupil samples
    Np = px.shape[0]
    skew = np.zeros((Np, 9), dtype=np.float32)
    skew[:, 0] = float(x0)
    skew[:, 1] = float(y0)
    skew[:, 2] = float(z0)
    skew[:, 3] = X.astype(np.float32)
    skew[:, 4] = Y.astype(np.float32)
    skew[:, 5] = Z.astype(np.float32)
    skew[:, 6] = float(wvln_mm)
    skew[:, 7] = float(opl1)
    skew[:, 8] = float(total_opl1)

    ### generate CHIEF RAY that is aimed at pupil center (0,0,0)
    dx_c = 0.0 - x0
    dy_c = 0.0 - y0
    dz_c = 0.0 - z0
    norm_c = (dx_c * dx_c + dy_c * dy_c + dz_c * dz_c) ** 0.5 + 1e-12
    Xc, Yc, Zc = dx_c / norm_c, dy_c / norm_c, dz_c / norm_c
    chief = np.array(
        [[x0, y0, z0, Xc, Yc, Zc, wvln_mm, float(opl1), float(total_opl1)]],
        dtype=np.float32,
    )

    skew_t = torch.tensor(skew, dtype=torch.float32, device=device)
    chief_t = torch.tensor(chief, dtype=torch.float32, device=device)
    return skew_t, chief_t


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


def generate_tilted_plane_wave(
    grid_size: int,
    side_length_mm: float,
    fieldx: float,
    fieldy: float,
    hFOV: float,
    wavelength_mm: float = 0.00055,   # default: 550 nm in mm
    output_dtype=np.complex128,
) -> np.ndarray:
    """
    Generate a unit-amplitude tilted plane wave on a square grid.

    The complex field is:
        U(x,y) = exp( i * k * (sin(theta_x)*x + sin(theta_y)*y) )
    where theta_x = fieldx*hFOV and theta_y = fieldy*hFOV (in degrees).

    [inputs]
        grid_size : int
            Number of samples per side (grid_size x grid_size).
        side_length_mm : float
            Physical side length of the square window [mm].
        fieldx, fieldy : float
            Normalized field coordinates (e.g., 0.0 to 1.0). The actual tilt angles are:
                theta_x = fieldx * hFOV
                theta_y = fieldy * hFOV
        hFOV : float
            Half field-of-view [degrees]. (So theta_x/theta_y are also in degrees.)
        wavelength_mm : float
            Wavelength [mm]. Default 0.00055 mm = 550 nm.
        output_dtype :
            Complex dtype for output.

    [output]
        U : np.ndarray (complex)
            Complex plane-wave field of shape (grid_size, grid_size) with |U|=1 everywhere.

    Assumptions:
        - +z axis is the optic axis.
        - Positive theta_x tilts the wave so phase increases with +x; likewise for y.
        - Small-angle: sin(theta) ~ theta(rad), but we use exact sin for robustness.
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be > 0.")
    if side_length_mm <= 0:
        raise ValueError("side_length_mm must be > 0.")
    if wavelength_mm <= 0:
        raise ValueError("wavelength_mm must be > 0.")

    # Physical sampling
    N = int(grid_size)
    L = float(side_length_mm)  # [mm]
    dx = L / N                 # [mm]

    # Centered coordinate grid in mm
    x = (np.arange(N) - (N // 2)) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")

    # Tilt angles in degrees -> radians
    theta_x = np.deg2rad(fieldx * hFOV)
    theta_y = np.deg2rad(fieldy * hFOV)

    # Wavenumber in rad/mm
    k = 2.0 * np.pi / float(wavelength_mm)

    # Plane-wave transverse spatial frequencies (k_x, k_y)
    # For a plane wave tilted by theta about x (in x-z plane), kx = k*sin(theta_x).
    kx = k * np.sin(theta_x)
    ky = k * np.sin(theta_y)

    # Unit-amplitude complex plane wave
    U = np.exp(1j * (kx * xx + ky * yy)).astype(output_dtype)
    return U
