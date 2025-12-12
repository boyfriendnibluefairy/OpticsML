import numpy as np
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve

def calculate_RMSE(ground_truth, reconstructed):
    """
    Compute the Root-Mean-Square Error (RMSE) between a ground truth image
    and a reconstructed image. Both images are normalized to [0,1] before
    the RMSE calculation. If reconstructed has different dimensions,
    it is resized to match the ground truth.

    [inputs]
        ground_truth : ndarray (H x W)
        reconstructed : ndarray (any size)

    [output]
        rmse : float
    """

    # Convert inputs to float64
    G = np.asarray(ground_truth, dtype=np.float64)
    R = np.asarray(reconstructed, dtype=np.float64)

    # Resize reconstructed image if sizes do not match
    if G.shape != R.shape:
        R = resize(R, G.shape, order=1, preserve_range=True, anti_aliasing=True)

    # -----------------------------------
    # Internal function for normalization
    # -----------------------------------
    def _normalize(img):
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)

        # Avoid division by zero if image is constant
        if img_max - img_min < 1e-12:
            return np.zeros_like(img)

        return (img - img_min) / (img_max - img_min)

    # Normalize both images to [0,1]
    G_norm = _normalize(G)
    R_norm = _normalize(R)

    # Compute RMSE
    diff = G_norm - R_norm
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    return rmse


def calculate_PSNR(ground_truth, reconstructed):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between a ground truth image
    and a reconstructed image. If the reconstructed image has different
    dimensions, it will be resized to match the ground truth.

    [inputs]
        ground_truth : ndarray
            Reference image (H x W) or (H x W x C).
        reconstructed : ndarray
            Reconstructed image to compare (may differ in size).

    [output]
        psnr : float
            Peak Signal-to-Noise Ratio in dB.
    """

    # Convert inputs to float64
    G = np.asarray(ground_truth, dtype=np.float64)
    R = np.asarray(reconstructed, dtype=np.float64)

    # Resize reconstructed image if needed
    if G.shape != R.shape:
        R = resize(
            R,
            G.shape,
            order=1,              # bilinear interpolation
            preserve_range=True,  # keep original value range
            anti_aliasing=True
        )

    # Infer L_max from the ground truth
    # If ground truth appears normalized, L_max ~ 1; else use its max
    max_val = G.max()
    if max_val == 0:
        # Degenerate case: all-zero ground truth, define L_max as 1 to avoid division by zero
        L_max = 1.0
    else:
        L_max = max_val

    # Compute MSE
    diff = G - R
    mse = np.mean(diff ** 2)

    if mse == 0:
        # Identical images
        return np.inf

    # Compute PSNR in dB
    psnr = 10.0 * np.log10((L_max ** 2) / mse)

    return psnr


def calculate_SSIM(ground_truth, reconstructed):
    """
    Compute the Structural Similarity Index Measure (SSIM) between a ground truth
    image and a reconstructed image. If the reconstructed image has different
    dimensions, it will be resized to match the ground truth.

    [inputs]
        ground_truth : ndarray
            Reference image (H x W) or (H x W x C).
        reconstructed : ndarray
            Reconstructed image to compare (may differ in size).

    [output]
        ssim_value : float
            SSIM value in [0, 1] (typically).
    """

    # Convert to float64 arrays
    G = np.asarray(ground_truth, dtype=np.float64)
    R = np.asarray(reconstructed, dtype=np.float64)

    # Resize reconstructed image if shape differs
    if G.shape != R.shape:
        R = resize(
            R,
            G.shape,
            order=1,              # bilinear interpolation
            preserve_range=True,  # keep the original value scale
            anti_aliasing=True
        )

    # If images are not normalized, normalize to [0,1]
    G_min, G_max = G.min(), G.max()
    if G_max > G_min:
        G = (G - G_min) / (G_max - G_min)
    R_min, R_max = R.min(), R.max()
    if R_max > R_min:
        R = (R - R_min) / (R_max - R_min)

    # Decide whether images are multichannel
    if G.ndim == 3 and G.shape[2] in (3, 4):
        multichannel = True
    else:
        multichannel = False

    # Compute SSIM
    ssim_value = ssim(
        G,
        R,
        data_range=G.max() - G.min() if G.max() != G.min() else 1.0,
        channel_axis=-1 if multichannel else None
    )

    return ssim_value


def calculate_GMSD(ground_truth, reconstructed, T=0.0026):
    """
    Compute the Gradient Magnitude Similarity Deviation (GMSD) between
    a ground truth image and a reconstructed image.

    If the reconstructed image has different dimensions, it will be
    resized to match the ground truth. Both images are normalized to
    [0, 1] before computing GMSD.

    [inputs]
        ground_truth : ndarray
            Reference image (H x W), preferably real-valued.
        reconstructed : ndarray
            Reconstructed image to compare (may differ in size).
        T : float, optional
            Small positive constant for stability in the GMS formula.
            Default is 0.0026 for normalized [0,1] images.

    [output]
        gmsd : float
            Gradient Magnitude Similarity Deviation (lower is better).
    """

    # Convert to float64 arrays
    G = np.asarray(ground_truth, dtype=np.float64)
    R = np.asarray(reconstructed, dtype=np.float64)

    # If reconstructed shape differs, resize to ground truth shape
    if G.shape != R.shape:
        R = resize(
            R,
            G.shape,
            order=1,              # bilinear interpolation
            preserve_range=True,  # keep original value scale
            anti_aliasing=True
        )

    # Normalize both images to [0, 1]
    def normalize(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            # Avoid division by zero if image is constant
            return np.zeros_like(img)

    G_norm = normalize(G)
    R_norm = normalize(R)

    # Define Sobel kernels for gradient computation
    # (You could use Prewitt; Sobel is standard and works well.)
    sobel_x = np.array([[ -1,  0,  1],
                        [ -2,  0,  2],
                        [ -1,  0,  1]], dtype=np.float64)
    sobel_y = np.array([[ -1, -2, -1],
                        [  0,  0,  0],
                        [  1,  2,  1]], dtype=np.float64)

    # Compute gradients for ground truth
    Gx = convolve(G_norm, sobel_x, mode='reflect')
    Gy = convolve(G_norm, sobel_y, mode='reflect')
    # Gradient magnitude
    mG = np.sqrt(Gx**2 + Gy**2)

    # Compute gradients for reconstructed
    Rx = convolve(R_norm, sobel_x, mode='reflect')
    Ry = convolve(R_norm, sobel_y, mode='reflect')
    mR = np.sqrt(Rx**2 + Ry**2)

    # Gradient Magnitude Similarity (GMS) map
    numerator = 2 * mG * mR + T
    denominator = mG**2 + mR**2 + T
    gms_map = numerator / denominator

    # GMSD: standard deviation of GMS map
    gmsd = np.std(gms_map)

    return gmsd

