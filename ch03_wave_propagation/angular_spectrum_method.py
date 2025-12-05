import numpy as np

def angular_spectrum_method(Uin, dx, dy, wvln, z):
    """Performs optical wave propagation using angular spectrum method

    [inputs]
        Uin = complex optical wavefield on the source plane
        dx = sampling interval along x-axis
        dy = sampling interval along y-axis
        z = distance between source and observation planes
    [output]
        Uout = complex optical wavefield on the observation plane
    """
    M, N = Uin.shape
    Lx = dx * N
    Ly = dy * M

    k = 2 * np.pi / wvln

    fx = np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / Lx)
    fy = np.arange(-1 / (2 * dy), 1 / (2 * dy), 1 / Ly)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z * np.sqrt(1. - wvln * wvln * FX * FX - wvln * wvln * FY * FY))
    H = np.fft.fftshift(H)
    Uout = np.fft.ifftshift(np.fft.ifft2(H * np.fft.fft2(np.fft.fftshift(Uin))))

    return Uout