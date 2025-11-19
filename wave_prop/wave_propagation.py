import numpy as np

def Fresnel(Uin, dx, dy, wave, z, method='kernel'):
    """Performs Fresnel diffraction propagation.

    [inputs]
        Uin is the optical field on the source plane.
        dx, dy is the sampling interval in x, y axis.
        z is the distance between source and observation planes.
        method is the method used to perform the Fourier transform.
            'kernel' indicates a convolution kernel is used.
            'freq' indicates the transfer function is used.

    [output]
        Uout is the complex field at the observation plane.
    """

    M, N = Uin.shape
    Lx, Ly = dx * N, dy * M
    k = 2 * np.pi / wave

    if method.lower() == 'freq':
        # use the transfer function in Fourier domain
        fx = np.arange(-1/(2*dx), 1/(2*dx), 1/Lx)
        fy = np.arange(-1/(2*dy), 1/(2*dy), 1/Ly)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(-1j * np.pi * wave * z * (FX*FX + FY*FY))
        H = np.fft.fftshift(H)
    elif method.lower() == 'kernel':
        # use the impusle response kernel
        x = np.arange(-N*dx/2, N*dx/2, dx)
        y = np.arange(-M*dy/2, M*dy/2, dy)
        X, Y = np.meshgrid(x, y)
        h = 1 / (1j*wave*z) * np.exp(1j*k/(2*z) * (X*X + Y*Y))
        H = np.fft.fft2(np.fft.fftshift(h)) * dx * dy
    else:
        raise ValueError('Unkown simulation method...')

    Uout = np.fft.ifftshift(np.fft.ifft2(H * np.fft.fft2(np.fft.fftshift(Uin))))
    return Uout

def Fraunhofer(Uin, dx, dy, wave, z):
    """Performs Fraunhofer diffraction propagation.

    [inputs]
        Uin is the optical field on the source plane.
        dx, dy is the sampling interval in x, y axis.
        z is the distance between source and observation planes.

    [output]
        - Uout is the field on the observation plane
        - corresponding coordinates.
    """

    M, N = Uin.shape
    k = 2 * np.pi / wave

    # lengths on the source plane
    Lx, Ly = dx * N, dy * M

    # lengths and samping intervals on the observation plane
    Lx2, Ly2 = wave * z / dx, wave * z / dy
    dx2, dy2 = wave * z / Ly, wave * z / Lx

    x2 = np.arange(-Lx2/2, Lx2/2, dx2)
    y2 = np.arange(-Ly2/2, Ly2/2, dy2)
    X2, Y2 = np.meshgrid(x2, y2)

    c = 1. / (1j*wave*z) * np.exp(1j*k/(2*np.pi) * (X2*X2+Y2*Y2))  # scale factor
    Uout = c * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(Uin))) * dx * dy

    return Uout, x2, y2

def RayleighSommerfeld(Uin, dx, dy, wave, z, method='kernel'):
    """Performs Rayleigh-Sommerfeld diffraction propagation.

    [inputs]
        Uin is the optical field on the source plane.
        dx is the sampling interval in x axis.
        dy is the sampling interval in y axis.
        z is the distance between source and observation planes.
        method is the method used to perform the Fourier transform.
            'kernel' indicates a convolution kernel is used.
            'freq' indicates the transfer function is used.
            The default is the kernel method.

    [output]
        Uout is the field on the observation plane.
    """

    M, N = Uin.shape
    Lx = dx * N
    Ly = dy * M

    k = 2 * np.pi / wave

    if method.lower() == 'freq':
        # use the transfer function in Fourier domain, = RS_AS method
        fx = np.arange(-1/(2*dx), 1/(2*dx), 1/Lx)
        fy = np.arange(-1/(2*dy), 1/(2*dy), 1/Ly)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(1j * k * z * np.sqrt(1. - wave*wave*FX*FX - wave*wave*FY*FY))
        H = np.fft.fftshift(H)
    elif method.lower() == 'kernel':
        # use the impusle response kernel, = RS_conv
        x = np.arange(-N*dx/2, N*dx/2, dx)
        y = np.arange(-M*dy/2, M*dy/2, dy)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X*X + Y*Y + z*z)
        h = z / (1j*wave) * np.exp(1j*k*r) / (r*r)
        H = np.fft.fft2(np.fft.fftshift(h)) * dx * dy
    else:
        raise ValueError('Unkown simulation method...')


    Uout = np.fft.ifftshift(np.fft.ifft2(H * np.fft.fft2(np.fft.fftshift(Uin))))

    return Uout