"""
main_02_image_hologram.py demonstrates recording and reconstruction of an image hologram.
Image hologram is a type of hologram where the object field passes through a lens and the
hologram is recorded one focal length away from the lens.
Complex wave field is propagated using Angular Spectrum.
"""

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from wave_prop.wave_propagation import RayleighSommerfeld


# ---------------------------------------------------------------------
# 1. Input data, set parameters
# ---------------------------------------------------------------------
# USAF256.bmp is a 256x256 8-bit grayscale object pattern.
# convert to grayscale if it is RGB.
img = imageio.imread("ray_tracing/test_image/USAF256.bmp")

if img.ndim == 3:
    # convert RGB to grayscale (luminance) if needed
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

# Display the object pattern
plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Object pattern")
plt.axis('off')

# Convert to double (float64) – needed to attach complex phase
Ii = img.astype(np.float64)

# Add random phase to the object (random roughness)
# PH is uniform in [0,1), so phase = 2π PH is uniform in [0, 2π).
PH = np.random.rand(256, 256)
Ii = Ii * np.exp(1j * 2 * np.pi * PH)

# ---------------------------------------------------------------------
# 2. Zero padding: embed 256x256 object into 512x512 simulation grid
# ---------------------------------------------------------------------
M = 512
I = np.zeros((M, M), dtype=np.complex128)

# Center the 256x256 object inside the 512x512 array
start = (M - 256) // 2   # = 128
end = start + 256        # = 384, but end index in Python slicing is exclusive
I[start:end, start:end] = Ii

# ---------------------------------------------------------------------
# 3. Physical parameters (converted to SI units)
# ---------------------------------------------------------------------
# z = 15 cm -> 0.15 m   (you can change to 0.01, 0.05, 0.15)
z = 0.15  # meters

# w = 6500 * 10^-8 cm -> 650 nm -> 650e-9 m
wavelength = 650e-9  # meters

# delta = 0.005 cm -> 50 µm -> 50e-6 m
dx = 50e-6  # meters
dy = 50e-6  # meters (square pixels)

# Meshgrid of indices r,c similar to MATLAB's r=1:M, c=1:M, [C,R] = meshgrid(c,r).
# We will reuse R for the "phase mismatch" factor later.
r = np.arange(M)   # 0..511 (others use 1..512, the offset is a global phase only)
c = np.arange(M)
C, R = np.meshgrid(c, r)

# ---------------------------------------------------------------------
# 4. Forward propagation (650 nm): object plane -> hologram plane
#    via Angular Spectrum or RS_AS
# ---------------------------------------------------------------------
# I is the complex object field with random phase, padded to 512x512.
# E is the complex field at the hologram (image) plane at distance z.

E = RayleighSommerfeld(I, dx, dy, wavelength, z, method='freq')

# ---------------------------------------------------------------------
# 5. Reconstruction (650 nm): monochromatic reconstruction
# ---------------------------------------------------------------------
# To reconstruct, we propagate the hologram field E backward by distance z,
# i.e., with -z."
# block using a propagation kernel with -z.
R1_field = RayleighSommerfeld(E, dx, dy, wavelength, -z, method='freq')

# Intensity is |R1|^2; normalize to [0,1] for display.
R1_intensity = np.abs(R1_field) ** 2
R1_intensity /= R1_intensity.max()

plt.figure()
plt.imshow(R1_intensity, cmap='gray')
plt.title("Reconstructed image (650 nm)")
plt.axis('off')

# ---------------------------------------------------------------------
# 6. White-light reconstruction: sum over wavelengths 450–650 nm
#    We simulate temporal incoherence
# ---------------------------------------------------------------------
# dw is used 41 wavelengths from 650 nm down to 450 nm in 5 nm steps.
dw = 50.0  # step in "50Å" units; corresponds to 5 nm per step
IMA = np.zeros((M, M), dtype=np.float64)  # accumulator for white-light intensity

# Precompute sin(10°) once – off-axis reference angle used in recording
sin_theta = np.sin(np.deg2rad(10.0))

for delta_w in range(41):  # g = 0..40
    # Reconstruction wavelength w2 in meters
    # (6500 - 50*g)*1e-8 cm = (6500 - 50g)*1e-10 m
    wavelength2 = (6500.0 - dw * delta_w) * 1e-10  # 650nm..450nm

    # Phase-mismatch factor due to wavelength shift ( E2=E.*exp(2i*pi*sind(10)*(w-w2)/w/w2.*R*delta); )
    # R*delta is (approx) the y-coordinate, and (w - w2)/(w*w2) controls the shift.
    # Here R is 0..M-1, and dy=dx is the pixel size.
    y_coords = R * dy  # physical y-coordinate of each row (from top), in meters

    phase_mismatch = np.exp(
        1j * 2 * np.pi * sin_theta * (wavelength - wavelength2) / (wavelength * wavelength2)
        * y_coords
    )

    # Apply the phase mismatch to the hologram field
    E2 = E * phase_mismatch

    # Backward propagation for this wavelength w2 over distance z
    R2_field = RayleighSommerfeld(E2, dx, dy, wavelength2, -z, method='freq')

    # Intensity at this wavelength
    R2_intensity = np.abs(R2_field) ** 2

    # Accumulate into white-light image
    IMA += R2_intensity

# Normalize the summed white-light reconstruction
IMA /= IMA.max()

plt.figure()
plt.imshow(IMA, cmap='gray')
plt.title("Reconstructed image (white light)")
plt.axis('off')

plt.show()
