"""
example_03_square_aperture_asm.py demonstrates square aperture diffraction using Angular Spectrum Method (RS_AS),
Using create_square_aperture(), we can modify the size and pixel pitch of the object plane. The square aperture size is
adjusted by updating the input square_pixels. If square_pixels = 64, then the aperture size is 64 x 64 pixels.

Based on the physical and numerical parameters and Nyquist frequency, ASM remains valid when the propagation distance is
below 0.0303 m. That's why at very large distances, i.e. 0.30 m, grid-like artifacts becomes visible.
"""
import numpy as np
import matplotlib.pyplot as plt
from ray_tracing.utils import *
from wave_prop.wave_propagation import RayleighSommerfeld
from ray_tracing.visualisation import add_scale_bar


### Define physical and numerical parameters

# Wavelength λ = 0.633 µm (meters)
wavelength = 0.633e-6  # m

# Sampling period (grid spacing) in the aperture plane:
# example : delta = 10 * λ
delta = 10.0 * wavelength  # m per pixel

# Propagation distance z
z = 0.04  # m
#z = 0.3  # m

# Grid size: M x M samples (space size)
M = 512  # same as MATLAB example

# Square aperture side length in pixels (22 in the example)
square_pixels = 33


### Create the square aperture & initial field

# OB(x,y) = object aperture; 1.0 = pixel value inside the square; 0.0 otherwise
OB = create_square_aperture(delta=delta, M=M, square_pixels=square_pixels)

# Treat the aperture as the complex field at z = 0 (with phase = 0)
U0 = OB.astype(np.complex128)


### Propagate the field using Angular Spectrum (RS_AS)

# Use dx = dy = delta (square pixels)
dx = delta
dy = delta

# Propagate from z=0 to z
Uz = RayleighSommerfeld(Uin=U0, dx=dx, dy=dy, wave=wavelength,z=z, method='freq')


### Compute intensity and normalize for display

# Intensity of propagated field: |Uz|^2
intensity = np.abs(Uz) ** 2

# Normalize intensity to [0, 1] for visualization
intensity_norm = intensity / intensity.max()


### Prepare side-by-side plots with scale bars

# Total physical size of the grid (useful for picking a scale bar)
Lx = dx * M  # meters
# Example: choose a scale bar of 0.5 mm
scale_bar_length_m = 0.5e-3  # 0.5 mm
scale_bar_label = "0.5 mm"

# Create a figure with 1x2 subplots (aperture & propagated intensity)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# -------------------------------------------------------
# 5a. Left plot: Aperture
# -------------------------------------------------------
ax0 = axes[0]
im0 = ax0.imshow(OB, cmap='gray', origin='lower')
ax0.set_title("Square aperture")
ax0.axis('off')

# Add scale bar to the aperture image
add_scale_bar(ax0, pixel_size=delta,
              length_m=scale_bar_length_m,
              label=scale_bar_label,
              loc='lower right',
              color='white')

# -------------------------------------------------------
# 5b. Right plot: Propagated intensity |Uz|^2
# -------------------------------------------------------
ax1 = axes[1]
im1 = ax1.imshow(intensity_norm, cmap='gray', origin='lower')
ax1.set_title(f"Diffraction pattern at z = {z:.2f} m")
ax1.axis('off')

# Add the same scale bar to the propagated intensity image
add_scale_bar(ax1, pixel_size=delta,
              length_m=scale_bar_length_m,
              label=scale_bar_label,
              loc='lower right',
              color='white')

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()