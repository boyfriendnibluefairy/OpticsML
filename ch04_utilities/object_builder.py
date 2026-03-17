import numpy as np

def create_pinhole(N=512, pixel_pitch_m=6.5e-6, aperture_radius_mm=1.0):
    """
    Create a hard circular aperture (pinhole) as a grayscale image.

    White pixels (255) represent the transparent circular aperture.
    Black pixels (0) represent the opaque region.

    [inputs]
        N : int, optional
            Image size in pixels. Output shape is (N, N).
        pixel_pitch_m : float, optional
            Pixel pitch in meters.
        aperture_radius_mm : float, optional
            Radius of the circular aperture in millimeters.

    [output]
        pinhole : np.ndarray
            2D uint8 array of shape (N, N), similar to imageio.imread() output
            for a grayscale image. Values are:
                255 inside the aperture
                  0 outside the aperture
    """
    # Convert aperture radius from mm to m
    aperture_radius_m = aperture_radius_mm * 1e-3

    # Create centered coordinate grid in meters
    coords = (np.arange(N) - (N - 1) / 2) * pixel_pitch_m
    xx, yy = np.meshgrid(coords, coords)

    # Radial distance from image center
    rr = np.sqrt(xx**2 + yy**2)

    # Hard circular aperture: white inside, black outside
    pinhole = np.where(rr <= aperture_radius_m, 255, 0).astype(np.uint8)

    return pinhole