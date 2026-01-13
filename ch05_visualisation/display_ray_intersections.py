import numpy as np
import torch
import matplotlib.pyplot as plt


def display_ray_intersection_plot(ray_fan, plane_z):
    """
    Display the intersection of a ray fan with a plane at z = plane_z.

    [inputs]
        ray_fan : list of torch.Tensor
            Rays with format [x, y, z, X, Y, Z, wvln, opl, total_opl]
        plane_z : float
            z-coordinate of the plane to intersect with the ray fan
    """
    intersection_points = []

    for ray in ray_fan:
        x2, y2, z2, X2, Y2, Z2, _, _, _ = torch.chunk(ray, 9, dim=0)

        t = (plane_z - z2) / Z2
        intersection_x = x2 + t * X2
        intersection_y = y2 + t * Y2

        intersection_points.append(
            (intersection_x.item(), intersection_y.item())
        )

    intersection_points = np.array(intersection_points)

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.scatter(
        intersection_points[:, 0],
        intersection_points[:, 1],
        s=1,
        marker='s'
    )

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)

    # Axis limits (kept identical to original)
    plt.xlim([-10.0, 10.0])
    plt.ylim([-10.0, 10.0])

    plt.show()
