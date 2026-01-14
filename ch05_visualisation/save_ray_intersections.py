import matplotlib.pyplot as plt
import torch
import os


def save_ray_intersection_plot(ray_fan, plane_z, filename='ray_fan_intersection.png'):
    """
    Save the intersection of a ray fan with a plane at z = plane_z to the 'output' folder.

    [inputs]
        ray_fan - list of Ray objects
        plane_z : float - z-coordinate of the plane to intersect with the ray fan
        filename : str - name of the output file (default: 'ray_fan_intersection.png')
    """
    intersection_points = []

    for ray in ray_fan:
        x2, y2, z2, X2, Y2, Z2, _, _, _ = torch.chunk(ray, 9, dim=0)
        t = (plane_z - z2) / Z2
        intersection_x = x2 + t * X2
        intersection_y = y2 + t * Y2
        intersection_points.append((intersection_x, intersection_y))

    intersection_points = np.array(intersection_points)

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=1, marker='s')
    formatted_z = fmt_underscore_3(plane_z)
    #plt.title(f'Ray Fan Intersection at z = {formatted_z}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)

    # Set axis limits
    plt.xlim([-10.0, 10.0])
    plt.ylim([-10.0, 10.0])

    # Ensure output directory exists
    output_dir = os.path.join(os.getcwd(), 'output_spot_diagram')
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")
