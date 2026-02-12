import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def sharpen_image(image: np.ndarray) -> np.ndarray:
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1],
    ])

    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened


def show_results(
    image: np.ndarray,
    output: np.ndarray,
    d_image: np.ndarray | None = None
) -> None:
    axes_count = 3 if d_image is not None else 2
    _, axes = plt.subplots(1, axes_count, figsize=(10, 4))

    axes.flat[0].imshow(image)
    axes.flat[0].set_title("Original")

    axes.flat[1].imshow(output, cmap="inferno")
    axes.flat[1].set_title("Predicted Depth")

    if d_image is not None:
        axes.flat[2].imshow(~d_image[..., 0], cmap="inferno")
        axes.flat[2].set_title("True Depth")

    plt.tight_layout()
    plt.show()


def get_image_colors(image: np.ndarray) -> np.ndarray:
    return image.reshape(-1, image.shape[2]) / 255.0


def visualize_pcd(
    pcd: o3d.geometry.PointCloud,
    colors: np.ndarray | None = None
) -> None:
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if os.environ["XDG_SESSION_TYPE"] == "wayland":
        os.environ["XDG_SESSION_TYPE"] = "x11"

    o3d.visualization.draw_geometries([pcd]) # type: ignore
