import numpy as np
import open3d as o3d
from ultralytics.engine.results import Results

PRIORS = {
    0: ["person", 170],
    56: ["chair", 80],
    57: ["couch", 80],
    59: ["bed", 100],
    60: ["dining_table", 75],
    39: ["bottle", 25],
    63: ["laptop", 30],
    2: ["car", 120],
    7: ["truck", 300],
    68: ["microwave", 30],
    69: ["oven", 80],
    72: ["refrigerator", 200],
}


def pixel_to_3d(u: int, v: int, depth: int, K: dict[str, float]) -> np.ndarray:
    """Turn a 2D pixel to 3D camera coordinates."""
    X = (u - K["cx"]) * depth / K["fx"]
    Y = (v - K["cy"]) * depth / K["fy"]
    Z = depth
    return np.array([X, Y, Z])


def get_pcd_points(depth_map: np.ndarray, K: dict[str, float] | None = None) -> np.ndarray:
    """Turn a depth map to point cloud points."""
    h, w = depth_map.shape

    points = np.zeros((h, w, 3))

    # For quick access to pixel coords
    indices = np.arange(h * w).reshape(h, w)
    V = indices // w
    U = indices % w

    # Use camera intrinsics, if provided, to transform pixels to coordinates
    if K is not None:
        points[:, :, 0] = (U - K["cx"]) * (depth_map / K["fx"])
        points[:, :, 1] = (V - K["cy"]) * (depth_map / K["fy"]) * -1
    else:
        points[:, :, 0] = U
        points[:, :, 1] = V * -1

    points[:, :, 2] = depth_map

    return points.reshape(-1, points.shape[2])


def distance_between_pixels(
    p1: tuple[int, int],
    p2: tuple[int, int],
    depth_map: np.ndarray,
    K: dict[str, float],
) -> float:
    """Calculate the distance between 2 pixels in 3D space."""
    u1, v1 = p1
    u2, v2 = p2

    P1 = pixel_to_3d(u1, v1, depth_map[v1, u1], K)
    P2 = pixel_to_3d(u2, v2, depth_map[v2, u2], K)

    return float(np.linalg.norm(P1 - P2))


def get_scale_from_detections(
    depth_map: np.ndarray,
    detections: Results,
    K: dict[str, float],
    conf_threshold: float = 0.5,
) -> float | None:
    """Use detected scene priors to calculate scale."""
    scales = []
    for box, cls, conf in zip(
        detections.boxes.xyxy,
        detections.boxes.cls,
        detections.boxes.conf
    ):
        cls = int(cls.item())

        if cls in PRIORS and conf >= conf_threshold:
            # Calculate the mid-top and mid-bottom points
            x_min, y_min, x_max, y_max = box
            x_c = int((x_min + x_max) / 2)
            y_min, y_max = int(y_min), int(y_max)

            P_bottom = pixel_to_3d(x_c, y_min, depth_map[y_min, x_c], K)
            P_top = pixel_to_3d(x_c, y_max, depth_map[y_max, x_c], K)

            h_rel = np.abs(P_top[1] - P_bottom[1]) # Distance of the vertical only
            # h_rel = np.linalg.norm(P_top - P_bottom)
            s = PRIORS[cls][1] / h_rel
            scales.append(s)

    if len(scales) == 0:
        return None

    return np.median(scales)


def get_scale_from_image_bottom(
    pcd_points: np.ndarray,
    bottom_factor: float = 0.05,
    camera_height: float = 160.0
) -> float:
    """Use the bottom of the image (assumed ground) and assumed camera height to calculate scale."""
    h, _ = pcd_points.shape
    points_slice = int(-bottom_factor * h)
    points = pcd_points[points_slice:]

    distances = np.linalg.norm(points, axis=1)
    median = np.median(distances)
    median_point_idx = np.abs(median - distances).argmin()
    median_point = points[median_point_idx]

    camera = np.zeros(3)
    hyp = np.linalg.norm(camera - median_point)

    camera[1] = median_point[1]
    hor = np.linalg.norm(camera - median_point)

    ver = ((hyp ** 2) - (hor ** 2)) ** 0.5
    return float(camera_height / ver)


def get_scale_from_ground_plane(
    pcd: o3d.geometry.PointCloud,
    camera_height: float = 160.0
) -> float | None:
    """Use a segmented vertical plane (assumed ground) and assumed camera height to calculate scale."""
    [a, b, c, d], _ = pcd.segment_plane(
        distance_threshold=0.5,
        ransac_n=10,
        num_iterations=500,
    )

    plane = np.array([a, b, c])
    normal = np.linalg.norm(plane)

    units = plane / normal
    if abs(units[1]) < 0.75:
        return None

    height_to_origin = abs(d) / normal

    return camera_height / height_to_origin
