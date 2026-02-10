import numpy as np
import open3d as o3d

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


def pixel_to_3d(u, v, depth, K):
    """Turn a 2D pixel to 3D camera coordinates."""
    X = (u - K["cx"]) * depth / K["fx"]
    Y = (v - K["cy"]) * depth / K["fy"]
    Z = depth
    return np.array([X, Y, Z])


def depth_to_pcd(depth_map, K=None):
    """Generate a point cloud from a depth map."""
    h, w = depth_map.shape

    points = np.zeros((h, w, 3))
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, points.shape[2]))

    return pcd


def distance_between_pixels(p1, p2, depth_map, K):
    """Calculate the distance between 2 pixels."""
    u1, v1 = p1
    u2, v2 = p2

    P1 = pixel_to_3d(u1, v1, depth_map[v1, u1], K)
    P2 = pixel_to_3d(u2, v2, depth_map[v2, u2], K)

    return np.linalg.norm(P1 - P2)


def scale_depth_map(depth_map, detections, K):
    scales = []
    for box, cls in zip(detections.boxes.xyxy, detections.boxes.cls):
        cls = cls.item()

        if cls in PRIORS:
            x_min, y_min, x_max, y_max = box
            x_c = int((x_min + x_max) / 2)
            y_min, y_max = int(y_min), int(y_max)

            P_bottom = pixel_to_3d(x_c, y_min, depth_map[y_min, x_c], K)
            P_top = pixel_to_3d(x_c, y_max, depth_map[y_max, x_c], K)

            h_rel = np.abs(P_top[1] - P_bottom[1])
            # h_rel = np.linalg.norm(P_top - P_bottom)
            s = PRIORS[cls][1] / h_rel
            scales.append(s)

    s = np.median(scales)
    depth_map *= s
    return depth_map
