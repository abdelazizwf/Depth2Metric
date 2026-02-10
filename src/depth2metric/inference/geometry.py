import numpy as np
import open3d as o3d


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
