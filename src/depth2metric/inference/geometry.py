import numpy as np
import open3d as o3d


def default_intrinsics(width, height):
    fx = fy = width
    cx = width / 2
    cy = height / 2
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }


def depth_to_pcd(depth_map, intrinsic_func=default_intrinsics):
    h, w = depth_map.shape
    K = intrinsic_func(w, h)

    points = np.zeros((h, w, 3))
    indices = np.arange(h * w).reshape(h, w, 3)

    V = indices // w
    U = indices % w

    points[:, :, 0] = (U - K["cx"]) * (depth_map / K["fx"])
    points[:, :, 1] = (V - K["cy"]) * (depth_map / K["fy"]) * -1
    points[:, :, 2] = depth_map

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd
