import cv2
import numpy as np
import open3d as o3d

from depth2metric.inference.camera import fallback_intrinsics, intrinsics_from_exif
from depth2metric.inference.geometry import depth_to_pcd, scale_depth_map
from depth2metric.inference.models import get_detections, get_final_depth
from depth2metric.inference.utils import get_image_colors


def depth_pcd(image_file, midas_and_transforms, yolo):
    image_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR_RGB)

    width, height, _ = image.shape
    try:
        K = intrinsics_from_exif(image_file, width, height)
    except RuntimeError:
        K = fallback_intrinsics(width, height)

    detections = get_detections(yolo, image)

    midas, transforms = midas_and_transforms
    depth_map = get_final_depth(midas, transforms, image)

    depth_map = scale_depth_map(depth_map, detections, K)
    pcd = depth_to_pcd(depth_map, K)

    colors = get_image_colors(image)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd = pcd.voxel_down_sample(voxel_size=0.75)

    return pcd


def pack_pointcloud(points, colors):
    N = points.shape[0]

    structured = np.zeros(N, dtype=[
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("r", np.uint8),
        ("g", np.uint8),
        ("b", np.uint8),
    ])

    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    structured["r"] = colors[:, 0]
    structured["g"] = colors[:, 1]
    structured["b"] = colors[:, 2]

    return structured.tobytes()
