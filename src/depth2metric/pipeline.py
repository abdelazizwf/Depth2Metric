from collections.abc import Callable
from typing import BinaryIO

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO  # type: ignore

from depth2metric.inference.camera import fallback_intrinsics, intrinsics_from_exif
from depth2metric.inference.geometry import (
    get_pcd_points,
    get_scale_from_detections,
    get_scale_from_image_bottom,
)
from depth2metric.inference.models import get_depth_map, get_detections
from depth2metric.inference.utils import get_image_colors


def points_to_pcd(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def depth_pcd(
    image_file: BinaryIO,
    midas: Callable,
    midas_transforms: Callable,
    yolo: YOLO
) -> o3d.geometry.PointCloud:
    """Read image, extract intrinsics, calculate scale, and return downsampled point cloud."""
    image_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR_RGB)

    if image is None:
        raise RuntimeError()

    width, height, _ = image.shape
    K = intrinsics_from_exif(image_file, width, height)
    if K is None:
        K = fallback_intrinsics(width, height)

    depth_map = get_depth_map(midas, midas_transforms, image)

    pcd_points = get_pcd_points(depth_map, K)

    scale = None
    detections = get_detections(yolo, image)
    if detections is not None:
        scale = get_scale_from_detections(depth_map, detections, K)

    if scale is None:
        # scale = get_scale_from_image_bottom(pcd_points)
        scale = 0.1 # Fallback for now

    depth_map *= scale
    pcd_points = get_pcd_points(depth_map, K)

    colors = get_image_colors(image)
    pcd = points_to_pcd(pcd_points, colors)

    pcd = pcd.voxel_down_sample(voxel_size=0.7)

    return pcd


def pack_pointcloud(pcd: o3d.geometry.PointCloud) -> bytes:
    """Pack point cloud into bytes."""
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)

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
