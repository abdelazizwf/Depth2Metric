import gzip
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO  # type: ignore

from depth2metric.common.settings import get_settings
from depth2metric.common.utils import get_logger
from depth2metric.inference.camera import fallback_intrinsics, intrinsics_from_exif
from depth2metric.inference.geometry import (
    get_pcd_points,
    get_scale_from_detections,
    get_scale_from_ground_plane,
    get_scale_from_image_bottom,
)
from depth2metric.inference.models import get_depth_map, get_detections
from depth2metric.inference.utils import get_image_colors

settings = get_settings()
logger = get_logger(__name__)

SAMPLES_DIR = Path(settings.samples_dir)
PRECOMP_DIR = Path(settings.precomputed_dir)


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
) -> tuple[o3d.geometry.PointCloud, float, str]:
    """Read image, extract intrinsics, calculate scale, and return downsampled point cloud."""
    image_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR_RGB)

    if image is None:
        logger.error("CV2 couldn't load image.")
        raise RuntimeError()

    width, height, _ = image.shape
    K = intrinsics_from_exif(image_file, width, height)
    if K is None:
        logger.info("No relevant EXIF metadata found.")
        K = fallback_intrinsics(width, height)

    depth_map = get_depth_map(midas, midas_transforms, image)

    pcd_points = get_pcd_points(depth_map, K)

    scale_factor, method = None, ""
    detections = get_detections(yolo, image)
    if detections is not None:
        scale_factor = get_scale_from_detections(depth_map, detections, K)
        method = "scene priors"

    if scale_factor is None:
        scale_factor = get_scale_from_ground_plane(points_to_pcd(pcd_points))
        method = "ground plane detection"

    if scale_factor is None:
        scale_factor = get_scale_from_image_bottom(pcd_points)
        method = "bottom image as ground"

    logger.info(f"Scale factor is {scale_factor:.5f} set using {method!r}.")

    depth_map *= scale_factor
    pcd_points = get_pcd_points(depth_map, K)

    colors = get_image_colors(image)
    pcd = points_to_pcd(pcd_points, colors)

    pcd = pcd.voxel_down_sample(voxel_size=settings.voxel_size)

    return pcd, scale_factor, method


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


def precompute_samples(midas: Callable, transforms: Callable, yolo: YOLO) -> dict[str, str]:
    if PRECOMP_DIR.exists():
        shutil.rmtree(PRECOMP_DIR)

    metadata = {}
    os.makedirs(PRECOMP_DIR)
    for image_file in SAMPLES_DIR.glob("*"):
        if re.search(r".+\.(jpg|jpeg|png)", image_file.name) is None:
            continue

        with open(image_file, "br") as f:
            pcd, scale_priors, method = depth_pcd(f, midas, transforms, yolo)
            buffer = gzip.compress(pack_pointcloud(pcd))
            metadata[image_file.name] = {
                "X-Scaling-Factor": str(scale_priors),
                "X-Scaling-Method": method,
            }

        with open(PRECOMP_DIR / (image_file.stem + ".bytes"), "bw") as f:
            f.write(buffer)

        logger.info(f"Computed PCD points buffer for {str(image_file)!r} successfully.")

    return metadata
