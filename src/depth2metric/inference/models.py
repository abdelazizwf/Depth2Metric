# type: ignore

import os
from collections.abc import Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER as YOLO_LOGGER

from depth2metric.common.settings import get_settings
from depth2metric.common.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()

YOLO_LOGGER.setLevel(40) # Suppress YOLO output


def get_midas(model_name: str | None = None) -> tuple[Callable, Callable]:
    """Load MiDaS model and related transforms."""
    if model_name is None:
        model_name = settings.midas_model

    midas = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_name in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    logger.info(f"Loaded MiDaS ({model_name}) successfully.")
    return midas, transform


def get_yolo(model_name: str | None = None) -> YOLO:
    if model_name is None:
        model_name = settings.yolo_model

    model_file = os.path.join(settings.models_dir, model_name + ".pt")
    yolo = YOLO(model_file)
    logger.info(f"Loaded {model_name} from {model_file!r} successfully.")
    return yolo


def get_detections(model: YOLO, image: np.ndarray) -> Results | None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image)
    if len(results) == 0:
        logger.debug("YOLO made no detections.")
        return None
    return results[0]


def get_depth(model: Callable, original_image: np.ndarray, tr_image: np.ndarray) -> np.ndarray:
    """Get the model output and scale it using interpolation."""
    with torch.no_grad():
        prediction = model(tr_image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=original_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output


def get_depth_map(model: Callable, transforms: Callable, image: np.ndarray) -> np.ndarray:
    """Get the final depth map possibly combining multiple outputs."""
    tr_image = transforms(image)
    orig_output = get_depth(model, image, tr_image)
    return orig_output
