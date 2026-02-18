# type: ignore

import os
from collections.abc import Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from depth2metric.common.settings import get_settings
from depth2metric.common.utils import get_logger
from depth2metric.inference.utils import sharpen_image

logger = get_logger(__name__)
settings = get_settings()


def get_midas(model_name: str = "DPT_Hybrid") -> tuple[Callable, Callable]:
    """Load MiDaS model and related transforms."""
    midas = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_name == "DPT_Large" or model_name == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    logger.info(f"Loaded MiDaS ({model_name}) successfully.")
    return midas, transform


def get_yolo(model_name: str = "yolo26n") -> YOLO:
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

    # sh_image = sharpen_image(image)
    # sh_output = get_depth(model, sh_image, transforms(sh_image))

    # output = cv2.add(orig_output, sh_output) / 2
    return orig_output
