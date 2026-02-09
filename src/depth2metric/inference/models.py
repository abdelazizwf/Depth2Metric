# type: ignore

import cv2
import torch

from depth2metric.inference.utils import sharpen_image


def get_MiDaS():
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    return midas, transform


def get_depth(model, original_image, tr_image):
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


def get_final_depth(model, transforms, image):
    tr_image = transforms(image)
    orig_output = get_depth(model, image, tr_image)

    # sh_image = sharpen_image(image)
    # sh_output = get_depth(model, sh_image, transforms(sh_image))

    # output = cv2.add(orig_output, sh_output) / 2
    return orig_output
