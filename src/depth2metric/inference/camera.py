from typing import BinaryIO

import exifread


def intrinsics_from_exif(
    img_file: BinaryIO,
    width: int,
    height: int
) -> dict[str, float] | None:
    """Calculate camera intrinsics from available EXIF metadata."""
    tags = exifread.process_file(img_file, builtin_types=True) # type: ignore

    f35 = tags.get("EXIF FocalLengthIn35mmFilm")
    fl = tags.get("EXIF FocalLength")

    if f35 is not None and fl is not None:
        # Get the sensor diagonal and width to height ratio to calculate sensor width
        crop_factor = f35 / fl
        wh_r = width / height
        diagonal_mm = 43.27 / crop_factor # 43.27 is the diagonal of frame 36x24
        wh_diag_r = ((1 ** 2) + (wh_r ** 2)) ** 0.5
        w_mm = diagonal_mm * (wh_r / wh_diag_r)
        fx = fl * (width / w_mm)
    elif f35 is not None:
        fx = (f35 / 36) * width
    elif fl is not None:
        w_mm = 6.5 # Fallback common value
        fx = (fl / w_mm) * width
    else:
        return None

    return {
        "cx": width / 2, "cy": height / 2,
        "fx": fx, "fy": fx,
    }


def fallback_intrinsics(width: int, height: int) -> dict[str, float]:
    """Basic intrinsics for pinhole camera."""
    return {
        "fx": width, "fy": width,
        "cx": width / 2, "cy": height / 2
    }
