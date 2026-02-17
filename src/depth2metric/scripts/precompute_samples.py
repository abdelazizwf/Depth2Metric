import gzip
import re
from pathlib import Path

from depth2metric.inference.models import get_midas, get_yolo
from depth2metric.pipeline import depth_pcd, pack_pointcloud

SAMPLES_DIR = Path("static/samples")
PRECOMP_DIR = Path("static/precomputed")


def main():
    midas, transforms = get_midas()
    yolo = get_yolo()

    for image_file in SAMPLES_DIR.glob("*"):
        if re.search(r".+\.(jpg|jpeg|png)", image_file.name) is None:
            continue

        with open(image_file, "br") as f:
            pcd = depth_pcd(f, midas, transforms, yolo)
            buffer = gzip.compress(pack_pointcloud(pcd))
        with open(PRECOMP_DIR / (image_file.stem + ".bytes"), "bw") as f:
            f.write(buffer)
        print(f"Computed PCD points buffer for {str(image_file)!r} successfully.")


if __name__ == "__main__":
    main()
