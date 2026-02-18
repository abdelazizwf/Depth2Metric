from depth2metric.inference.models import get_midas, get_yolo
from depth2metric.pipeline import precompute_samples


def main():
    midas, transforms = get_midas()
    yolo = get_yolo()
    precompute_samples(midas, transforms, yolo)


if __name__ == "__main__":
    main()
