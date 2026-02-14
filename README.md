<p align="center" style="margin-bottom: 0px !important;">
  <img width="180" src="./static/favicon.png" alt="Depth2Metric logo" align="center">
</p>
<div id="user-content-toc" align="center" style="margin-top: 0px !important;">
  <ul style="list-style: none;">
    <summary>
      <h1><a style="color: white; text-decoration: none;" href="https://depth2metric.abdelazizwf.dev">Depth2Metric</a></h1>
    </summary>
  </ul>
</div>

## Overview

This project converts a single RGB image into a scaled 3D point cloud and allows users to measure real-world distances directly in the browser. It combines monocular depth estimation, geometric camera modeling, and object-based scale estimation to approximate metric measurements without requiring specialized hardware such as LiDAR or stereo cameras.

### Features

- Monocular depth estimation using deep learning
- Camera intrinsics extracted from available EXIF metadata
- 3D point cloud reconstruction via back-projection
- Automatic depth scaling using scene priors (known objects detected in the image)
- Interactive browser-based measurement tool

## Demo

![Demo](resources/demo.webp)

## Installation

> This will download many python dependencies AND any deep learning model used, which are MiDaS DPT-Hybrid and YOLOv26n by default. Expect heavy internet usage.

- Clone this project using `git`

  ```bash
  git clone https://github.com/abdelazizwf/Depth2Metric.git
  ```

- Navigate to the project's directory

  ```bash
  cd Depth2Metric
  ```

- Two methods for local deployment:

  - Using `docker compose`

    ```bash
    docker compose -f compose.local.yaml up --build
    ```

  - Using `uv`

    ```bash
    uv sync --no-dev
    ```

    ```bash
    uv run fastapi dev src/depth2metric/main.py
    ```

- Visit <localhost:8000>

## Technical Details and Limitations

Estimating real-world measurements from a single RGB image is fundamentally challenging because, unlike stereo cameras or LiDAR, a single image does not contain any depth information. Furthermore, images inherently distort geometric scale, which means no measurement with real units can be made. Some other challenges include: unknown camera intrinsics, distortions, noise, occlusions, and more.

This project uses a multi-staged pipeline to address some of those issues and build a scaled 3D reconstruction of the image. These stages are:

- **Monocular depth estimation using a pre-trained neural network** ([MiDaS](https://arxiv.org/abs/1907.01341) by default). This results in relative depth values for each pixel.
- **Building a pinhole camera model using camera intrinsics.** Focal length in pixel space is calculated from extracted EXIF metadata if available. Otherwise, sensible defaults are used.
- **Back-projection of the image into 3D using the camera model.** This ensures the scene is geometrically correct.
- **Metric scale estimation to scale the arbitrary geometry units into metric units.** Several methods are used to estimate the scale factor:
  - **Object-based scene priors**: An object detection model ([YOLOv26n](https://docs.ultralytics.com/models/yolo26/#overview) by default) is used to detect objects with known real-world sizes in the image. If a detection is made, the ratio between the reconstructed object size and the real-world size provides a scale factor.
  - **Ground plane detection with assumed camera height**: If object priors are unavailable, the system attempts to detect the dominant ground plane using geometric plane segmentation. Assuming a typical camera height (e.g., 1.6 m), the distance between the reconstructed camera origin and detected ground plane provides a scale estimate.
  - **Bottom-image ground heuristic**: As a lightweight fallback, the system assumes that the lowest portion of the image (bottom ~5%) corresponds to the ground surface. The reconstructed distance to those points is used together with an assumed camera height to estimate scale.
- After scaling, the back-projected points form a metric 3D point cloud of the scene. To improve performance and reduce bandwidth, the point cloud is reduced using voxel grid downsampling, and then packed into a binary buffer and sent back to the user.

The result is a 3D scaled reconstruction of the scene that the user can use to measure distances in metric units, without requiring prior scene knowledge or special hardware.

### Limitations

Despite the geometric pipeline, several limitations remain:

- Absolute measurements depend on scale estimation accuracy
- Monocular depth models introduce noise and artifacts
- Unknown camera pose and tilt can distort reconstruction
- Reflective or texture-less surfaces degrade depth quality
- Scene priors rely on approximate object dimensions
- Ground assumptions may fail in cluttered environments

Measurements should therefore be considered approximate rather than survey-grade.

## Future Work

### Core

- Better fine-tuned depth estimation models
- Learned scale prediction models
- Edge-aware filtering and preprocessing
- Camera pose and intrinsics estimation
- Refined scene prior detections
- Uncertainty estimation for measurements
- Multi-view reconstruction support

### UX

- User choice of models and scaling methods
- Integration with spatial coordinate systems
- Surface reconstruction
- Measurement snapping to surfaces
- Point cloud export and download
