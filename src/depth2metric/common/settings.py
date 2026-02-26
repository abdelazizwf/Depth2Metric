from typing import Any

from pydantic import DirectoryPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PRIORS = {
    0: ["person", 170],
    56: ["chair", 80],
    57: ["couch", 80],
    59: ["bed", 100],
    60: ["dining_table", 75],
    39: ["bottle", 25],
    63: ["laptop", 30],
    2: ["car", 120],
    7: ["truck", 300],
    68: ["microwave", 30],
    69: ["oven", 80],
    72: ["refrigerator", 200],
}

class Settings(BaseSettings):
    logging_level: str = Field("INFO")
    models_dir: DirectoryPath = Field("./models") # type: ignore
    samples_dir: DirectoryPath = Field("./static/samples") # type: ignore
    precomputed_dir: str = Field("./static/precomputed")

    # Models
    midas_model: str = Field("DPT_Hybrid")
    yolo_model: str = Field("yolo26n")

    # Scaling & Geometry
    assumed_camera_height: float = Field(160.0)
    voxel_size: float = Field(0.7)
    ransac_distance_threshold: float = Field(0.5)
    ransac_n: int = Field(10)
    ransac_iterations: int = Field(500)
    ground_vertical_threshold: float = Field(0.75)
    fallback_scale_factor: float = Field(0.3)

    # Edge Case Fixes
    enable_edge_cropping: bool = Field(True)
    edge_cropping_percentage: float = Field(0.07)

    enable_percentile_clipping: bool = Field(True)
    percentile_low: float = Field(3.0)
    percentile_high: float = Field(95.0)

    # Scene Priors
    size_priors: dict[int, list[Any]] = Field(default_factory=lambda: DEFAULT_PRIORS)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_settings():
    return Settings() # type: ignore
