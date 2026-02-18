from pydantic import DirectoryPath, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    logging_level: str = Field("INFO")
    models_dir: DirectoryPath = Field("./models")
    samples_dir: DirectoryPath = Field("./static/samples")
    precomputed_dir: str = Field("./static/precomputed")
    assumed_camera_height: float = Field(160.0)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_settings():
    return Settings()
