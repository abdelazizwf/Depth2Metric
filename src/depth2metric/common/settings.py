from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    logging_level: str = Field("DEBUG")
    models_dir: str = Field("./models")
    samples_dir: str = Field("./static/samples")
    precomputed_dir: str = Field("./static/precomputed")

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_settings():
    return Settings()
