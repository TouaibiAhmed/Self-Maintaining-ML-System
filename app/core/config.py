"""Application settings loaded from environment variables or a .env file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Validated runtime settings for the prediction service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_dir: Path = Field(..., description="Path to the active model artifact directory")
    log_level: LogLevel
    max_body_bytes: PositiveInt

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, value: Path) -> Path:
        if not value.is_dir():
            raise ValueError(f"MODEL_DIR must be an existing directory: {value}")
        return value
