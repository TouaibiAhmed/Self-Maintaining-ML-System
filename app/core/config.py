"""Application settings loaded from environment variables or a .env file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, field_validator, model_validator
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
    database_url: str
    db_pool_size: PositiveInt = 5
    db_max_overflow: int = Field(default=10, ge=0)
    db_connect_timeout: PositiveInt = 5
    min_labels: PositiveInt = Field(
        default=100,
        description="Minimum labeled rows before performance metrics are reported",
    )
    monitor_window_size: PositiveInt = Field(
        default=500, description="Rows in the current monitoring window"
    )
    monitor_min_rows: PositiveInt = Field(
        default=300, description="Below this many rows a monitoring run is insufficient_data"
    )
    drift_share_threshold: float = Field(
        default=0.30,
        gt=0,
        le=1,
        description="Share of drifting features at or above which drift is detected",
    )
    monitor_interval_seconds: PositiveInt = Field(
        default=300, description="Seconds between monitoring runs in --loop mode"
    )

    @model_validator(mode="after")
    def validate_monitor_window(self) -> "Settings":
        if self.monitor_min_rows > self.monitor_window_size:
            raise ValueError("MONITOR_MIN_ROWS must not exceed MONITOR_WINDOW_SIZE")
        return self

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, value: Path) -> Path:
        if not value.is_dir():
            raise ValueError(f"MODEL_DIR must be an existing directory: {value}")
        return value
