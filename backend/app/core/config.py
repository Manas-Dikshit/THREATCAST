"""Typed application settings (CONTRACT.md §3). Loads from environment / .env."""

import functools

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_port: int = 5173
    postgres_port: int = 5432

    database_url: str = "postgresql://threatcast:threatcast@localhost:5432/threatcast"

    ml_model_path: str = "./ml/artifacts/world_model.pt"
    ml_metadata_path: str = "./ml/artifacts/model_metadata.json"
    ml_device: str = "auto"
    ml_sequence_length: int = 5
    ml_prediction_horizon: int = 3

    time_window_seconds: float = 10.0
    max_upload_size_mb: int = 500

    log_level: str = "INFO"


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
