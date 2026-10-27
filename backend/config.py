"""Central configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(key: str, default: Path) -> Path:
    value = os.getenv(key)
    return Path(value) if value else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    return int(value)


class Settings:
    """Runtime settings for the IR enhancement pipeline."""

    project_root: Path = PROJECT_ROOT
    upload_dir: Path = _env_path("THERMASCOPE_UPLOAD_DIR", PROJECT_ROOT / "outputs" / "uploads")
    jobs_dir: Path = _env_path("THERMASCOPE_JOBS_DIR", PROJECT_ROOT / "outputs" / "jobs")
    swinir_checkpoint: Path = _env_path(
        "THERMASCOPE_SWINIR_CHECKPOINT",
        PROJECT_ROOT / "models" / "checkpoints" / "thermal-swinir.pt",
    )
    detector_weights: Path = _env_path(
        "THERMASCOPE_DETECTOR_WEIGHTS",
        PROJECT_ROOT / "models" / "checkpoints" / "yolov8n.pt",
    )
    # If true, missing SwinIR weights raise an error instead of classical fallback.
    require_swinir_weights: bool = _env_bool("THERMASCOPE_REQUIRE_SWINIR_WEIGHTS", False)
    target_size: int = _env_int("THERMASCOPE_TARGET_SIZE", 512)
    scale_factor: int = _env_int("THERMASCOPE_SCALE", 2)
    max_upload_mb: int = _env_int("THERMASCOPE_MAX_UPLOAD_MB", 200)
    job_ttl_hours: int = _env_int("THERMASCOPE_JOB_TTL_HOURS", 24)
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    device: str = os.getenv(
        "THERMASCOPE_DEVICE",
        "cuda" if __import__("torch").cuda.is_available() else "cpu",
    )


settings = Settings()
