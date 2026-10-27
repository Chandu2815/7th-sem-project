"""Image source abstraction for Satellite IR now and CCTV IR later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


class ImageSourceKind(str, Enum):
    SATELLITE_IR = "SatelliteIR"
    CCTV_IR = "CCTVIR"


@dataclass
class LoadedIRImage:
    """Normalized payload handed to the enhancement pipeline."""

    thermal: np.ndarray  # float32 HxW in [0, 1]
    original_rgb_or_gray: np.ndarray  # HxW or HxWx3 display copy of original
    source_kind: ImageSourceKind
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    guidance_rgb: Optional[np.ndarray] = None  # optional CxHxW for guided colorize


class ImageSource(ABC):
    """Load and prepare an IR image for the shared pipeline stages."""

    kind: ImageSourceKind

    @abstractmethod
    def load(self) -> LoadedIRImage:
        raise NotImplementedError


class SatelliteIRSource(ImageSource):
    """Single-file satellite infrared image (JPG/PNG/TIFF)."""

    kind = ImageSourceKind.SATELLITE_IR

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> LoadedIRImage:
        from backend.pipeline.validation import read_ir_image

        thermal, display, meta = read_ir_image(self.path)
        return LoadedIRImage(
            thermal=thermal,
            original_rgb_or_gray=display,
            source_kind=self.kind,
            path=self.path,
            meta=meta,
        )


class CCTVIRSource(ImageSource):
    """Placeholder for future CCTV infrared stills / frames."""

    kind = ImageSourceKind.CCTV_IR

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> LoadedIRImage:
        raise NotImplementedError(
            "CCTV infrared support is planned. Use SatelliteIR for now "
            "(upload a single IR JPG/PNG/TIFF)."
        )


def create_image_source(path: Path, kind: ImageSourceKind = ImageSourceKind.SATELLITE_IR) -> ImageSource:
    if kind == ImageSourceKind.SATELLITE_IR:
        return SatelliteIRSource(path)
    if kind == ImageSourceKind.CCTV_IR:
        return CCTVIRSource(path)
    raise ValueError(f"Unsupported ImageSource kind: {kind}")
