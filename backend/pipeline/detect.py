"""Object interpretation / detection stage (swappable backend)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float]  # xyxy


@dataclass
class DetectResult:
    detections: list[Detection] = field(default_factory=list)
    annotated: Optional[np.ndarray] = None  # HxWx3 uint8
    backend: str = "none"
    message: str = ""


def detect_objects(colorized_rgb: np.ndarray, conf: float = 0.25) -> DetectResult:
    """Run object detection on the colorized image.

    Uses local YOLO weights only (no auto-download) so the API does not hang.
    Falls back to a stub result when weights/ultralytics are unavailable.
    """
    rgb_u8 = _to_uint8_rgb(colorized_rgb)

    try:
        return _detect_yolo(rgb_u8, conf=conf)
    except Exception as exc:
        logger.warning("YOLO detector skipped (%s)", exc)
        return DetectResult(
            detections=[],
            annotated=rgb_u8,
            backend="stub",
            message=(
                "Object detection skipped (no local YOLO weights or ultralytics). "
                f"Optional: place weights at '{settings.detector_weights}'. "
                "Enhancement results above are still valid."
            ),
        )


def _detect_yolo(rgb_u8: np.ndarray, conf: float) -> DetectResult:
    weights_path = Path(settings.detector_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"No local detector weights at {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.predict(source=rgb_u8, conf=conf, verbose=False, device=settings.device)
    detections: list[Detection] = []
    annotated = rgb_u8.copy()

    if not results:
        return DetectResult(detections=[], annotated=annotated, backend="yolo", message="No results")

    r0 = results[0]
    names = r0.names or {}
    if r0.boxes is not None and len(r0.boxes):
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls_id in zip(xyxy, confs, clss):
            detections.append(
                Detection(
                    label=str(names.get(int(cls_id), cls_id)),
                    confidence=float(c),
                    bbox=[float(x) for x in box.tolist()],
                )
            )
        plotted = r0.plot()  # BGR
        annotated = plotted[:, :, ::-1].copy()

    return DetectResult(
        detections=detections,
        annotated=annotated,
        backend="yolo",
        message=f"{len(detections)} object(s) detected",
    )


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 1)
        if a.max() <= 1.0:
            a = a * 255.0
        a = a.astype(np.uint8)
    return a
