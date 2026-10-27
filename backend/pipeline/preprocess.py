"""Single-image preprocessing for the IR enhancement pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessArtifacts:
    original: np.ndarray  # display copy (HxW or HxWx3)
    preprocessed: np.ndarray  # float32 HxW in [0,1] at model size
    thermal_model: np.ndarray  # same as preprocessed (model input)
    meta: dict[str, Any]


def preprocess_ir(
    thermal: np.ndarray,
    original_display: np.ndarray,
    target_size: int | None = None,
) -> PreprocessArtifacts:
    """Resize + normalize IR for SwinIR while preserving the original for comparison."""
    target = target_size or settings.target_size
    h, w = thermal.shape[:2]

    # Deterministic LANCZOS resize to square canvas (letterbox to preserve aspect).
    img = Image.fromarray((np.clip(thermal, 0, 1) * 255).astype(np.uint8), mode="L")
    scale = target / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (target, target), color=0)
    offset = ((target - nw) // 2, (target - nh) // 2)
    canvas.paste(resized, offset)
    pre = np.asarray(canvas, dtype=np.float32) / 255.0

    meta = {
        "original_shape": [h, w],
        "model_size": target,
        "resized_shape": [nh, nw],
        "letterbox_offset": list(offset),
        "scale": scale,
    }
    logger.info("Preprocessed IR %dx%d → %dx%d (letterbox)", w, h, target, target)
    return PreprocessArtifacts(
        original=original_display,
        preprocessed=pre,
        thermal_model=pre,
        meta=meta,
    )
