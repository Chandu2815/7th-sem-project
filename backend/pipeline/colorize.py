"""Colorization stage — keeps enhanced IR intact; produces a separate color view."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.colorization import AdaptiveColorizer, ColorizeMode

logger = logging.getLogger(__name__)


@dataclass
class ColorizeResult:
    colorized: np.ndarray  # HxWx3 float32 [0,1]
    mode: str
    enhanced_ir_preserved: np.ndarray  # same as input enhanced (not mutated)


def colorize_ir(
    enhanced_ir: np.ndarray,
    guidance_rgb: Optional[np.ndarray] = None,
) -> ColorizeResult:
    """Colorize enhanced IR without mutating the scientific IR array."""
    ir_copy = np.array(enhanced_ir, dtype=np.float32, copy=True)
    colorizer = AdaptiveColorizer()
    # For single-file satellite IR uploads there is usually no co-registered RGB.
    rgb_out, mode = colorizer.colorize(
        ir_copy,
        rgb=guidance_rgb,
        mode=ColorizeMode.AUTO if guidance_rgb is not None else ColorizeMode.SELF_GUIDED,
        sentinel_available=False,
    )
    logger.info("Colorization mode=%s", mode.value)
    return ColorizeResult(
        colorized=rgb_out.astype(np.float32),
        mode=mode.value,
        enhanced_ir_preserved=ir_copy,
    )
