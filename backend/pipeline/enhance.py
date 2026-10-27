"""SwinIR enhancement stage (modular wrapper around models.swinir)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import settings
from models.swinir import SwinIREnhancer

logger = logging.getLogger(__name__)


class MissingWeightsError(RuntimeError):
    """Raised when SwinIR weights are required but missing."""


@dataclass
class EnhanceResult:
    enhanced: np.ndarray  # float32 HxW
    backend: str  # "swinir" | "classical_fallback"
    checkpoint: str
    warning: Optional[str] = None


_enhancer: SwinIREnhancer | None = None


def get_enhancer(force_reload: bool = False) -> SwinIREnhancer:
    global _enhancer
    if _enhancer is None or force_reload:
        ckpt = settings.swinir_checkpoint
        if settings.require_swinir_weights and not Path(ckpt).is_file():
            raise MissingWeightsError(
                f"SwinIR weights not found at '{ckpt}'. "
                "Set THERMASCOPE_SWINIR_CHECKPOINT or place thermal-swinir.pt "
                "under models/checkpoints/."
            )
        _enhancer = SwinIREnhancer(
            checkpoint=ckpt,
            scale=settings.scale_factor,
            device=settings.device,
            use_classical_if_untrained=not settings.require_swinir_weights,
        )
    return _enhancer


def enhance_ir(thermal: np.ndarray) -> EnhanceResult:
    """Run SwinIR (or configured fallback) on a preprocessed IR array."""
    enhancer = get_enhancer()
    enhanced = enhancer.enhance_array(np.asarray(thermal, dtype=np.float32))
    # Normalize display range
    enhanced = np.clip(enhanced.astype(np.float32), 0.0, None)
    if enhanced.max() > 1.0:
        lo, hi = float(enhanced.min()), float(enhanced.max())
        enhanced = (enhanced - lo) / (hi - lo + 1e-8)

    warning = None
    backend = "swinir"
    if enhancer.load_status == "random_init" or enhancer.use_classical:
        backend = "classical_fallback"
        warning = (
            "SwinIR trained weights are not installed yet, so a built-in classical "
            "enhance mode was used (denoise + upsample). Results are still valid for demo. "
            "To use neural SwinIR later, add thermal-swinir.pt under models/checkpoints/."
        )
        logger.info(warning)

    return EnhanceResult(
        enhanced=enhanced.astype(np.float32),
        backend=backend,
        checkpoint=str(settings.swinir_checkpoint),
        warning=warning,
    )
