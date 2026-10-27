"""Adaptive colorization: guided (RGB available) vs self-guided thermal."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class ColorizeMode(str, Enum):
    GUIDED = "guided"
    SELF_GUIDED = "self_guided"
    AUTO = "auto"


def _normalize01(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr)
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(np.where(valid, out, 0.0), 0.0, 1.0)


def _resize_to(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize HxW or CxHxW array to target (H, W) using PIL (memory-friendly)."""
    from PIL import Image

    th, tw = shape_hw
    if arr.ndim == 2:
        if arr.shape == (th, tw):
            return arr
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        return np.array(img.resize((tw, th), Image.BILINEAR), dtype=np.float32)
    # CxHxW
    if arr.shape[-2:] == (th, tw):
        return arr
    channels = []
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i].astype(np.float32), mode="F")
        channels.append(np.array(img.resize((tw, th), Image.BILINEAR), dtype=np.float32))
    return np.stack(channels, axis=0)


class AdaptiveColorizer:
    """Colorize enhanced thermal with RGB guidance or self-guided mapping.

    Guided path: fuse thermal luminance with Sentinel/Landsat RGB chrominance.
    Self-guided: map thermal gradients to a perceptually ordered colormap (physics-tinted).
    """

    def __init__(self, guidance_weight: float = 0.7, physics_lambda: float = 0.15):
        self.guidance_weight = guidance_weight
        self.physics_lambda = physics_lambda

    def select_mode(self, rgb: Optional[np.ndarray], sentinel_available: bool = False) -> ColorizeMode:
        if sentinel_available or (rgb is not None and np.isfinite(rgb).mean() > 0.2):
            return ColorizeMode.GUIDED
        return ColorizeMode.SELF_GUIDED

    def colorize(
        self,
        thermal_sr: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        mode: ColorizeMode = ColorizeMode.AUTO,
        sentinel_available: bool = False,
    ) -> tuple[np.ndarray, ColorizeMode]:
        """Return RGB float32 HxWx3 in [0,1] and the mode used."""
        if mode == ColorizeMode.AUTO:
            mode = self.select_mode(rgb, sentinel_available)

        if mode == ColorizeMode.GUIDED and rgb is not None:
            out = self._guided(thermal_sr, rgb)
        else:
            mode = ColorizeMode.SELF_GUIDED
            out = self._self_guided(thermal_sr)

        out = self._physics_refine(out, thermal_sr)
        return out.astype(np.float32), mode

    def _guided(self, thermal: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Cross-modal: keep RGB color, modulate with thermal structure."""
        h, w = thermal.shape
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
            rgb_hwc = np.transpose(rgb[:3], (1, 2, 0))
        else:
            rgb_hwc = rgb[..., :3]
        rgb_r = _resize_to(np.transpose(rgb_hwc, (2, 0, 1)), (h, w))
        rgb_hwc = np.transpose(rgb_r, (1, 2, 0))
        for c in range(3):
            rgb_hwc[..., c] = _normalize01(rgb_hwc[..., c])

        lum_t = _normalize01(thermal)
        lum_rgb = 0.299 * rgb_hwc[..., 0] + 0.587 * rgb_hwc[..., 1] + 0.114 * rgb_hwc[..., 2]
        # Replace luminance partially with thermal while keeping chroma
        alpha = self.guidance_weight
        scale = np.where(lum_rgb > 1e-3, lum_t / (lum_rgb + 1e-3), 1.0)
        scale = (1 - alpha) + alpha * np.clip(scale, 0.5, 1.5)
        out = np.clip(rgb_hwc * scale[..., None], 0.0, 1.0)
        logger.info("Guided colorization (weight=%.2f)", self.guidance_weight)
        return out

    def _self_guided(self, thermal: np.ndarray) -> np.ndarray:
        """Map temperature to ironbow-like RGB without external RGB."""
        t = _normalize01(gaussian_filter(np.nan_to_num(thermal, nan=0.0), sigma=0.5))
        # Edge-aware structure from gradients
        gy, gx = np.gradient(t)
        edges = _normalize01(np.hypot(gx, gy))

        # Ironbow / thermal palette approximation
        r = np.clip(1.5 * t, 0, 1)
        g = np.clip(1.5 * t - 0.4, 0, 1)
        b = np.clip(1.0 - 1.2 * t + 0.2 * edges, 0, 1)
        out = np.stack([r, g, b], axis=-1)
        logger.info("Self-guided thermal colorization")
        return out

    def _physics_refine(self, rgb: np.ndarray, thermal: np.ndarray) -> np.ndarray:
        """Light Stefan-Boltzmann-inspired brightness consistency + edge preserve."""
        t = _normalize01(thermal)
        # Brightness should correlate with T^4 proxy
        phys = np.clip(t ** 4, 0, 1)
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        target = (1 - self.physics_lambda) * lum + self.physics_lambda * phys
        scale = np.where(lum > 1e-3, target / (lum + 1e-3), 1.0)
        return np.clip(rgb * scale[..., None], 0.0, 1.0)
