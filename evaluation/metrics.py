"""Image quality metrics for enhanced vs reference imagery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class EvaluationResult:
    psnr: float
    ssim: float
    lpips: float  # placeholder approximate perceptual distance
    rmse: float
    scene_id: str
    model_version: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _lpips_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap perceptual proxy when true LPIPS weights are unavailable."""
    a = _to_float01(a)
    b = _to_float01(b)
    if a.shape != b.shape:
        # compare on min shared spatial size
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    diff = np.abs(a - b)
    # emphasize edges
    return float(np.mean(diff) + 0.5 * np.mean(diff ** 2))


def evaluate_pair(
    pred: np.ndarray,
    reference: np.ndarray,
    scene_id: str = "",
    model_version: str = "thermal-swinir-v0.1",
) -> EvaluationResult:
    """Compute PSNR, SSIM, LPIPS-proxy, RMSE between prediction and reference."""
    p = _to_float01(pred)
    r = _to_float01(reference)
    h = min(p.shape[0], r.shape[0])
    w = min(p.shape[1], r.shape[1])
    p, r = p[:h, :w], r[:h, :w]

    # grayscale for PSNR/SSIM if multi-channel
    if p.ndim == 3:
        p_g = p.mean(axis=2)
        r_g = r.mean(axis=2) if r.ndim == 3 else r
    else:
        p_g, r_g = p, r if r.ndim == 2 else r.mean(axis=2)

    psnr = float(peak_signal_noise_ratio(r_g, p_g, data_range=1.0))
    ssim = float(structural_similarity(r_g, p_g, data_range=1.0))
    rmse = float(np.sqrt(np.mean((p_g - r_g) ** 2)))
    lpips = _lpips_proxy(p, r)

    return EvaluationResult(
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        rmse=rmse,
        scene_id=scene_id,
        model_version=model_version,
        notes="lpips is a proxy metric without AlexNet weights",
    )
