"""Result artifact generation, comparison mosaic, and cleanup helpers."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class JobArtifacts:
    job_id: str
    original_path: str
    preprocessed_path: str
    enhanced_path: str
    colorized_path: str
    comparison_path: str
    annotated_path: str = ""
    report_path: str = ""
    detections: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def job_dir(job_id: str) -> Path:
    path = settings.jobs_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_gray(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_u8_gray(arr), mode="L").save(path)
    return path


def save_rgb(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_u8_rgb(arr), mode="RGB").save(path)
    return path


def save_display_original(path: Path, arr: np.ndarray) -> Path:
    a = np.asarray(arr)
    if a.ndim == 2:
        return save_gray(path, a)
    return save_rgb(path, a)


def _finite_minmax(a: np.ndarray) -> tuple[float, float]:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _to_u8_gray(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = np.nanmean(a, axis=2)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.dtype == np.uint8 or (a.max() > 1.5 and a.max() <= 255):
        return np.clip(a, 0, 255).astype(np.uint8)
    lo, hi = _finite_minmax(a)
    scaled = (a - lo) / (hi - lo + 1e-8)
    return (np.clip(scaled, 0, 1) * 255.0).astype(np.uint8)


def _to_u8_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        g = _to_u8_gray(a)
        return np.stack([g, g, g], axis=-1)
    out = np.zeros((*a.shape[:2], 3), dtype=np.uint8)
    for c in range(min(3, a.shape[2] if a.shape[-1] <= 4 else a.shape[0])):
        if a.shape[-1] <= 4:
            channel = a[..., c]
        else:
            channel = a[c]
        out[..., c] = _to_u8_gray(channel)
    return out


def make_comparison(
    original: np.ndarray,
    preprocessed: np.ndarray,
    enhanced: np.ndarray,
    colorized: np.ndarray,
    out_path: Path,
    labels: tuple[str, ...] = ("Original", "Preprocessed", "Enhanced IR", "Colorized"),
) -> Path:
    """2x2 labeled comparison mosaic for download/UI."""
    panels = [
        _to_u8_rgb(original),
        _to_u8_rgb(preprocessed),
        _to_u8_rgb(enhanced),
        _to_u8_rgb(colorized),
    ]
    th, tw = panels[2].shape[:2]
    panels = [_resize_rgb(p, (tw, th)) for p in panels]

    mosaic = np.zeros((th * 2, tw * 2, 3), dtype=np.uint8)
    mosaic[0:th, 0:tw] = panels[0]
    mosaic[0:th, tw : tw * 2] = panels[1]
    mosaic[th : th * 2, 0:tw] = panels[2]
    mosaic[th : th * 2, tw : tw * 2] = panels[3]

    img = Image.fromarray(mosaic)
    draw = ImageDraw.Draw(img)
    positions = [(8, 8), (tw + 8, 8), (8, th + 8), (tw + 8, th + 8)]
    for label, pos in zip(labels, positions):
        draw.rectangle(
            [pos[0] - 4, pos[1] - 2, pos[0] + 8 * len(label) + 8, pos[1] + 16],
            fill=(0, 0, 0),
        )
        draw.text(pos, label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def cleanup_old_jobs(ttl_hours: int | None = None) -> int:
    ttl = ttl_hours if ttl_hours is not None else settings.job_ttl_hours
    root = settings.jobs_dir
    if not root.is_dir():
        return 0
    cutoff = time.time() - ttl * 3600
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("Cleaned %d expired job folder(s)", removed)
    return removed


def cleanup_uploads(ttl_hours: int | None = None) -> int:
    ttl = ttl_hours if ttl_hours is not None else settings.job_ttl_hours
    root = settings.upload_dir
    if not root.is_dir():
        return 0
    cutoff = time.time() - ttl * 3600
    removed = 0
    for child in root.iterdir():
        try:
            if child.stat().st_mtime < cutoff:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def _resize_rgb(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(arr)
    return np.asarray(img.resize(size_wh, Image.Resampling.BILINEAR))
