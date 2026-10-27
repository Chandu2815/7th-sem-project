"""Landsat Collection 2 Level-2 preprocessing pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "Processed"

REQUIRED_BANDS = ("ST_B10", "SR_B2", "SR_B3", "SR_B4")

# Landsat C2 L2 scale/offset (USGS Collection 2)
ST_SCALE = 0.00341802
ST_OFFSET = 149.0
SR_SCALE = 2.75e-05
SR_OFFSET = -0.2


@dataclass
class PreprocessConfig:
    """Configuration for scene preprocessing."""

    patch_size: int = 128
    stride: int = 64
    scale_factor: int = 4
    cloud_threshold: float = 0.0  # reserved for QA_PIXEL when present
    clip_sigma: float = 3.0
    raw_dir: Path = DEFAULT_RAW_DIR
    output_dir: Path = DEFAULT_PROCESSED_DIR


@dataclass
class PreprocessResult:
    """Artifacts produced by preprocessing one scene."""

    scene_id: str
    output_dir: str
    thermal_path: str
    rgb_path: str
    patches_dir: str
    num_patches: int
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


def _log(result: PreprocessResult, message: str) -> None:
    logger.info(message)
    result.logs.append(message)


def _read_band(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float64)
        profile = ds.profile.copy()
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs
    meta = {
        "profile": {k: (str(v) if k in ("crs", "transform") else v) for k, v in profile.items()},
        "nodata": nodata,
        "transform": transform,
        "crs": crs,
        "width": profile["width"],
        "height": profile["height"],
    }
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, meta


def calibrate_thermal(dn: np.ndarray) -> np.ndarray:
    """Convert ST_B10 DN to brightness temperature (Kelvin)."""
    return dn * ST_SCALE + ST_OFFSET


def calibrate_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert SR band DN to surface reflectance."""
    return np.clip(dn * SR_SCALE + SR_OFFSET, 0.0, 1.0)


def simple_cloud_mask(rgb: np.ndarray, reflectance_high: float = 0.45) -> np.ndarray:
    """Heuristic cloud mask when QA_PIXEL is unavailable (True = valid)."""
    # rgb shape: (3, H, W) in reflectance
    bright = np.nanmean(rgb, axis=0) > reflectance_high
    return ~bright


def align_thermal_to_rgb(
    thermal: np.ndarray,
    thermal_meta: dict[str, Any],
    rgb_ref_meta: dict[str, Any],
) -> np.ndarray:
    """Reproject thermal onto the RGB grid (same CRS/transform/shape)."""
    dst = np.full(
        (rgb_ref_meta["height"], rgb_ref_meta["width"]),
        np.nan,
        dtype=np.float64,
    )
    reproject(
        source=thermal.astype(np.float64),
        destination=dst,
        src_transform=thermal_meta["transform"],
        src_crs=thermal_meta["crs"],
        dst_transform=rgb_ref_meta["transform"],
        dst_crs=rgb_ref_meta["crs"],
        resampling=Resampling.cubic,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def zscore_normalize(arr: np.ndarray, clip_sigma: float = 3.0) -> tuple[np.ndarray, dict[str, float]]:
    """Per-array z-score with sigma clipping; NaNs preserved."""
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr, {"mean": 0.0, "std": 1.0}
    vals = arr[valid]
    mean = float(np.mean(vals))
    std = float(np.std(vals)) or 1.0
    out = (arr - mean) / std
    out = np.clip(out, -clip_sigma, clip_sigma)
    out = np.where(valid, out, np.nan)
    return out, {"mean": mean, "std": std}


def extract_patches(
    thermal: np.ndarray,
    rgb: np.ndarray,
    patch_size: int,
    stride: int,
    out_dir: Path,
) -> int:
    """Write paired thermal/RGB patches as .npz files. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = thermal.shape
    count = 0
    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            t_patch = thermal[y : y + patch_size, x : x + patch_size]
            r_patch = rgb[:, y : y + patch_size, x : x + patch_size]
            if t_patch.shape != (patch_size, patch_size):
                continue
            if r_patch.shape[1:] != (patch_size, patch_size):
                continue
            if np.isnan(t_patch).mean() > 0.3:
                continue
            np.savez_compressed(
                out_dir / f"patch_{count:05d}.npz",
                thermal=t_patch.astype(np.float32),
                rgb=r_patch.astype(np.float32),
                y=y,
                x=x,
            )
            count += 1
    return count


def _write_geotiff(path: Path, data: np.ndarray, ref_meta: dict[str, Any], count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2] if data.ndim == 3 else data.shape[0],
        "width": data.shape[-1] if data.ndim == 3 else data.shape[1],
        "count": count,
        "dtype": "float32",
        "crs": ref_meta["crs"],
        "transform": ref_meta["transform"],
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as ds:
        if data.ndim == 2:
            ds.write(data.astype(np.float32), 1)
        else:
            for i in range(count):
                ds.write(data[i].astype(np.float32), i + 1)


def preprocess_scene(scene_id: str, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Run full preprocessing for one scene under dataset/raw/<scene_id>."""
    config = config or PreprocessConfig()
    scene_dir = Path(config.raw_dir) / scene_id
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    out_dir = Path(config.output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = PreprocessResult(
        scene_id=scene_id,
        output_dir=str(out_dir),
        thermal_path="",
        rgb_path="",
        patches_dir="",
        num_patches=0,
    )

    # --- Band load ---
    result.steps.append({"id": "raw-scene", "name": "Raw Scene", "status": "completed", "progress": 100})
    bands: dict[str, np.ndarray] = {}
    metas: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_BANDS:
        path = scene_dir / f"{name}.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Missing band {name}: {path}")
        bands[name], metas[name] = _read_band(path)
    _log(result, f"Loaded bands for {scene_id}")

    # --- Radiometric calibration ---
    thermal_k = calibrate_thermal(bands["ST_B10"])
    rgb = np.stack(
        [
            calibrate_reflectance(bands["SR_B2"]),
            calibrate_reflectance(bands["SR_B3"]),
            calibrate_reflectance(bands["SR_B4"]),
        ],
        axis=0,
    )
    result.steps.append(
        {"id": "radiometric-calibration", "name": "Radiometric Calibration", "status": "completed", "progress": 100}
    )
    _log(result, "Applied C2 L2 scale/offset (ST Kelvin, SR reflectance)")

    # --- Cloud masking ---
    valid = simple_cloud_mask(rgb)
    thermal_k = np.where(valid, thermal_k, np.nan)
    rgb = np.where(valid[None, ...], rgb, np.nan)
    retained = float(np.nanmean(valid.astype(np.float64))) * 100.0
    result.steps.append({"id": "cloud-masking", "name": "Cloud Masking", "status": "completed", "progress": 100})
    _log(result, f"Cloud masking retained {retained:.1f}% pixels")

    # --- Alignment ---
    rgb_meta = metas["SR_B2"]
    if (
        metas["ST_B10"]["width"] != rgb_meta["width"]
        or metas["ST_B10"]["height"] != rgb_meta["height"]
        or metas["ST_B10"]["transform"] != rgb_meta["transform"]
    ):
        thermal_k = align_thermal_to_rgb(thermal_k, metas["ST_B10"], rgb_meta)
        _log(result, "Reprojected thermal onto RGB grid")
    else:
        _log(result, "Thermal already aligned to RGB grid")
    result.steps.append(
        {"id": "thermal-rgb-alignment", "name": "Thermal-RGB Alignment", "status": "completed", "progress": 100}
    )

    # --- Normalization ---
    thermal_n, t_stats = zscore_normalize(thermal_k, config.clip_sigma)
    rgb_n = np.zeros_like(rgb)
    rgb_stats = []
    for i in range(3):
        rgb_n[i], st = zscore_normalize(rgb[i], config.clip_sigma)
        rgb_stats.append(st)
    result.steps.append({"id": "normalization", "name": "Normalization", "status": "completed", "progress": 100})
    _log(result, "Applied z-score normalization with sigma clipping")

    # --- Write calibrated rasters ---
    thermal_path = out_dir / "thermal_kelvin.tif"
    rgb_path = out_dir / "rgb_reflectance.tif"
    thermal_norm_path = out_dir / "thermal_normalized.tif"
    _write_geotiff(thermal_path, np.nan_to_num(thermal_k, nan=0.0), rgb_meta)
    _write_geotiff(rgb_path, np.nan_to_num(rgb, nan=0.0), rgb_meta, count=3)
    _write_geotiff(thermal_norm_path, np.nan_to_num(thermal_n, nan=0.0), rgb_meta)

    # --- Patches ---
    patches_dir = out_dir / "patches"
    num_patches = extract_patches(thermal_n, rgb_n, config.patch_size, config.stride, patches_dir)
    result.steps.append(
        {"id": "patch-extraction", "name": "Patch Extraction", "status": "completed", "progress": 100}
    )
    result.steps.append(
        {"id": "processed-dataset", "name": "Processed Dataset", "status": "completed", "progress": 100}
    )
    _log(result, f"Extracted {num_patches} patches ({config.patch_size}px, stride {config.stride})")

    result.thermal_path = str(thermal_norm_path)
    result.rgb_path = str(rgb_path)
    result.patches_dir = str(patches_dir)
    result.num_patches = num_patches
    result.metadata = {
        "thermal_stats": t_stats,
        "rgb_stats": rgb_stats,
        "retained_pct": retained,
        "patch_size": config.patch_size,
        "stride": config.stride,
        "scale_factor": config.scale_factor,
        "crs": str(rgb_meta["crs"]),
    }

    meta_path = out_dir / "preprocess_report.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    _log(result, f"Wrote report: {meta_path}")
    return result
