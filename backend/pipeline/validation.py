"""Image input validation and reading for uploaded IR files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)

# Avoid decompression-bomb false positives on Landsat-sized rasters.
Image.MAX_IMAGE_PIXELS = 200_000_000


class ValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def validate_upload_filename(filename: str) -> str:
    if not filename:
        raise ValidationError("Missing filename.")
    name = Path(filename).name
    ext = Path(name).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValidationError(
            f"Unsupported format '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    return name


def validate_file_size(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= 0:
        raise ValidationError("Empty file.")
    if size_mb > settings.max_upload_mb:
        raise ValidationError(
            f"File too large ({size_mb:.1f} MB). Max is {settings.max_upload_mb} MB."
        )


def read_ir_image(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read an IR image as float32 thermal [0,1] HxW plus a display array.

    Supports JPG/PNG via Pillow and GeoTIFF (Landsat) via rasterio when needed.
    Rejects plain color photographs that are not infrared/satellite imagery.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"File not found: {path}")
    validate_file_size(path)

    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        try:
            thermal, display, meta = _read_with_rasterio(path)
            assert_infrared_content(path, display, meta)
            return thermal, display, meta
        except ValidationError:
            raise
        except Exception as rasterio_exc:
            logger.warning("rasterio read failed (%s); trying Pillow", rasterio_exc)

    try:
        thermal, display, meta = _read_with_pillow(path)
        assert_infrared_content(path, display, meta)
        return thermal, display, meta
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Could not read image: {exc}") from exc


_IR_NAME_HINTS = (
    "st_b10",
    "sr_b",
    "lwir",
    "thermal",
    "infrared",
    "_ir_",
    "-ir-",
    "landsat",
    "lc08",
    "lc09",
    "lc8",
    "lc9",
    "sentinel",
    "tirs",
    "brightnesstemp",
    "lst",
)


def _filename_suggests_ir(path: Path) -> bool:
    name = path.name.lower()
    return any(h in name for h in _IR_NAME_HINTS)


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler–Susstrunk colorfulness (approx). Higher = more colorful photo."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return 0.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(np.nanstd(rg) ** 2 + np.nanstd(yb) ** 2) + 0.3 * np.sqrt(np.nanmean(rg) ** 2 + np.nanmean(yb) ** 2))


def _mean_saturation(rgb: np.ndarray) -> float:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    mx = np.nanmax(rgb, axis=-1)
    mn = np.nanmin(rgb, axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return float(np.nanmean(sat))


def assert_infrared_content(path: Path, display: np.ndarray, meta: dict[str, Any]) -> None:
    """Reject plain RGB photographs; allow IR / satellite / grayscale imagery.

    Raises:
        ValidationError: when the file looks like a normal color photo.
    """
    mode = str(meta.get("mode", ""))
    filename_ok = _filename_suggests_ir(path)

    # Single-band / grayscale GeoTIFF or L/I/F modes → treat as IR.
    if mode in {"GeoTIFF", "L", "I", "I;16", "F", "LA"}:
        logger.info("Accepted as IR (mode=%s, file=%s)", mode, path.name)
        return

    arr = np.asarray(display)
    if arr.ndim == 2:
        logger.info("Accepted as IR (grayscale array, file=%s)", path.name)
        return

    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # Near-grayscale RGB (IR saved as 3 identical channels)
        sample = arr
        if sample.size > 1_500_000:
            step = max(1, int(np.sqrt(sample.size / 500_000)))
            sample = sample[::step, ::step]
        colorfulness = _colorfulness(sample)
        saturation = _mean_saturation(sample)

        # Strong IR filename override for weakly colorful composites
        if filename_ok and colorfulness < 45 and saturation < 0.45:
            logger.info(
                "Accepted as IR via filename hints (colorfulness=%.1f sat=%.2f file=%s)",
                colorfulness,
                saturation,
                path.name,
            )
            return

        # Plain photos are typically colorful
        if colorfulness >= 20 or saturation >= 0.22:
            raise ValidationError(
                "This looks like a normal color photo, not an infrared / satellite IR image. "
                "Please upload a Landsat thermal/reflectance band (.TIF), a grayscale IR image, "
                "or an IR JPG/PNG. Plain photos are not processed."
            )

        logger.info(
            "Accepted as low-chroma IR-like RGB (colorfulness=%.1f sat=%.2f file=%s)",
            colorfulness,
            saturation,
            path.name,
        )
        return

    raise ValidationError("Unsupported image layout for IR validation.")


def _read_with_rasterio(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        width, height = ds.width, ds.height
        if width < 16 or height < 16:
            raise ValidationError(f"Image too small ({width}x{height}). Minimum 16x16.")
        if width > 20000 or height > 20000:
            raise ValidationError(f"Image too large ({width}x{height}).")

        # Downsample huge Landsat scenes while reading to keep memory/time reasonable.
        max_side = max(settings.target_size * 4, 2048)
        scale = min(1.0, max_side / max(width, height))
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))

        if scale < 1.0:
            data = ds.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.average,
            ).astype(np.float32)
            logger.info(
                "Read GeoTIFF %s downsampled %dx%d → %dx%d",
                path.name,
                width,
                height,
                out_w,
                out_h,
            )
        else:
            data = ds.read(1).astype(np.float32)
            out_h, out_w = height, width

        nodata = ds.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

    thermal = _to_unit_range(data)
    display = thermal
    meta = {
        "width": out_w,
        "height": out_h,
        "source_width": width,
        "source_height": height,
        "mode": "GeoTIFF",
        "path": str(path),
        "filename": path.name,
        "downsampled": scale < 1.0,
    }
    logger.info("Validated GeoTIFF %s (%dx%d)", path.name, out_w, out_h)
    return thermal.astype(np.float32), display, meta


def _read_with_pillow(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with Image.open(path) as img:
        img.load()
        mode = img.mode
        width, height = img.size
        if width < 16 or height < 16:
            raise ValidationError(f"Image too small ({width}x{height}). Minimum 16x16.")
        if width > 20000 or height > 20000:
            raise ValidationError(f"Image too large ({width}x{height}).")

        # Downsample very large images before converting to arrays.
        max_side = max(settings.target_size * 4, 2048)
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            nw, nh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            width, height = nw, nh

        if mode in ("I;16", "I", "F"):
            arr = np.array(img, dtype=np.float32)
            display = arr
        elif mode == "L":
            arr = np.array(img, dtype=np.float32)
            display = arr
        else:
            rgb = np.array(img.convert("RGB"), dtype=np.float32)
            display = rgb
            arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    thermal = _to_unit_range(arr)
    meta = {
        "width": width,
        "height": height,
        "mode": mode,
        "path": str(path),
        "filename": path.name,
    }
    return thermal.astype(np.float32), display, meta


def _to_unit_range(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
