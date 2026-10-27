"""End-to-end image enhancement pipeline (PPT ImageEnhancementPipeline)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from PIL import Image

from evaluation.metrics import EvaluationResult, evaluate_pair
from models.colorization import AdaptiveColorizer, ColorizeMode
from models.swinir import SwinIREnhancer
from preprocessing.pipeline import PreprocessConfig, PreprocessResult, preprocess_scene

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "enhanced"


@dataclass
class PipelineResult:
    scene_id: str
    status: str
    mode: str
    enhanced_thermal_path: str
    colorized_path: str
    geotiff_path: str
    preview_path: str
    evaluation: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    created_at: str = ""


class ImageEnhancementPipeline:
    """preprocess → SwinIR SR/denoise → adaptive colorize → physics refine → export."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        scale: int = 4,
        output_dir: Path | str = DEFAULT_OUTPUT,
        patch_size: int = 128,
    ):
        self.scale = scale
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enhancer = SwinIREnhancer(checkpoint=checkpoint, scale=scale)
        self.colorizer = AdaptiveColorizer()
        self.patch_size = patch_size

    def preprocess(self, scene_id: str) -> PreprocessResult:
        cfg = PreprocessConfig(patch_size=self.patch_size, scale_factor=self.scale)
        return preprocess_scene(scene_id, cfg)

    def enhance(self, thermal: np.ndarray) -> np.ndarray:
        return self.enhancer.enhance_array(thermal)

    def super_resolve(self, thermal: np.ndarray) -> np.ndarray:
        return self.enhance(thermal)

    def colorize(
        self,
        thermal_sr: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        sentinel_available: bool = False,
    ) -> tuple[np.ndarray, ColorizeMode]:
        return self.colorizer.colorize(
            thermal_sr, rgb=rgb, mode=ColorizeMode.AUTO, sentinel_available=sentinel_available
        )

    def physics_aware_refine(self, rgb: np.ndarray, thermal_sr: np.ndarray) -> np.ndarray:
        return self.colorizer._physics_refine(rgb, thermal_sr)

    def run(
        self,
        scene_id: str,
        sentinel_available: bool = False,
        skip_preprocess_if_exists: bool = True,
    ) -> PipelineResult:
        logs: list[str] = []
        logs.append(f"Pipeline start: {scene_id}")

        processed_dir = PROJECT_ROOT / "dataset" / "Processed" / scene_id
        report_path = processed_dir / "preprocess_report.json"
        if skip_preprocess_if_exists and report_path.is_file():
            logs.append("Reusing existing preprocess artifacts")
            with report_path.open(encoding="utf-8") as f:
                pre_dict = json.load(f)
            thermal_path = Path(pre_dict["thermal_path"])
            rgb_path = Path(pre_dict["rgb_path"])
        else:
            pre = self.preprocess(scene_id)
            pre_dict = asdict(pre)
            thermal_path = Path(pre.thermal_path)
            rgb_path = Path(pre.rgb_path)
            logs.extend(pre.logs)

        with rasterio.open(thermal_path) as ds:
            thermal = ds.read(1).astype(np.float32)
            profile = ds.profile.copy()

        with rasterio.open(rgb_path) as ds:
            rgb = ds.read().astype(np.float32)

        # Memory-safe enhancement (downsamples large scenes before SR)
        logs.append(f"SwinIR enhance (status={self.enhancer.load_status})")
        thermal_sr = self._enhance_tiled(thermal)
        logs.append(f"Enhanced thermal shape: {thermal_sr.shape}")

        # Match RGB guidance to enhanced resolution before colorize
        from PIL import Image as PILImage

        rgb_for_color = np.stack(
            [
                np.array(
                    PILImage.fromarray(rgb[i].astype(np.float32), mode="F").resize(
                        (thermal_sr.shape[1], thermal_sr.shape[0]), PILImage.BILINEAR
                    ),
                    dtype=np.float32,
                )
                for i in range(min(3, rgb.shape[0]))
            ],
            axis=0,
        )

        color_rgb, mode = self.colorize(
            thermal_sr, rgb=rgb_for_color, sentinel_available=sentinel_available
        )
        logs.append(f"Colorization mode: {mode.value}")

        out_dir = self.output_dir / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        thermal_out = out_dir / "thermal_sr.tif"
        color_tif = out_dir / "colorized.tif"
        color_png = out_dir / "colorized_preview.png"
        geotiff = out_dir / "enhanced_stack.tif"

        # Effective scale vs original raster for geotransform
        eff_scale = thermal_sr.shape[0] / max(thermal.shape[0], 1)
        self._write_single(thermal_out, thermal_sr, profile, eff_scale)
        self._write_rgb_geotiff(color_tif, color_rgb, profile, eff_scale)
        self._write_rgb_geotiff(geotiff, color_rgb, profile, eff_scale)
        Image.fromarray((np.clip(color_rgb, 0, 1) * 255).astype(np.uint8)).save(color_png)

        # Evaluate against resized input as weak reference
        from scipy.ndimage import zoom

        ref = zoom(
            np.nan_to_num(thermal, nan=0.0),
            (thermal_sr.shape[0] / thermal.shape[0], thermal_sr.shape[1] / thermal.shape[1]),
            order=3,
        )
        h = min(thermal_sr.shape[0], ref.shape[0])
        w = min(thermal_sr.shape[1], ref.shape[1])
        ev = evaluate_pair(thermal_sr[:h, :w], ref[:h, :w], scene_id=scene_id)
        logs.append(f"Eval PSNR={ev.psnr:.2f} SSIM={ev.ssim:.3f}")

        result = PipelineResult(
            scene_id=scene_id,
            status="completed",
            mode=mode.value,
            enhanced_thermal_path=str(thermal_out),
            colorized_path=str(color_tif),
            geotiff_path=str(geotiff),
            preview_path=str(color_png),
            evaluation=ev.to_dict(),
            preprocess={k: pre_dict.get(k) for k in ("num_patches", "metadata", "steps") if k in pre_dict},
            logs=logs,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with (out_dir / "pipeline_report.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        return result

    def _enhance_tiled(self, thermal: np.ndarray, tile: int = 256, overlap: int = 16) -> np.ndarray:
        """Process large rasters at a memory-safe working resolution.

        Full Landsat scenes (~7k×7k) at 4× would exceed typical RAM, so we
        downsample to ``max_side`` first, run SR, and keep that enhanced size
        (do not expand back to original×scale).
        """
        h, w = thermal.shape
        max_side = 1024
        data = np.nan_to_num(thermal, nan=0.0)
        if max(h, w) > max_side:
            from scipy.ndimage import zoom

            scale_down = max_side / max(h, w)
            data = zoom(data, scale_down, order=1).astype(np.float32)
        return self.enhancer.enhance_array(data)

    @staticmethod
    def _write_single(path: Path, data: np.ndarray, ref_profile: dict, scale: float) -> None:
        transform = ref_profile["transform"]
        scale = float(scale) if scale else 1.0
        new_transform = transform * transform.scale(1 / scale, 1 / scale)
        profile = ref_profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            transform=new_transform,
            compress="lzw",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(data.astype(np.float32), 1)

    @staticmethod
    def _write_rgb_geotiff(path: Path, rgb: np.ndarray, ref_profile: dict, scale: float) -> None:
        # rgb HxWx3
        transform = ref_profile["transform"]
        scale = float(scale) if scale else 1.0
        new_transform = transform * transform.scale(1 / scale, 1 / scale)
        profile = ref_profile.copy()
        profile.update(
            height=rgb.shape[0],
            width=rgb.shape[1],
            count=3,
            dtype="float32",
            transform=new_transform,
            compress="lzw",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as ds:
            for i in range(3):
                ds.write(rgb[:, :, i].astype(np.float32), i + 1)
