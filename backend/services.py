"""DatasetManager, ReportGenerator — PPT class-diagram services."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from models.pipeline import ImageEnhancementPipeline, PipelineResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "Processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "enhanced"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PREVIEWS_DIR = PROJECT_ROOT / "outputs" / "previews"


class DatasetManager:
    """List / inspect / validate / run inference for local Landsat scenes."""

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        model_weights_path: str = str(PROJECT_ROOT / "models" / "checkpoints" / "thermal-swinir.pt"),
    ):
        self.raw_dir = Path(raw_dir)
        self.model_weights_path = model_weights_path
        self.local_repository = str(PROJECT_ROOT / "dataset")
        self._pipeline: ImageEnhancementPipeline | None = None

    def _get_pipeline(self, scale: int = 4) -> ImageEnhancementPipeline:
        if self._pipeline is None:
            self._pipeline = ImageEnhancementPipeline(
                checkpoint=self.model_weights_path,
                scale=scale,
                output_dir=OUTPUT_DIR,
            )
        return self._pipeline

    def list_datasets(self) -> list[dict[str, Any]]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        scenes = []
        if not self.raw_dir.is_dir():
            return scenes
        for p in sorted(self.raw_dir.iterdir()):
            if not p.is_dir():
                continue
            bands = [b.stem for b in p.glob("*.tif")]
            status = "downloaded"
            if (PROCESSED_DIR / p.name / "preprocess_report.json").is_file():
                status = "preprocessed"
            if (OUTPUT_DIR / p.name / "pipeline_report.json").is_file():
                status = "processed"
            size = sum(f.stat().st_size for f in p.glob("*.tif"))
            scenes.append(
                {
                    "id": p.name,
                    "sceneId": p.name,
                    "satellite": "Landsat-9" if p.name.startswith("LC09") else "Landsat-8",
                    "status": status,
                    "bands": bands,
                    "downloadSizeBytes": size,
                    "location": "Local dataset",
                    "cloudCover": 0.0,
                    "acquisitionDate": _guess_date(p.name),
                    "pathRow": _guess_path_row(p.name),
                    "centerLat": 0.0,
                    "centerLon": 0.0,
                    "processingProgress": 100 if status == "processed" else (45 if status == "preprocessed" else 0),
                    "sentinelRgbAvailable": "SR_B2" in bands,
                    "boundingBox": {"west": 0, "south": 0, "east": 0, "north": 0},
                    "thermalBand": {"name": "ST_B10", "resolution": "100m", "sizeBytes": 0},
                    "rgbBands": [
                        {"name": "SR_B2", "resolution": "30m", "sizeBytes": 0},
                        {"name": "SR_B3", "resolution": "30m", "sizeBytes": 0},
                        {"name": "SR_B4", "resolution": "30m", "sizeBytes": 0},
                    ],
                    "previewGradient": "from-orange-950 via-red-900/80 to-purple-950",
                }
            )
        return scenes

    def get_scene_metadata(self, scene_id: str) -> dict[str, Any] | None:
        for s in self.list_datasets():
            if s["id"] == scene_id or s["sceneId"] == scene_id:
                return s
        return None

    def inspect_dataset(self, dataset_id: str) -> bool:
        scene_dir = self.raw_dir / dataset_id
        required = ("ST_B10.tif", "SR_B2.tif", "SR_B3.tif", "SR_B4.tif")
        return scene_dir.is_dir() and all((scene_dir / b).is_file() for b in required)

    def validate_dataset(self, dataset_id: str) -> bool:
        return self.inspect_dataset(dataset_id)

    def run_inference_pipeline(
        self,
        scene_id: str,
        model_type: str = "swinir",
        scale: int = 4,
        sentinel_available: bool = False,
    ) -> PipelineResult:
        if not self.validate_dataset(scene_id):
            raise FileNotFoundError(f"Scene invalid or incomplete: {scene_id}")
        pipe = ImageEnhancementPipeline(
            checkpoint=self.model_weights_path,
            scale=scale,
            output_dir=OUTPUT_DIR,
        )
        return pipe.run(scene_id, sentinel_available=sentinel_available)


class ReportGenerator:
    """Generate JSON evaluation reports."""

    def __init__(self, output_path: Path | str = REPORTS_DIR):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_report(self, result: PipelineResult | dict, dataset: dict | None = None) -> dict:
        data = result if isinstance(result, dict) else {
            "scene_id": result.scene_id,
            "status": result.status,
            "mode": result.mode,
            "evaluation": result.evaluation,
            "paths": {
                "thermal": result.enhanced_thermal_path,
                "colorized": result.colorized_path,
                "geotiff": result.geotiff_path,
                "preview": result.preview_path,
            },
            "logs": result.logs,
            "created_at": result.created_at,
            "dataset": dataset or {},
        }
        scene_id = data.get("scene_id", "unknown")
        path = self.output_path / f"{scene_id}_report.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        data["report_path"] = str(path)
        return data

    def export_results(self, report: dict) -> bool:
        return Path(report.get("report_path", "")).is_file()


def list_outputs() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for scene_dir in sorted(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.is_dir() else []:
        if not scene_dir.is_dir():
            continue
        report = scene_dir / "pipeline_report.json"
        meta: dict[str, Any] = {}
        if report.is_file():
            meta = json.loads(report.read_text(encoding="utf-8"))
        geotiff = scene_dir / "enhanced_stack.tif"
        size = geotiff.stat().st_size if geotiff.is_file() else 0
        items.append(
            {
                "id": f"out-{scene_dir.name}",
                "sceneId": scene_dir.name,
                "title": f"{scene_dir.name} — SR + Colorized",
                "type": "combined",
                "format": "GeoTIFF",
                "resolution": "upscaled",
                "sizeBytes": size,
                "createdAt": meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                "crs": "EPSG:32643",
                "bands": ["RGB colorized", "ST_B10_enhanced"],
                "thumbnailGradient": "from-orange-950 via-red-900/80 to-purple-950",
                "previewPath": meta.get("preview_path", ""),
            }
        )
    return items


def _guess_date(scene_id: str) -> str:
    # LC08_L2SP_PPPRRR_YYYYMMDD_...
    parts = scene_id.split("_")
    for p in parts:
        if len(p) == 8 and p.isdigit():
            return f"{p[:4]}-{p[4:6]}-{p[6:8]}"
    return ""


def _guess_path_row(scene_id: str) -> str:
    parts = scene_id.split("_")
    for p in parts:
        if len(p) == 6 and p.isdigit():
            return f"{p[:3]}/{p[3:]}"
    return ""
