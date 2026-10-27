"""Orchestrates: Upload → Validate → Preprocess → SwinIR → Colorize → Detect → Results."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.pipeline.colorize import colorize_ir
from backend.pipeline.detect import detect_objects
from backend.pipeline.enhance import MissingWeightsError, enhance_ir
from backend.pipeline.preprocess import preprocess_ir
from backend.pipeline.results import (
    JobArtifacts,
    cleanup_old_jobs,
    cleanup_uploads,
    job_dir,
    make_comparison,
    save_display_original,
    save_gray,
    save_rgb,
    write_report,
)
from backend.pipeline.sources import ImageSourceKind, create_image_source
from backend.pipeline.validation import ValidationError, validate_upload_filename

logger = logging.getLogger(__name__)

# In-memory job status for UI progress polling
_JOBS: dict[str, dict[str, Any]] = {}


def get_job(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)


def list_recent_jobs(limit: int = 20) -> list[dict[str, Any]]:
    items = sorted(_JOBS.values(), key=lambda j: j.get("created_at", ""), reverse=True)
    return items[:limit]


def _set_job(job_id: str, **kwargs: Any) -> dict[str, Any]:
    job = _JOBS.setdefault(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "error": None,
            "warnings": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    job.update(kwargs)
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    return job


def save_upload(filename: str, data: bytes) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe = validate_upload_filename(filename)
    dest = settings.upload_dir / f"{uuid.uuid4().hex}_{safe}"
    dest.write_bytes(data)
    return dest


def run_ir_pipeline(
    image_path: Path,
    source_kind: ImageSourceKind = ImageSourceKind.SATELLITE_IR,
    job_id: str | None = None,
    run_detection: bool = True,
) -> dict[str, Any]:
    """Execute the full IR enhancement + interpretation pipeline."""
    cleanup_old_jobs()
    cleanup_uploads()

    job_id = job_id or uuid.uuid4().hex
    _set_job(job_id, status="running", stage="validate", progress=5, source_kind=source_kind.value)

    try:
        source = create_image_source(Path(image_path), source_kind)
        _set_job(job_id, stage="load", progress=10)
        loaded = source.load()

        _set_job(job_id, stage="preprocess", progress=25)
        pre = preprocess_ir(loaded.thermal, loaded.original_rgb_or_gray)

        _set_job(job_id, stage="enhance", progress=45)
        enh = enhance_ir(pre.thermal_model)

        _set_job(job_id, stage="colorize", progress=65)
        enhanced_ir = enh.enhanced
        col = colorize_ir(enhanced_ir, guidance_rgb=loaded.guidance_rgb)

        detections_payload: list[dict[str, Any]] = []
        annotated = None
        detect_msg = ""
        detect_backend = "skipped"
        if run_detection:
            _set_job(job_id, stage="detect", progress=80)
            det = detect_objects(col.colorized)
            detections_payload = [
                {"label": d.label, "confidence": d.confidence, "bbox": d.bbox} for d in det.detections
            ]
            annotated = det.annotated
            detect_msg = det.message
            detect_backend = det.backend

        _set_job(job_id, stage="results", progress=90)
        out = job_dir(job_id)
        original_path = save_display_original(out / "original.png", pre.original)
        pre_path = save_gray(out / "preprocessed.png", pre.preprocessed)
        enh_path = save_gray(out / "enhanced_ir.png", enhanced_ir)
        color_path = save_rgb(out / "colorized.png", col.colorized)
        comparison_path = make_comparison(
            pre.original, pre.preprocessed, enhanced_ir, col.colorized, out / "comparison.png"
        )
        annotated_path = ""
        if annotated is not None:
            annotated_path = str(save_rgb(out / "annotated.png", annotated))

        warnings = [w for w in [enh.warning, detect_msg if detect_backend == "stub" else None] if w]

        artifacts = JobArtifacts(
            job_id=job_id,
            original_path=str(original_path),
            preprocessed_path=str(pre_path),
            enhanced_path=str(enh_path),
            colorized_path=str(color_path),
            comparison_path=str(comparison_path),
            annotated_path=annotated_path,
            detections=detections_payload,
            warnings=warnings,
            meta={
                "source_kind": source_kind.value,
                "enhance_backend": enh.backend,
                "colorize_mode": col.mode,
                "detect_backend": detect_backend,
                "preprocess": pre.meta,
                "input": loaded.meta,
                "device": settings.device,
            },
        )
        report = {
            "job_id": job_id,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "original": f"/api/ir/jobs/{job_id}/files/original.png",
                "preprocessed": f"/api/ir/jobs/{job_id}/files/preprocessed.png",
                "enhanced": f"/api/ir/jobs/{job_id}/files/enhanced_ir.png",
                "colorized": f"/api/ir/jobs/{job_id}/files/colorized.png",
                "comparison": f"/api/ir/jobs/{job_id}/files/comparison.png",
                "annotated": f"/api/ir/jobs/{job_id}/files/annotated.png" if annotated_path else None,
            },
            "downloads": {
                "enhanced": f"/api/ir/jobs/{job_id}/download/enhanced",
                "colorized": f"/api/ir/jobs/{job_id}/download/colorized",
                "comparison": f"/api/ir/jobs/{job_id}/download/comparison",
            },
            "detections": detections_payload,
            "warnings": warnings,
            "meta": artifacts.meta,
            "detect_message": detect_msg,
        }
        report_path = write_report(out / "report.json", report)
        artifacts.report_path = str(report_path)

        result = {
            **report,
            "local_paths": {
                "original": artifacts.original_path,
                "preprocessed": artifacts.preprocessed_path,
                "enhanced": artifacts.enhanced_path,
                "colorized": artifacts.colorized_path,
                "comparison": artifacts.comparison_path,
                "annotated": artifacts.annotated_path,
                "report": artifacts.report_path,
            },
        }
        _set_job(job_id, status="completed", stage="done", progress=100, result=result, warnings=warnings)
        logger.info("IR pipeline completed job=%s", job_id)
        return result

    except NotImplementedError as exc:
        _set_job(job_id, status="failed", stage="error", progress=100, error=str(exc))
        raise
    except (ValidationError, MissingWeightsError) as exc:
        _set_job(job_id, status="failed", stage="error", progress=100, error=str(exc))
        raise
    except Exception as exc:
        logger.exception("IR pipeline failed job=%s", job_id)
        _set_job(job_id, status="failed", stage="error", progress=100, error=str(exc))
        raise
