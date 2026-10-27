"""API routes for IR image upload → enhance → colorize → detect pipeline."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import settings
from backend.pipeline.enhance import MissingWeightsError
from backend.pipeline.results import job_dir
from backend.pipeline.runner import get_job, list_recent_jobs, run_ir_pipeline, save_upload
from backend.pipeline.sources import ImageSourceKind
from backend.pipeline.validation import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ir", tags=["ir-pipeline"])

_DOWNLOAD_MAP = {
    "enhanced": "enhanced_ir.png",
    "colorized": "colorized.png",
    "comparison": "comparison.png",
    "annotated": "annotated.png",
    "original": "original.png",
    "preprocessed": "preprocessed.png",
}


@router.post("/enhance")
async def enhance_upload(
    file: UploadFile = File(...),
    source_kind: str = Form(default="SatelliteIR"),
    run_detection: bool = Form(default=True),
):
    """Upload an IR image and run the full enhancement pipeline."""
    try:
        kind = ImageSourceKind(source_kind)
    except ValueError as exc:
        raise HTTPException(400, "Invalid source_kind. Use SatelliteIR (CCTVIR reserved).") from exc

    if kind == ImageSourceKind.CCTV_IR:
        raise HTTPException(501, "CCTV infrared support is not implemented yet. Use SatelliteIR.")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(400, f"File exceeds {settings.max_upload_mb} MB limit.")

    job_id = uuid.uuid4().hex
    try:
        path = save_upload(file.filename or "upload.png", raw)
    except ValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        result = run_ir_pipeline(
            image_path=path,
            source_kind=kind,
            job_id=job_id,
            run_detection=run_detection,
        )
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except MissingWeightsError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("enhance_upload failed")
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc

    return result


@router.get("/jobs")
def jobs(limit: int = 20):
    return list_recent_jobs(limit=limit)


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job:
        return job
    # Fall back to on-disk report
    report = job_dir(job_id) / "report.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    raise HTTPException(404, f"Job not found: {job_id}")


@router.get("/jobs/{job_id}/files/{filename}")
def job_file(job_id: str, filename: str):
    safe = Path(filename).name
    path = job_dir(job_id) / safe
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@router.get("/jobs/{job_id}/download/{kind}")
def job_download(job_id: str, kind: str):
    filename = _DOWNLOAD_MAP.get(kind)
    if not filename:
        raise HTTPException(400, f"Unknown download kind '{kind}'")
    path = job_dir(job_id) / filename
    if not path.is_file():
        raise HTTPException(404, f"{kind} not available for this job")
    return FileResponse(path, filename=f"{job_id}_{filename}", media_type="image/png")


@router.get("/config")
def ir_config():
    return {
        "allowed_extensions": list(settings.allowed_extensions),
        "max_upload_mb": settings.max_upload_mb,
        "target_size": settings.target_size,
        "scale_factor": settings.scale_factor,
        "device": settings.device,
        "swinir_checkpoint": str(settings.swinir_checkpoint),
        "require_swinir_weights": settings.require_swinir_weights,
        "source_kinds": ["SatelliteIR"],
        "future_source_kinds": ["CCTVIR"],
    }
