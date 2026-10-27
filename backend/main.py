"""FastAPI application — ThermaScope AI backend."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ir_routes import router as ir_router
from backend.services import DatasetManager, ReportGenerator, list_outputs
from backend.stubs import (
    detect_objects_stub,
    get_default_settings,
    get_pipeline_nodes,
    get_system_health,
    get_training_metrics,
    list_llvip_stub,
    login_stub,
)
from training.train import TrainConfig, run_training_stub

app = FastAPI(title="ThermaScope AI", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ir_router)

dm = DatasetManager()
reports = ReportGenerator()


class PipelineRunRequest(BaseModel):
    scene_id: str
    scale: int = 4
    sentinel_available: bool = False
    model_type: str = "swinir"


class LoginRequest(BaseModel):
    username: str
    password: str = ""
    role: str = "user"


class DetectRequest(BaseModel):
    image_path: str = ""
    scene_id: str = ""


@app.get("/api/health")
def health():
    return {"status": "ok", "components": get_system_health()}


@app.get("/api/dashboard")
def dashboard():
    scenes = dm.list_datasets()
    processed = sum(1 for s in scenes if s["status"] == "processed")
    storage = sum(s.get("downloadSizeBytes", 0) for s in scenes)
    return {
        "stats": {
            "downloadedScenes": len(scenes),
            "processedScenes": processed,
            "storageUsedBytes": storage,
            "storageTotalBytes": 500_000_000_000,
            "trainingStatus": "idle",
            "currentModel": "SwinIR Thermal SR",
            "modelVersion": "thermal-swinir-v0.1",
            "gpuAvailable": False,
            "gpuName": "CPU fallback",
            "lastProcessedAt": "",
            "pipelineStages": [
                {"id": "download", "name": "Scene Acquisition", "status": "completed"},
                {"id": "preprocess", "name": "Preprocessing", "status": "waiting"},
                {"id": "inference", "name": "Model Inference", "status": "waiting"},
                {"id": "export", "name": "GeoTIFF Export", "status": "waiting"},
            ],
        },
        "activities": [
            {
                "id": "act-local",
                "type": "system",
                "message": f"Catalog indexed — {len(scenes)} local scene(s)",
                "timestamp": "",
                "status": "info",
            }
        ],
        "health": get_system_health(),
    }


@app.get("/api/scenes")
def scenes(search: str = "", satellite: str = "all", status: str = "all", cloudMax: float = 100):
    result = dm.list_datasets()
    if search:
        q = search.lower()
        result = [s for s in result if q in s["sceneId"].lower()]
    if satellite != "all":
        result = [s for s in result if s["satellite"] == satellite]
    if status != "all":
        result = [s for s in result if s["status"] == status]
    return result


@app.get("/api/scenes/{scene_id}")
def scene_detail(scene_id: str):
    meta = dm.get_scene_metadata(scene_id)
    if not meta:
        # also try matching by id field
        raise HTTPException(404, f"Scene not found: {scene_id}")
    return meta


@app.post("/api/scenes/{scene_id}/inspect")
def inspect(scene_id: str):
    ok = dm.inspect_dataset(scene_id)
    return {"scene_id": scene_id, "valid": ok}


@app.post("/api/pipeline/run")
def run_pipeline(body: PipelineRunRequest):
    try:
        result = dm.run_inference_pipeline(
            scene_id=body.scene_id,
            model_type=body.model_type,
            scale=body.scale,
            sentinel_available=body.sentinel_available,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc
    report = reports.generate_report(result)
    return {"result": report}


@app.get("/api/preprocessing/{scene_id}")
def preprocessing_status(scene_id: str):
    report = PROJECT_ROOT / "dataset" / "Processed" / scene_id / "preprocess_report.json"
    if not report.is_file():
        return {
            "steps": [
                {"id": "raw-scene", "name": "Raw Scene", "description": "Awaiting preprocess", "status": "waiting", "progress": 0},
            ],
            "logs": [],
        }
    import json

    data = json.loads(report.read_text(encoding="utf-8"))
    steps = data.get("steps", [])
    # normalize shape for frontend
    normalized = []
    for s in steps:
        normalized.append(
            {
                "id": s.get("id", ""),
                "name": s.get("name", ""),
                "description": s.get("name", ""),
                "status": s.get("status", "completed"),
                "progress": s.get("progress", 100),
            }
        )
    return {"steps": normalized, "logs": data.get("logs", [])}


@app.get("/api/pipeline/nodes")
def pipeline_nodes():
    return get_pipeline_nodes()


@app.get("/api/training")
def training():
    return get_training_metrics()


@app.post("/api/training/start")
def training_start():
    return run_training_stub(TrainConfig())


@app.get("/api/evaluation")
def evaluation(scene_id: str | None = None):
    items = []
    out_root = PROJECT_ROOT / "outputs" / "enhanced"
    if out_root.is_dir():
        for d in out_root.iterdir():
            report = d / "pipeline_report.json"
            if not report.is_file():
                continue
            import json

            data = json.loads(report.read_text(encoding="utf-8"))
            ev = data.get("evaluation", {})
            if scene_id and data.get("scene_id") != scene_id:
                continue
            items.append(
                {
                    "psnr": ev.get("psnr", 0),
                    "ssim": ev.get("ssim", 0),
                    "lpips": ev.get("lpips", 0),
                    "rmse": ev.get("rmse", 0),
                    "sceneId": data.get("scene_id", d.name),
                    "modelVersion": ev.get("model_version", "thermal-swinir-v0.1"),
                    "evaluatedAt": data.get("created_at", ""),
                }
            )
    return items


@app.get("/api/outputs")
def outputs():
    return list_outputs()


@app.get("/api/exports")
def exports():
    outs = list_outputs()
    return [
        {
            "id": o["id"],
            "filename": Path(o.get("previewPath") or f"{o['sceneId']}.tif").name or f"{o['sceneId']}_enhanced.tif",
            "sceneId": o["sceneId"],
            "format": o["format"],
            "sizeBytes": o["sizeBytes"],
            "exportedAt": o["createdAt"],
            "status": "completed",
        }
        for o in outs
    ]


@app.get("/api/settings")
def settings():
    return get_default_settings()


@app.post("/api/auth/login")
def login(body: LoginRequest):
    return login_stub(body.username, body.password, body.role)


@app.post("/api/detect")
def detect(body: DetectRequest):
    return detect_objects_stub(body.image_path or body.scene_id)


@app.get("/api/llvip")
def llvip():
    return list_llvip_stub()


@app.get("/")
def root():
    return {"name": "ThermaScope AI", "docs": "/docs"}
