"""Phase-2 stubs: auth roles, YOLO interpretation, LLVIP loader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLVIP_DIR = PROJECT_ROOT / "dataset" / "llvip"


def login_stub(username: str, password: str = "", role: str = "user") -> dict[str, Any]:
    return {
        "success": True,
        "token": f"stub-token-{role}-{username}",
        "user": {
            "username": username,
            "role": role,
            "permissions": (
                ["manageUsers", "manageSystem", "manageDatasets", "viewReports"]
                if role == "admin"
                else ["selectDataset", "runPipeline", "exportResults"]
            ),
        },
        "message": "Auth stub — replace with real credentials in Phase 2",
    }


def detect_objects_stub(source: str = "") -> dict[str, Any]:
    return {
        "status": "stub",
        "model": "YOLOv11",
        "source": source,
        "detections": [
            {"label": "building", "confidence": 0.0, "bbox": [0, 0, 0, 0]},
            {"label": "road", "confidence": 0.0, "bbox": [0, 0, 0, 0]},
        ],
        "segmentation": {"available": False},
        "message": "Object interpretation not executed — wire Ultralytics YOLOv11 here",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def list_llvip_stub() -> dict[str, Any]:
    LLVIP_DIR.mkdir(parents=True, exist_ok=True)
    pairs = list(LLVIP_DIR.glob("*")) if LLVIP_DIR.is_dir() else []
    return {
        "dataset": "LLVIP",
        "path": str(LLVIP_DIR),
        "num_items": len(pairs),
        "paired": True,
        "message": "Place LLVIP infrared/visible pairs under dataset/llvip/",
        "url": "https://github.com/bupt-ai-cz/LLVIP",
    }


def get_system_health() -> list[dict[str, Any]]:
    return [
        {"component": "Planetary Computer STAC", "status": "healthy", "latencyMs": 0, "message": "Client available"},
        {"component": "Local Storage", "status": "healthy", "message": str(PROJECT_ROOT / "dataset")},
        {"component": "SwinIR Module", "status": "healthy", "message": "SwinIRLite ready"},
        {"component": "Colorization Network", "status": "healthy", "message": "Adaptive guided/self-guided"},
        {"component": "Object Interpretation", "status": "degraded", "message": "YOLOv11 stub only"},
    ]


def get_pipeline_nodes() -> list[dict[str, Any]]:
    return [
        {
            "id": "thermal-input",
            "label": "Thermal Image",
            "purpose": "Ingest preprocessed ST_B10 brightness temperature",
            "input": "ST_B10 GeoTIFF",
            "output": "Normalized thermal tensor",
            "model": "—",
            "parameters": {"bands": 1, "dtype": "float32"},
            "status": "completed",
        },
        {
            "id": "swinir",
            "label": "SwinIR Super Resolution",
            "purpose": "SR + denoise via SwinIR-style residual network",
            "input": "Thermal tensor",
            "output": "Enhanced thermal",
            "model": "SwinIRLite",
            "parameters": {"scale": 4, "embed_dim": 64},
            "status": "completed",
        },
        {
            "id": "color-guidance",
            "label": "Color Guidance Selection",
            "purpose": "Choose Sentinel/Landsat RGB guided vs self-guided path",
            "input": "Scene metadata",
            "output": "Guidance mode flag",
            "model": "—",
            "parameters": {"sentinel_window_days": 5},
            "status": "completed",
        },
        {
            "id": "decision",
            "label": "Sentinel RGB Available?",
            "purpose": "Branch selection",
            "input": "Guidance mode flag",
            "output": "Pipeline branch",
            "model": "—",
            "parameters": {},
            "status": "completed",
        },
        {
            "id": "cross-modal",
            "label": "Cross-Modal Guided Colorization",
            "purpose": "Fuse thermal with RGB guidance",
            "input": "Enhanced thermal + RGB",
            "output": "Pseudo-RGB",
            "model": "AdaptiveColorizer",
            "parameters": {"guidance_weight": 0.7},
            "status": "waiting",
        },
        {
            "id": "self-guided",
            "label": "Self-Guided Thermal Colorization",
            "purpose": "Colorize from thermal gradients without external RGB",
            "input": "Enhanced thermal",
            "output": "Pseudo-RGB",
            "model": "AdaptiveColorizer",
            "parameters": {"physics_lambda": 0.15},
            "status": "waiting",
        },
        {
            "id": "physics-refine",
            "label": "Physics-Aware Refinement",
            "purpose": "Thermodynamic brightness consistency",
            "input": "RGB + thermal",
            "output": "Refined stack",
            "model": "PhysRefine",
            "parameters": {"lambda_phys": 0.15},
            "status": "waiting",
        },
        {
            "id": "geotiff-output",
            "label": "GeoTIFF Output",
            "purpose": "Export enhanced GeoTIFF + preview PNG",
            "input": "Refined stack",
            "output": "GeoTIFF + JSON",
            "model": "—",
            "parameters": {"tiled": True},
            "status": "waiting",
        },
        {
            "id": "object-interp",
            "label": "Object Interpretation",
            "purpose": "YOLOv11 / SegFormer stub for PPT downstream CV",
            "input": "Colorized RGB",
            "output": "Detections",
            "model": "YOLOv11 (stub)",
            "parameters": {},
            "status": "waiting",
        },
    ]


def get_training_metrics() -> dict[str, Any]:
    return {
        "epoch": 0,
        "totalEpochs": 200,
        "trainLoss": 0.0,
        "valLoss": 0.0,
        "learningRate": 0.0001,
        "gpuUsage": 0,
        "gpuMemoryUsed": 0,
        "gpuMemoryTotal": 0,
        "modelVersion": "thermal-swinir-v0.1",
        "datasetName": "Landsat C2 L2",
        "checkpoint": "models/checkpoints/thermal-swinir.pt",
        "batchSize": 8,
        "samplesProcessed": 0,
        "totalSamples": 0,
        "eta": "—",
        "bestValLoss": 0.0,
        "stepsPerEpoch": 0,
        "currentStep": 0,
    }


def get_default_settings() -> dict[str, Any]:
    return {
        "datasetPath": str(PROJECT_ROOT / "dataset"),
        "modelCheckpoint": str(PROJECT_ROOT / "models" / "checkpoints" / "thermal-swinir.pt"),
        "modelSelection": "thermal-swinir-v0.1",
        "gpuDevice": "cpu",
        "patchSize": 128,
        "scaleFactor": 4,
        "batchSize": 8,
        "numWorkers": 2,
        "outputFormat": "GeoTIFF",
        "theme": "dark",
    }
