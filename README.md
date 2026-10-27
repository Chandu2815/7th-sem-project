# ThermaScope AI — Infrared Enhancement + Object Interpretation

Modular pipeline for **satellite infrared** images (CCTV IR designed for later):

```
Upload → Validate → Resize/Preprocess → SwinIR Enhance → Colorization → Object Interpretation → Compare/Download
```

## Install & run

```bash
pip install -r requirements.txt

# API
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# UI
cd frontend && npm install && npm run dev
```

Open the UI → **IR Enhance** (`/enhance`).

API docs: http://127.0.0.1:8000/docs

## Pipeline modules

| Module | Path |
|--------|------|
| Config | `backend/config.py` |
| ImageSource (SatelliteIR / CCTVIR stub) | `backend/pipeline/sources.py` |
| Validation | `backend/pipeline/validation.py` |
| Preprocess | `backend/pipeline/preprocess.py` |
| SwinIR enhance | `backend/pipeline/enhance.py` → `models/swinir.py` |
| Colorization | `backend/pipeline/colorize.py` → `models/colorization.py` |
| Object detect | `backend/pipeline/detect.py` |
| Results / cleanup | `backend/pipeline/results.py` |
| Orchestrator | `backend/pipeline/runner.py` |
| API | `backend/ir_routes.py` |
| UI | `frontend/src/features/enhance/EnhancePage.tsx` |

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `THERMASCOPE_SWINIR_CHECKPOINT` | `models/checkpoints/thermal-swinir.pt` | SwinIR weights |
| `THERMASCOPE_REQUIRE_SWINIR_WEIGHTS` | `false` | If `true`, missing weights → HTTP 503 |
| `THERMASCOPE_DETECTOR_WEIGHTS` | `models/checkpoints/yolov8n.pt` | YOLO weights |
| `THERMASCOPE_TARGET_SIZE` | `512` | Preprocess letterbox size |
| `THERMASCOPE_SCALE` | `2` | SR scale factor |
| `THERMASCOPE_MAX_UPLOAD_MB` | `50` | Upload limit |
| `THERMASCOPE_JOB_TTL_HOURS` | `24` | Auto-cleanup jobs/uploads |
| `THERMASCOPE_DEVICE` | auto cuda/cpu | Inference device |

## Model weights

1. **SwinIR** (optional for demo): `models/checkpoints/thermal-swinir.pt`  
   Without it the API uses classical denoise+upsample and returns a **warning** (or errors if `THERMASCOPE_REQUIRE_SWINIR_WEIGHTS=true`).
2. **YOLO** (optional): `models/checkpoints/yolov8n.pt` or Ultralytics will fetch `yolov8n.pt` on first detect.  
   Without ultralytics, detection returns an empty list + message; enhancement still works.

## Future CCTV

`ImageSourceKind.CCTVIR` is reserved. Implement `CCTVIRSource.load()` (and optional video frame extraction) without changing SwinIR/colorize/result/UI contracts.

## Landsat scene tools (existing)

Download / preview / GeoTIFF scene pipeline remains under `scripts/` and `models/pipeline.py`.
