# Architecture overview (PPT-aligned)

## Modules
- DatasetManager (`backend/services.py`) — list/inspect/validate/run
- ImageEnhancementPipeline (`models/pipeline.py`) — preprocess → SwinIR → colorize → export
- ReportGenerator (`backend/services.py`) — JSON reports under `outputs/reports/`

## Actors (stubs)
- User: select dataset, run pipeline, export
- Admin: manage users/system/datasets (auth stub)

## Downstream
- Object interpretation: `POST /api/detect` (YOLOv11 stub)
- LLVIP: `GET /api/llvip` + `dataset/llvip/`
