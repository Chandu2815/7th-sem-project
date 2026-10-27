"""Training stub for SwinIR fine-tuning on thermal patches."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    dataset_dir: str = str(PROJECT_ROOT / "dataset" / "Processed")
    checkpoint_dir: str = str(PROJECT_ROOT / "models" / "checkpoints")
    scale: int = 4
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-4
    patch_size: int = 128
    device: str = "cuda"
    model_name: str = "thermal-swinir-v0.1"


def run_training_stub(config: TrainConfig | None = None) -> dict:
    """Record a training job config without running a full GPU loop.

    Replace this stub with a real DataLoader + AdamW loop when checkpoints
    and labeled HR patches are ready.
    """
    config = config or TrainConfig()
    out = Path(config.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "stub_registered",
        "message": "Training loop not executed — config saved for Phase-2 full train.",
        "config": asdict(config),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    path = out / "last_train_config.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    logger.info("Training stub registered → %s", path)
    return status
