"""CLI entry: run enhancement pipeline on one scene."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import ImageEnhancementPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ThermaScope enhancement pipeline")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sentinel", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    pipe = ImageEnhancementPipeline(scale=args.scale)
    result = pipe.run(args.scene_id, sentinel_available=args.sentinel)
    print(json.dumps({
        "scene_id": result.scene_id,
        "status": result.status,
        "mode": result.mode,
        "geotiff": result.geotiff_path,
        "preview": result.preview_path,
        "evaluation": result.evaluation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
