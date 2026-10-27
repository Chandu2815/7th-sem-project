"""CLI: preprocess one Landsat scene."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pipeline import PreprocessConfig, preprocess_scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Landsat C2 L2 scene")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    result = preprocess_scene(
        args.scene_id,
        PreprocessConfig(patch_size=args.patch_size, stride=args.stride),
    )
    print(json.dumps({
        "scene_id": result.scene_id,
        "num_patches": result.num_patches,
        "thermal_path": result.thermal_path,
        "rgb_path": result.rgb_path,
        "output_dir": result.output_dir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
