from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# If you keep this repo structure, this makes backend imports work.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from pineapple_monitoring.infer import infer_main  # noqa: E402


@dataclass
class Prediction:
    health_label: str
    health_confidence: float
    growth_stage_label: str
    growth_stage_month_number: int
    growth_stage_confidence: float
    stunted_flag: bool | None
    gradcam_overlay_path: str
    raw: dict[str, Any]


def predict(
    checkpoint: str,
    image: str,
    save_dir: str = "outputs",
    backbone: str = "efficientnet_b0",
    image_size: int = 224,
    device: str = "auto",
    expected_month: int | None = None,
    stunted_threshold: int = 1,
) -> Prediction:
    args = argparse.Namespace(
        checkpoint=checkpoint,
        image=image,
        backbone=backbone,
        image_size=image_size,
        device=device,
        expected_month=expected_month,
        stunted_threshold=stunted_threshold,
        save_dir=save_dir,
    )
    infer_main(args)

    pred_path = Path(save_dir) / "prediction.json"
    payload = json.loads(pred_path.read_text(encoding="utf-8"))

    stunted = payload.get("stunted_growth", {}).get("flag", None)

    return Prediction(
        health_label=payload["health"]["label"],
        health_confidence=float(payload["health"]["confidence"]),
        growth_stage_label=payload["growth_stage"]["label"],
        growth_stage_month_number=int(payload["growth_stage"]["month_number"]),
        growth_stage_confidence=float(payload["growth_stage"]["confidence"]),
        stunted_flag=stunted,
        gradcam_overlay_path=str(payload["gradcam_overlay"]),
        raw=payload,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict pineapple health + growth stage")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--expected_month", type=int, default=None)
    p.add_argument("--stunted_threshold", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = predict(
        checkpoint=args.checkpoint,
        image=args.image,
        save_dir=args.save_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        device=args.device,
        expected_month=args.expected_month,
        stunted_threshold=args.stunted_threshold,
    )
    print(pred)


if __name__ == "__main__":
    _ = torch.cuda.is_available()
    main()
