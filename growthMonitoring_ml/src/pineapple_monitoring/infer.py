from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .constants import IDX_TO_HEALTH, IDX_TO_MONTH
from .data import build_transforms
from .advice import advice_to_dict, generate_farmer_advice
from .gradcam import GradCAM
from .model import MultiTaskNet, get_default_cam_layer
from .utils import ensure_dir, resolve_device, save_json


def _load_rgb_uint8(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _preprocess(image_rgb_uint8: np.ndarray, image_size: int) -> torch.Tensor:
    # Match eval transforms (resize+center crop+normalize)
    tfm = build_transforms(image_size=image_size, train=False)
    pil = Image.fromarray(image_rgb_uint8)
    x = tfm(pil).unsqueeze(0)
    return x


def infer_main(args: argparse.Namespace) -> None:
    device = resolve_device(str(args.device))

    save_dir = ensure_dir(args.save_dir)

    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone = str(args.backbone)
    image_size = int(args.image_size)

    model = MultiTaskNet(backbone=backbone).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rgb = _load_rgb_uint8(args.image)
    x = _preprocess(rgb, image_size=image_size).to(device)

    target_layer = get_default_cam_layer(model)
    cam = GradCAM(model, target_layer)

    with torch.inference_mode():
        out = model(x)
        health_probs = F.softmax(out.health_logits, dim=1)
        month_probs = F.softmax(out.month_logits, dim=1)

        health_idx = int(health_probs.argmax(dim=1).item())
        month_idx = int(month_probs.argmax(dim=1).item())

        health_conf = float(health_probs[0, health_idx].item())
        month_conf = float(month_probs[0, month_idx].item())

    # Grad-CAM needs gradients -> no inference_mode
    out_for_cam = model(x)
    cam_res = cam(
        input_tensor=x,
        class_idx=health_idx,
        original_rgb_uint8=rgb,
        score_tensor=out_for_cam.health_logits,
    )
    cam.close()

    expected_month = args.expected_month
    stunted: bool | None = None
    if expected_month is not None:
        exp_idx = int(expected_month) - 1
        stunted = (month_idx <= exp_idx - int(args.stunted_threshold))

    overlay_path = save_dir / "gradcam_overlay.jpg"
    cv2.imwrite(str(overlay_path), cam_res.overlay_bgr)

    result: dict[str, Any] = {
        "image": str(args.image),
        "health": {
            "label": IDX_TO_HEALTH[health_idx],
            "confidence": health_conf,
            "index": health_idx,
        },
        "growth_stage": {
            "label": IDX_TO_MONTH[month_idx],
            "confidence": month_conf,
            "index": month_idx,
            "month_number": month_idx + 1,
        },
        "stunted_growth": {
            "expected_month": expected_month,
            "threshold": int(args.stunted_threshold),
            "flag": stunted,
        },
        "gradcam_overlay": str(overlay_path),
    }

    advice = generate_farmer_advice(
        health_label=IDX_TO_HEALTH[health_idx],
        health_confidence=health_conf,
        growth_stage_month=month_idx + 1,
        stunted_flag=stunted,
    )
    result["farmer_advice"] = advice_to_dict(advice)

    save_json(save_dir / "prediction.json", result)
    print(result)
