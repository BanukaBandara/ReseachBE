from __future__ import annotations

import base64
import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# If you keep this repo structure, this makes backend imports work.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from pineapple_monitoring.constants import IDX_TO_HEALTH, IDX_TO_MONTH  # noqa: E402
from pineapple_monitoring.data import build_transforms  # noqa: E402
from pineapple_monitoring.gradcam import GradCAM  # noqa: E402
from pineapple_monitoring.model import MultiTaskNet, get_default_cam_layer  # noqa: E402
from pineapple_monitoring.utils import resolve_device  # noqa: E402


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


DEVICE = resolve_device(_env_str("GROWTH_ML_DEVICE", "auto"))
BACKBONE = _env_str("GROWTH_ML_BACKBONE", "efficientnet_b0")
IMAGE_SIZE = _env_int("GROWTH_ML_IMAGE_SIZE", 224)

DEFAULT_CKPT = Path(__file__).resolve().parent / "pineapple_multitask_best.pt"
CHECKPOINT_PATH = Path(_env_str("GROWTH_ML_CHECKPOINT", str(DEFAULT_CKPT))).expanduser().resolve()
MODEL_VERSION = CHECKPOINT_PATH.name


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _transform

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    _model = _load_model()
    _transform = build_transforms(image_size=IMAGE_SIZE, train=False)
    yield


app = FastAPI(title="Growth Monitoring ML", version=MODEL_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

_model: MultiTaskNet | None = None
_transform = None


def _load_model() -> MultiTaskNet:
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    model = MultiTaskNet(backbone=BACKBONE).to(DEVICE)

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": str(DEVICE),
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
        "checkpoint": str(CHECKPOINT_PATH),
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    global _model, _transform

    if _model is None or _transform is None:
        return JSONResponse(status_code=503, content={"success": False, "message": "Model not loaded"})

    try:
        raw = await file.read()
        if not raw:
            return JSONResponse(status_code=400, content={"success": False, "message": "Empty file"})

        from PIL import Image
        import numpy as np
        import cv2

        pil = Image.open(BytesIO(raw)).convert("RGB")
        rgb_uint8 = np.array(pil, dtype=np.uint8)

        x = _transform(pil).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            out = _model(x)
            health_probs = F.softmax(out.health_logits, dim=1)
            month_probs = F.softmax(out.month_logits, dim=1)

            health_idx = int(health_probs.argmax(dim=1).item())
            month_idx = int(month_probs.argmax(dim=1).item())

            health_conf = float(health_probs[0, health_idx].item())
            month_conf = float(month_probs[0, month_idx].item())

            health_prob_map = {IDX_TO_HEALTH[i]: float(health_probs[0, i].item()) for i in range(health_probs.size(1))}
            month_prob_map = {str(IDX_TO_MONTH[i]): float(month_probs[0, i].item()) for i in range(month_probs.size(1))}

        # Grad-CAM needs gradients -> no inference_mode
        target_layer = get_default_cam_layer(_model)
        cam = GradCAM(_model, target_layer)
        try:
            out_for_cam = _model(x)
            cam_res = cam(
                input_tensor=x,
                class_idx=health_idx,
                original_rgb_uint8=rgb_uint8,
                score_tensor=out_for_cam.health_logits,
            )
        finally:
            cam.close()

        ok, buf = cv2.imencode(".png", cam_res.overlay_bgr)
        overlay_b64 = base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None

        # Keep the response compatible with the existing Node controller.
        payload: dict[str, Any] = {
            "success": True,
            "condition": IDX_TO_HEALTH[health_idx],
            "condition_confidence": round(health_conf * 100.0, 2),
            "month": int(month_idx + 1),
            "predicted_class": IDX_TO_HEALTH[health_idx],
            "confidence": round(health_conf * 100.0, 2),
            "probabilities": {
                "health": health_prob_map,
                "growth_stage": month_prob_map,
                "health_confidence": health_conf,
                "growth_stage_confidence": month_conf,
            },
            "model_version": MODEL_VERSION,
            "gradcam_overlay_png_base64": overlay_b64,
        }

        return JSONResponse(content=payload)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Prediction failed", "details": str(exc)},
        )


def main() -> None:
    import uvicorn

    host = _env_str("GROWTH_ML_HOST", "127.0.0.1")
    port = _env_int("GROWTH_ML_PORT", 8001)
    uvicorn.run("growth_api:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
