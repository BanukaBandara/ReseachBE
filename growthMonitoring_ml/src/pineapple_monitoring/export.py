from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import MultiTaskNet
from .utils import ensure_dir


class ExportWrapper(torch.nn.Module):
    """Wrap ModelOutput dataclass into a tuple for ONNX/TorchScript friendliness."""

    def __init__(self, model: MultiTaskNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out.health_logits, out.month_logits


def export_main(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model = MultiTaskNet(backbone=str(args.backbone))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    wrapper = ExportWrapper(model)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    dummy = torch.randn(1, 3, int(args.image_size), int(args.image_size))

    if args.format == "torchscript":
        traced = torch.jit.trace(wrapper, dummy)
        try:
            traced = torch.jit.optimize_for_inference(traced)
        except Exception:
            pass
        traced.save(str(out_path))
        return

    if args.format == "onnx":
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=["health_logits", "month_logits"],
            dynamic_axes={"image": {0: "batch"}, "health_logits": {0: "batch"}, "month_logits": {0: "batch"}},
            opset_version=int(args.opset),
        )
        return

    raise ValueError(f"Unknown export format: {args.format}")
