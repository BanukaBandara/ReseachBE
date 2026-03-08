from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class CamResult:
    heatmap: np.ndarray  # HxW in [0,1]
    overlay_bgr: np.ndarray  # HxWx3 uint8


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_acts)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_grads)

    def close(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def _save_acts(self, module: nn.Module, inp, out) -> None:
        self._acts = out

    def _save_grads(self, module: nn.Module, grad_input, grad_output) -> None:
        self._grads = grad_output[0]

    @torch.no_grad()
    def _normalize(self, cam: np.ndarray) -> np.ndarray:
        cam = np.maximum(cam, 0)
        denom = cam.max() - cam.min()
        if denom < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)
        cam = (cam - cam.min()) / denom
        return cam.astype(np.float32)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        original_rgb_uint8: np.ndarray,
        score_tensor: torch.Tensor,
    ) -> CamResult:
        """Compute Grad-CAM for `class_idx`.

        - input_tensor: 1x3xHxW normalized tensor
        - original_rgb_uint8: HxWx3 RGB uint8 image (for overlay)
        - score_tensor: logits tensor (1xC) used for backprop target
        """
        self.model.zero_grad(set_to_none=True)

        # Backprop through the selected score
        score = score_tensor[:, class_idx].sum()
        score.backward(retain_graph=True)

        acts = self._acts
        grads = self._grads
        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        # Global-average-pool gradients -> weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # 1xCx1x1
        cam = (weights * acts).sum(dim=1, keepdim=False)  # 1xHxW
        cam_np = cam.squeeze(0).detach().cpu().numpy()
        cam_np = self._normalize(cam_np)

        h, w = original_rgb_uint8.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_color = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.cvtColor(original_rgb_uint8, cv2.COLOR_RGB2BGR), 0.55, heatmap_color, 0.45, 0
        )
        return CamResult(heatmap=cam_resized, overlay_bgr=overlay)
