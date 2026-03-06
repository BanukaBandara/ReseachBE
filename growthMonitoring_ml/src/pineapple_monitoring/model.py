from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelOutput:
    health_logits: torch.Tensor
    month_logits: torch.Tensor


class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str, num_health: int = 3, num_months: int = 12, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.feature_extractor = base
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            base = models.resnet50(weights=weights)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.feature_extractor = base
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.shared = nn.Sequential(
            nn.Dropout(p=dropout),
        )
        self.health_head = nn.Linear(feat_dim, num_health)
        self.month_head = nn.Linear(feat_dim, num_months)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        feats = self.feature_extractor(x)
        feats = self.shared(feats)
        return ModelOutput(health_logits=self.health_head(feats), month_logits=self.month_head(feats))


def get_default_cam_layer(model: MultiTaskNet) -> nn.Module:
    if model.backbone_name == "efficientnet_b0":
        # Last conv block of EfficientNet features
        return model.feature_extractor.features[-1]
    if model.backbone_name == "resnet50":
        return model.feature_extractor.layer4[-1]
    raise ValueError(f"No default CAM layer for backbone={model.backbone_name}")
