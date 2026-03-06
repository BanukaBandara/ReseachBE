from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import HEALTH_TO_IDX, MONTH_TO_IDX


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    health: int
    month: int


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def scan_dataset(root: str | Path) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples: list[Sample] = []
    for month_dir in sorted(root.iterdir()):
        if not month_dir.is_dir():
            continue
        month_name = month_dir.name
        if month_name not in MONTH_TO_IDX:
            continue
        month_idx = MONTH_TO_IDX[month_name]
        for health_dir in sorted(month_dir.iterdir()):
            if not health_dir.is_dir():
                continue
            health_name = health_dir.name
            if health_name not in HEALTH_TO_IDX:
                continue
            health_idx = HEALTH_TO_IDX[health_name]
            for img_path in health_dir.rglob("*"):
                if _is_image_file(img_path):
                    samples.append(Sample(path=img_path, health=health_idx, month=month_idx))

    if not samples:
        raise RuntimeError(
            "No images found. Expected structure like M1/healthy/*.jpg etc. "
            f"Got root={root}"
        )
    return samples


def build_transforms(image_size: int, train: bool, strong_aug: bool = False) -> transforms.Compose:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if train:
        tfms: list[transforms.Transform] = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.RandomAutocontrast(p=0.15),
        ]

        if strong_aug:
            tfms.append(transforms.RandAugment(num_ops=2, magnitude=9))

        tfms.extend([transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std)])

        if strong_aug:
            tfms.append(
                transforms.RandomErasing(
                    p=0.20, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"
                )
            )

        return transforms.Compose(tfms)

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )


def split_samples(
    samples: list[Sample], seed: int, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    indices = list(range(len(samples)))

    # Prefer stratification to keep class distributions stable across splits.
    # Try joint (health, month) -> health -> month -> random.
    y_joint = [samples[i].health * 12 + samples[i].month for i in indices]
    y_health = [samples[i].health for i in indices]
    y_month = [samples[i].month for i in indices]

    def _safe_split(strat_labels: list[int] | None):
        return train_test_split(
            indices,
            test_size=(1.0 - train_ratio),
            random_state=seed,
            shuffle=True,
            stratify=strat_labels,
        )

    try:
        train_idx, tmp_idx = _safe_split(y_joint)
    except Exception:
        try:
            train_idx, tmp_idx = _safe_split(y_health)
        except Exception:
            try:
                train_idx, tmp_idx = _safe_split(y_month)
            except Exception:
                rnd = random.Random(seed)
                rnd.shuffle(indices)
                n_train = int(len(indices) * train_ratio)
                train_idx, tmp_idx = indices[:n_train], indices[n_train:]

    # Split tmp into val and test
    val_share_of_tmp = val_ratio / (1.0 - train_ratio)
    tmp_labels_joint = [samples[i].health * 12 + samples[i].month for i in tmp_idx]
    tmp_labels_health = [samples[i].health for i in tmp_idx]
    tmp_labels_month = [samples[i].month for i in tmp_idx]

    def _safe_split_tmp(strat_labels: list[int] | None):
        return train_test_split(
            tmp_idx,
            test_size=(1.0 - val_share_of_tmp),
            random_state=seed,
            shuffle=True,
            stratify=strat_labels,
        )

    try:
        val_idx, test_idx = _safe_split_tmp(tmp_labels_joint)
    except Exception:
        try:
            val_idx, test_idx = _safe_split_tmp(tmp_labels_health)
        except Exception:
            try:
                val_idx, test_idx = _safe_split_tmp(tmp_labels_month)
            except Exception:
                rnd = random.Random(seed)
                rnd.shuffle(tmp_idx)
                n_val = int(len(tmp_idx) * val_share_of_tmp)
                val_idx, test_idx = tmp_idx[:n_val], tmp_idx[n_val:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples


class PineappleDataset(Dataset):
    def __init__(self, samples: Iterable[Sample], tfm: transforms.Compose):
        self.samples = list(samples)
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.tfm(img)
        return x, torch.tensor(s.health, dtype=torch.long), torch.tensor(s.month, dtype=torch.long), str(s.path)
