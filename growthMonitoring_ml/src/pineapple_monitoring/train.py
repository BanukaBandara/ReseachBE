from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext

try:
    # Newer PyTorch AMP API
    from torch.amp import GradScaler, autocast  # type: ignore

    _HAS_TORCH_AMP = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

    _HAS_TORCH_AMP = False
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .constants import HEALTH_CLASSES, MONTH_CLASSES
from .data import PineappleDataset, scan_dataset, split_samples, build_transforms
from .metrics import compute_metrics
from .model import MultiTaskNet
from .utils import EarlyStopping, ensure_dir, resolve_device, save_json, set_seed


def _class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def _make_sampler(health_labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(health_labels)
    weights = torch.tensor([1.0 / counts[int(y)] for y in health_labels], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _make_joint_sampler(health_labels: list[int], month_labels: list[int]) -> WeightedRandomSampler:
    if len(health_labels) != len(month_labels):
        raise ValueError("health_labels and month_labels must have same length")
    keys = list(zip(health_labels, month_labels))
    counts = Counter(keys)
    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def _run_epoch_eval(
    model: MultiTaskNet,
    loader: DataLoader,
    device: torch.device,
    loss_health_fn: nn.Module,
    loss_month_fn: nn.Module,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0

    y_true_h: list[int] = []
    y_pred_h: list[int] = []
    y_true_m: list[int] = []
    y_pred_m: list[int] = []

    for x, y_h, y_m, _paths in loader:
        x = x.to(device, non_blocking=True)
        y_h = y_h.to(device, non_blocking=True)
        y_m = y_m.to(device, non_blocking=True)

        out = model(x)
        loss_h = loss_health_fn(out.health_logits, y_h)
        loss_m = loss_month_fn(out.month_logits, y_m)
        loss = alpha * loss_h + beta * loss_m

        total_loss += float(loss.item()) * x.size(0)

        y_true_h.extend(y_h.detach().cpu().tolist())
        y_pred_h.extend(out.health_logits.argmax(dim=1).detach().cpu().tolist())
        y_true_m.extend(y_m.detach().cpu().tolist())
        y_pred_m.extend(out.month_logits.argmax(dim=1).detach().cpu().tolist())

    total_loss /= max(1, len(loader.dataset))

    mh = compute_metrics(y_true_h, y_pred_h)
    mm = compute_metrics(y_true_m, y_pred_m)

    return {
        "loss": total_loss,
        "health": {"accuracy": mh.accuracy, "f1_macro": mh.f1_macro, "confusion": mh.conf_mat},
        "month": {"accuracy": mm.accuracy, "f1_macro": mm.f1_macro, "confusion": mm.conf_mat},
    }


def train_main(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    device = resolve_device(str(args.device))
    use_amp = device.type == "cuda"

    data_root = Path(args.data_root)
    samples = scan_dataset(data_root)
    train_s, val_s, test_s = split_samples(samples, seed=int(args.seed))

    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(args.checkpoint_dir)

    strong_aug = bool(getattr(args, "strong_aug", False))
    train_tfm = build_transforms(int(args.image_size), train=True, strong_aug=strong_aug)
    eval_tfm = build_transforms(int(args.image_size), train=False)

    train_ds = PineappleDataset(train_s, train_tfm)
    val_ds = PineappleDataset(val_s, eval_tfm)
    test_ds = PineappleDataset(test_s, eval_tfm)

    train_health = [s.health for s in train_s]
    train_month = [s.month for s in train_s]

    health_w = _class_weights(train_health, num_classes=len(HEALTH_CLASSES)).to(device)
    month_w = _class_weights(train_month, num_classes=len(MONTH_CLASSES)).to(device)

    sampler_mode = str(getattr(args, "sampler", "health") or "health").lower()
    if sampler_mode == "joint":
        sampler = _make_joint_sampler(train_health, train_month)
    elif sampler_mode == "health":
        sampler = _make_sampler(train_health)
    elif sampler_mode == "none":
        sampler = None
    else:
        raise ValueError("--sampler must be one of: joint, health, none")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = MultiTaskNet(backbone=str(args.backbone)).to(device)

    freeze_epochs = int(getattr(args, "freeze_epochs", 0) or 0)
    if freeze_epochs > 0:
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    loss_health_fn = nn.CrossEntropyLoss(weight=health_w, label_smoothing=label_smoothing)
    loss_month_fn = nn.CrossEntropyLoss(weight=month_w, label_smoothing=label_smoothing)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    early = EarlyStopping(patience=int(args.patience), min_delta=0.0)

    if _HAS_TORCH_AMP:
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler(enabled=use_amp)

    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    best_val = float("inf")

    alpha = float(args.alpha)
    beta = float(args.beta)

    run_meta = {
        "data_root": str(data_root),
        "backbone": str(args.backbone),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "alpha": alpha,
        "beta": beta,
        "device": str(device),
        "use_amp": bool(use_amp),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
    }
    save_json(out_dir / "run_meta.json", run_meta)

    for epoch in range(1, int(args.epochs) + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in model.feature_extractor.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        total = 0
        running = 0.0

        for x, y_h, y_m, _paths in pbar:
            x = x.to(device, non_blocking=True)
            y_h = y_h.to(device, non_blocking=True)
            y_m = y_m.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and _HAS_TORCH_AMP:
                amp_ctx = autocast(device_type="cuda", enabled=True)
            elif use_amp:
                amp_ctx = autocast(enabled=True)
            else:
                amp_ctx = nullcontext()

            with amp_ctx:
                out = model(x)
                loss_h = loss_health_fn(out.health_logits, y_h)
                loss_m = loss_month_fn(out.month_logits, y_m)
                loss = alpha * loss_h + beta * loss_m

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += x.size(0)
            running += float(loss.item()) * x.size(0)
            pbar.set_postfix({"loss": running / max(1, total), "lr": optimizer.param_groups[0]["lr"]})

        scheduler.step()

        val_metrics = _run_epoch_eval(
            model,
            val_loader,
            device,
            loss_health_fn=loss_health_fn,
            loss_month_fn=loss_month_fn,
            alpha=alpha,
            beta=beta,
        )
        save_json(out_dir / f"val_epoch_{epoch:03d}.json", val_metrics)

        torch.save(
            {
                "model": model.state_dict(),
                "backbone": str(args.backbone),
                "image_size": int(args.image_size),
                "epoch": epoch,
                "val": val_metrics,
                "run_meta": run_meta,
            },
            last_path,
        )

        val_loss = float(val_metrics["loss"])
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": str(args.backbone),
                    "image_size": int(args.image_size),
                    "epoch": epoch,
                    "val": val_metrics,
                    "run_meta": run_meta,
                },
                best_path,
            )

        if early.step(val_loss):
            break

    # Final test on best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = _run_epoch_eval(
        model,
        test_loader,
        device,
        loss_health_fn=loss_health_fn,
        loss_month_fn=loss_month_fn,
        alpha=alpha,
        beta=beta,
    )
    save_json(out_dir / "test_metrics.json", test_metrics)
