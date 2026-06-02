from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BACKGROUND_ID, NUM_SEMANTIC_CLASSES, TRAIN_ID_TO_NAME
from data import average_metrics, ensure_dir, make_semantic_loaders
from model.UNet.model import ModelAPI


def compute_semantic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    ious: list[float] = []
    for class_id in TRAIN_ID_TO_NAME:
        if class_id == BACKGROUND_ID:
            continue
        pred = preds == class_id
        target = labels == class_id
        union = (pred | target).sum().item()
        if union:
            ious.append(float((pred & target).sum().item() / union))
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": float((preds == labels).sum().item() / max(1, labels.numel())),
    }


def amp_is_enabled(args, device: torch.device) -> bool:
    return bool(getattr(args, "amp", True) and device.type == "cuda")


def autocast_context(device: torch.device, enabled: bool):
    return autocast(device_type=device.type, enabled=enabled)


def make_grad_scaler(enabled: bool) -> GradScaler:
    return GradScaler("cuda", enabled=enabled)


def multiclass_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_SEMANTIC_CLASSES,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    targets = F.one_hot(labels.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=probs.dtype)
    if not include_background:
        probs = probs[:, 1:]
        targets = targets[:, 1:]
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def semantic_loss(logits: torch.Tensor, labels: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, float]]:
    ce_weight = float(getattr(args, "unet_ce_weight", 1.0))
    dice_weight = float(getattr(args, "unet_dice_weight", 0.5))
    ce = F.cross_entropy(logits, labels)
    dice = multiclass_dice_loss(logits, labels)
    loss = ce_weight * ce + dice_weight * dice
    return loss, {"ce_loss": float(ce.detach().cpu()), "dice_loss": float(dice.detach().cpu())}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    ce_losses: list[float] = []
    dice_losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    context = torch.enable_grad() if is_train else torch.inference_mode()
    use_amp = amp_is_enabled(args, device)
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                logits = model(images)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss, loss_parts = semantic_loss(logits, labels, args)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(loss_parts["ce_loss"])
            dice_losses.append(loss_parts["dice_loss"])
            metric_items.append(compute_semantic_metrics(logits.detach(), labels.detach()))
    metrics = average_metrics(metric_items)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["ce_loss"] = float(np.mean(ce_losses)) if ce_losses else 0.0
    metrics["dice_loss"] = float(np.mean(dice_losses)) if dice_losses else 0.0
    return metrics


def train_torch_model(args, architecture: str, train_loader: DataLoader, valid_loader: DataLoader, epoch_runner: Callable, monitor_metric: str, maximize: bool) -> None:
    arch_output_dir = ensure_dir(Path(args.output_dir) / architecture)
    model_api = ModelAPI.create(args.model_name_or_path).prepare_for_training(args.device)
    optimizer = torch.optim.AdamW(model_api.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        if getattr(args, "unet_scheduler", "cosine") == "cosine"
        else None
    )
    scaler = make_grad_scaler(amp_is_enabled(args, model_api.device))
    best_value = -float("inf") if maximize else float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_runner(model_api.module, train_loader, model_api.device, args, optimizer, scaler)
        valid_metrics = epoch_runner(model_api.module, valid_loader, model_api.device, args, None, None)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        model_api.save(arch_output_dir / "last.pt", args.image_size, valid_metrics)
        current = valid_metrics[monitor_metric]
        improved = current > best_value if maximize else current < best_value
        if improved:
            best_value = current
            model_api.save(arch_output_dir / "best.pt", args.image_size, valid_metrics)
        if scheduler is not None:
            scheduler.step()
    (arch_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def train(args) -> None:
    train_loader, valid_loader = make_semantic_loaders(args)
    train_torch_model(args, "unet", train_loader, valid_loader, run_epoch, "mean_iou", True)
