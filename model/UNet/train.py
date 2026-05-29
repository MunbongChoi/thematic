from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BACKGROUND_ID, TRAIN_ID_TO_NAME
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            metric_items.append(compute_semantic_metrics(logits.detach(), labels.detach()))
    metrics = average_metrics(metric_items)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_torch_model(args, architecture: str, train_loader: DataLoader, valid_loader: DataLoader, epoch_runner: Callable, monitor_metric: str, maximize: bool) -> None:
    arch_output_dir = ensure_dir(Path(args.output_dir) / architecture)
    model_api = ModelAPI.create(args.model_name_or_path).prepare_for_training(args.device)
    optimizer = torch.optim.AdamW(model_api.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = -float("inf") if maximize else float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_runner(model_api.module, train_loader, model_api.device, optimizer)
        valid_metrics = epoch_runner(model_api.module, valid_loader, model_api.device, None)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        model_api.save(arch_output_dir / "last.pt", args.image_size, valid_metrics)
        current = valid_metrics[monitor_metric]
        improved = current > best_value if maximize else current < best_value
        if improved:
            best_value = current
            model_api.save(arch_output_dir / "best.pt", args.image_size, valid_metrics)
    (arch_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def train(args) -> None:
    train_loader, valid_loader = make_semantic_loaders(args)
    train_torch_model(args, "unet", train_loader, valid_loader, run_epoch, "mean_iou", True)
