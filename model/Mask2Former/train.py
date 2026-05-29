from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ensure_dir, make_instance_loaders
from model.Mask2Former.model import ModelAPI


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["pixel_values"].to(device, non_blocking=True)
            mask_labels = [mask.to(device, non_blocking=True) for mask in batch["mask_labels"]]
            class_labels = [labels.to(device, non_blocking=True) for labels in batch["class_labels"]]
            outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


def train(args) -> None:
    train_loader, valid_loader = make_instance_loaders(args)
    arch_output_dir = ensure_dir(Path(args.output_dir) / "mask2former")
    model_api = ModelAPI.create(args.model_name_or_path).prepare_for_training(args.device)
    optimizer = torch.optim.AdamW(model_api.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model_api.module, train_loader, model_api.device, optimizer)
        valid_metrics = run_epoch(model_api.module, valid_loader, model_api.device, None)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        model_api.save(arch_output_dir / "last.pt", args.image_size, valid_metrics)
        if valid_metrics["loss"] < best_value:
            best_value = valid_metrics["loss"]
            model_api.save(arch_output_dir / "best.pt", args.image_size, valid_metrics)
    (arch_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
