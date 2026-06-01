from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import BACKGROUND_ID, DEFAULT_IMAGE_SIZE, DEFAULT_SEED, OUTPUT_ROOT, PREPARED_ROOT, TRAIN_ID_TO_NAME
from data import SemanticSegmentationDataset, collate_semantic, ensure_dir, seed_everything, split_samples
from model.UNet.model import ModelAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training entrypoint for UNet.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--prepared-dir", default=str(PREPARED_ROOT))
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU process.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def distributed_backend() -> str:
    return "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"


def setup_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=distributed_backend(), rank=rank, world_size=world_size)
    return local_rank, rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def make_ddp_loaders(args: argparse.Namespace, rank: int, world_size: int) -> tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    train_samples, valid_samples = split_samples(args)
    train_dataset = SemanticSegmentationDataset(train_samples, args.image_size)
    valid_dataset = SemanticSegmentationDataset(valid_samples, args.image_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_semantic,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return (
        DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs),
        DataLoader(valid_dataset, sampler=valid_sampler, **loader_kwargs),
        train_sampler,
        valid_sampler,
    )


def add_metric_stats(stats: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
    preds = logits.argmax(dim=1)
    iou_sum = 0.0
    iou_count = 0
    for class_id in TRAIN_ID_TO_NAME:
        if class_id == BACKGROUND_ID:
            continue
        pred = preds == int(class_id)
        target = labels == int(class_id)
        union = (pred | target).sum().item()
        if union:
            iou_sum += float((pred & target).sum().item() / union)
            iou_count += 1
    stats[0] += float(loss.detach())
    stats[1] += 1.0
    stats[2] += iou_sum
    stats[3] += float(iou_count)
    correct = float((preds == labels).sum().item())
    total = float(labels.numel())
    stats[4] += correct
    stats[5] += total


def reduce_stats(stats: torch.Tensor) -> dict[str, float]:
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return {
        "loss": float(stats[0].item() / max(1.0, stats[1].item())),
        "mean_iou": float(stats[2].item() / max(1.0, stats[3].item())),
        "pixel_accuracy": float(stats[4].item() / max(1.0, stats[5].item())),
    }


def run_epoch(
    model: DistributedDataParallel,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats = torch.zeros((6,), dtype=torch.float64, device=device)
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        progress = tqdm(loader, leave=False, disable=rank != 0)
        for batch in progress:
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
            add_metric_stats(stats, logits.detach(), labels.detach(), loss)
    return reduce_stats(stats)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    local_rank, rank, world_size, device = setup_distributed()
    try:
        train_loader, valid_loader, train_sampler, _ = make_ddp_loaders(args, rank, world_size)
        arch_output_dir = ensure_dir(Path(args.output_dir) / "unet") if rank == 0 else Path(args.output_dir) / "unet"
        model_api = ModelAPI.create(args.model_name_or_path)
        model = model_api.module.to(device)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_value = -float("inf")
        history: list[dict[str, object]] = []
        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(ddp_model, train_loader, device, rank, optimizer)
            valid_metrics = run_epoch(ddp_model, valid_loader, device, rank, None)
            row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
            if rank == 0:
                history.append(row)
                print(json.dumps(row, indent=2))
                model_api.module = ddp_model
                model_api.save(arch_output_dir / "last.pt", args.image_size, valid_metrics)
                if valid_metrics["mean_iou"] > best_value:
                    best_value = valid_metrics["mean_iou"]
                    model_api.save(arch_output_dir / "best.pt", args.image_size, valid_metrics)
        if rank == 0:
            (arch_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
