from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import DEFAULT_IMAGE_SIZE, DEFAULT_SEED, OUTPUT_ROOT, PREPARED_ROOT
from data import InstanceSegmentationDataset, collate_instance, ensure_dir, seed_everything, split_samples
from model.Mask2Former.model import ModelAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training entrypoint for Mask2Former.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--prepared-dir", default=str(PREPARED_ROOT))
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size.")
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
    train_dataset = InstanceSegmentationDataset(train_samples, args.image_size)
    valid_dataset = InstanceSegmentationDataset(valid_samples, args.image_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_instance,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **loader_kwargs)
    return train_loader, valid_loader, train_sampler, valid_sampler


def reduce_loss(total_loss: float, count: int, device: torch.device) -> float:
    totals = torch.tensor([total_loss, float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item() / max(1.0, totals[1].item()))


def run_epoch(
    model: DistributedDataParallel,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        progress = tqdm(loader, leave=False, disable=rank != 0)
        for batch in progress:
            images = batch["pixel_values"].to(device, non_blocking=True)
            mask_labels = [mask.to(device, non_blocking=True) for mask in batch["mask_labels"]]
            class_labels = [labels.to(device, non_blocking=True) for labels in batch["class_labels"]]
            outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
            if loss.ndim > 0:
                loss = loss.mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            count += 1
    return {"loss": reduce_loss(total_loss, count, device)}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    local_rank, rank, world_size, device = setup_distributed()
    try:
        train_loader, valid_loader, train_sampler, _ = make_ddp_loaders(args, rank, world_size)
        arch_output_dir = ensure_dir(Path(args.output_dir) / "mask2former") if rank == 0 else Path(args.output_dir) / "mask2former"
        model_api = ModelAPI.create(args.model_name_or_path)
        model = model_api.module.to(device)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_value = float("inf")
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
                if valid_metrics["loss"] < best_value:
                    best_value = valid_metrics["loss"]
                    model_api.save(arch_output_dir / "best.pt", args.image_size, valid_metrics)
        if rank == 0:
            (arch_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
