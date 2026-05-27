from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train import (
    PanopticSegmentationDataset,
    RoadSegmentationDataset,
    collate_batch,
    collate_panoptic_batch,
    parse_ann_codes,
    prepare_yolo_dataset,
    run_epoch,
)
from model import build_mask2former_processor, build_yolo_model, load_checkpoint, resolve_torch_device
import torch
from torch.utils.data import DataLoader, random_split

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a road extraction checkpoint.")
    parser.add_argument("--checkpoint", default="runs/road_extraction/best_model.pt")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--architecture", default="auto", type=str.lower, choices=["auto", "segformer", "unet", "yolo", "mask2former"])
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--image-size", type=int, default=512, help="Required for YOLO evaluation.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="GPU device for evaluation. Use '0' or 'cuda:0' for torch models, and '0,1,2,3' for YOLO multi-GPU.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--yolo-data-dir", default="outputs/yolo_eval_dataset")
    parser.add_argument(
        "--target-ann-codes",
        default="30",
        help="Comma-separated ANN_CD values to rasterize as road for GeoJSON geometry labels.",
    )
    return parser.parse_args()


def segment_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    return intersection / union if union else 0.0


def panoptic_quality_for_image(
    pred_segmentation: torch.Tensor,
    pred_segments_info: list[dict],
    gt_masks: torch.Tensor,
    gt_classes: torch.Tensor,
) -> dict[str, float]:
    pred_segments: list[tuple[int, torch.Tensor]] = []
    for info in pred_segments_info:
        label_id = int(info.get("label_id", info.get("category_id", -1)))
        segment_id = int(info["id"])
        mask = pred_segmentation == segment_id
        if mask.any():
            pred_segments.append((label_id, mask))

    gt_segments = [(int(label), gt_masks[idx] > 0.5) for idx, label in enumerate(gt_classes)]
    matches: list[tuple[float, int, int]] = []
    for pred_idx, (pred_class, pred_mask) in enumerate(pred_segments):
        for gt_idx, (gt_class, gt_mask) in enumerate(gt_segments):
            if pred_class != gt_class:
                continue
            iou = segment_iou(pred_mask, gt_mask)
            if iou > 0.5:
                matches.append((iou, pred_idx, gt_idx))

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    iou_sum = 0.0
    for iou, pred_idx, gt_idx in sorted(matches, reverse=True):
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        iou_sum += iou

    tp = len(used_gt)
    fp = len(pred_segments) - len(used_pred)
    fn = len(gt_segments) - len(used_gt)
    denom = tp + 0.5 * fp + 0.5 * fn
    building_pred = [mask for label, mask in pred_segments if label == 1]
    building_gt = [mask for label, mask in gt_segments if label == 1]
    pred_building_union = torch.zeros_like(pred_segmentation, dtype=torch.bool)
    gt_building_union = torch.zeros_like(pred_segmentation, dtype=torch.bool)
    for mask in building_pred:
        pred_building_union |= mask
    for mask in building_gt:
        gt_building_union |= mask

    return {
        "pq": iou_sum / denom if denom else 1.0,
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denom if denom else 1.0,
        "building_count_error": abs(len(building_pred) - len(building_gt)),
        "building_iou": segment_iou(pred_building_union, gt_building_union),
    }


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}


def evaluate_panoptic(args: argparse.Namespace, model: torch.nn.Module, checkpoint: dict) -> None:
    image_size = int(checkpoint.get("image_size", args.image_size))
    dataset_root = Path(args.dataset_root)
    split_dir = dataset_root / args.split
    dataset = PanopticSegmentationDataset(split_dir, image_size, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_panoptic_batch,
    )
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()
    processor = build_mask2former_processor(checkpoint["model_name_or_path"])
    metric_items: list[dict[str, float]] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device)
            mask_labels = [masks.to(device) for masks in batch["mask_labels"]]
            class_labels = [labels.to(device) for labels in batch["class_labels"]]
            outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu()))
            processed = processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[(image_size, image_size)] * images.shape[0],
                label_ids_to_fuse={int(idx) for idx, label in checkpoint["id2label"].items() if int(idx) != 1 and label != "building"},
            )
            for idx, result in enumerate(processed):
                pred_segmentation = result["segmentation"].detach().cpu()
                metric_items.append(
                    panoptic_quality_for_image(
                        pred_segmentation,
                        result["segments_info"],
                        batch["mask_labels"][idx],
                        batch["class_labels"][idx],
                    )
                )

    metrics = average_metric_dicts(metric_items)
    if losses:
        metrics["loss"] = float(np.mean(losses))
    print(json.dumps(metrics, indent=2))


def evaluate_yolo(args: argparse.Namespace) -> None:
    data_yaml = prepare_yolo_dataset(args, Path(args.yolo_data_dir))
    model = build_yolo_model(args.checkpoint)
    val_kwargs = {
        "data": str(data_yaml),
        "task": "segment",
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "split": "val",
        "workers": args.num_workers,
    }
    if args.device:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)
    print(metrics)


def main() -> None:
    args = parse_args()
    if args.architecture == "yolo":
        evaluate_yolo(args)
        return

    try:
        model, checkpoint = load_checkpoint(args.checkpoint)
    except Exception:
        if args.architecture == "auto":
            evaluate_yolo(args)
            return
        raise

    architecture = checkpoint["architecture"]
    if args.architecture != "auto" and args.architecture != architecture:
        raise ValueError(f"Checkpoint architecture is {architecture!r}, but --architecture={args.architecture!r}.")
    if architecture == "mask2former":
        evaluate_panoptic(args, model, checkpoint)
        return
    image_size = int(checkpoint.get("image_size", 512))
    target_ann_codes = parse_ann_codes(args.target_ann_codes)

    dataset_root = Path(args.dataset_root)
    split_dir = dataset_root / args.split
    if args.split == "valid" and not ((split_dir / "image").exists() and (split_dir / "label").exists()):
        full_dataset = RoadSegmentationDataset(dataset_root / "train", image_size, args.limit, target_ann_codes)
        if not 0.0 < args.val_ratio < 1.0:
            raise ValueError("--val-ratio must be between 0 and 1.")
        if len(full_dataset) < 2:
            raise ValueError("At least two training samples are required when dataset/valid is unavailable.")
        val_size = max(1, int(len(full_dataset) * args.val_ratio))
        if val_size >= len(full_dataset):
            val_size = len(full_dataset) - 1
        train_size = len(full_dataset) - val_size
        _, dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        dataset = RoadSegmentationDataset(split_dir, image_size, args.limit, target_ann_codes)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    device = resolve_torch_device(args.device)
    model.to(device)
    metrics = run_epoch(model, loader, architecture, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
