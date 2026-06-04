from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from config import MODEL_ID_TO_NAME, MODEL_ID_TO_TRAIN_ID
from data import load_rgb_image
from infer_common import (
    image_to_tensor,
    iter_images,
    iter_tile_windows,
    mask_bbox,
    parse_gsd_args,
    parse_tile_args,
    require_output_crs,
    save_panoptic_outputs,
    should_use_tiles,
)
from model.Mask2Former.model import build_processor


@dataclass
class Mask2FormerTileSegment:
    model_id: int
    train_id: int
    class_name: str
    score: float
    mask: np.ndarray
    write_box: tuple[int, int, int, int]


def predict(
    model: torch.nn.Module,
    checkpoint: dict,
    image,
    device: torch.device,
    image_size_arg: int,
    processor: Any | None = None,
    prepared: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if not prepared:
        model.to(device)
        model.eval()
    image_size = int(checkpoint.get("image_size", image_size_arg))
    processor = processor or build_processor(checkpoint.get("model_name_or_path"))
    tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        outputs = model(pixel_values=tensor)
        processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
    panoptic = processed["segmentation"].detach().cpu().numpy().astype(np.int32)
    semantic = np.zeros((image.height, image.width), dtype=np.uint8)
    segments: list[dict[str, object]] = []
    for info in processed["segments_info"]:
        model_id = int(info.get("label_id", info.get("category_id", 0)))
        train_id = MODEL_ID_TO_TRAIN_ID.get(model_id, 0)
        segment_mask = panoptic == int(info["id"])
        area_px = int(segment_mask.sum())
        if area_px == 0:
            continue
        semantic[segment_mask] = train_id
        segments.append(
            {
                "id": int(info["id"]),
                "category_id": train_id,
                "train_id": train_id,
                "model_class_id": model_id,
                "class_name": MODEL_ID_TO_NAME.get(model_id, str(model_id)),
                "score": float(info.get("score", 0.0)),
                "area_px": area_px,
                "bbox": mask_bbox(segment_mask),
            }
        )
    return semantic, panoptic, segments


def min_segment_score(args) -> float:
    return max(0.0, min(1.0, float(getattr(args, "min_segment_score", 0.0))))


def min_segment_area(args) -> int:
    return max(0, int(getattr(args, "min_mask_area_px", 0) or 0))


def max_segment_overlap(args) -> float:
    return max(0.0, min(1.0, float(getattr(args, "max_mask_overlap", 1.0))))


def collect_tile_segments(
    tile_panoptic: np.ndarray,
    tile_segments: list[dict[str, object]],
    write_box: tuple[int, int, int, int],
    relative_write_box: tuple[int, int, int, int],
    args,
) -> list[Mask2FormerTileSegment]:
    rx0, ry0, rx1, ry1 = relative_write_box
    min_area = min_segment_area(args)
    min_score = min_segment_score(args)
    output: list[Mask2FormerTileSegment] = []
    for segment in tile_segments:
        score = float(segment.get("score", 0.0))
        if score < min_score:
            continue
        segment_id = int(segment["id"])
        mask = (tile_panoptic == segment_id)[ry0:ry1, rx0:rx1]
        area_px = int(mask.sum())
        if area_px < min_area:
            continue
        train_id = int(segment.get("train_id", segment.get("category_id", 0)))
        model_id = int(segment.get("model_class_id", train_id))
        output.append(
            Mask2FormerTileSegment(
                model_id=model_id,
                train_id=train_id,
                class_name=str(segment.get("class_name", MODEL_ID_TO_NAME.get(model_id, str(model_id)))),
                score=score,
                mask=mask,
                write_box=write_box,
            )
        )
    return output


def paste_tile_segments(
    tile_segments: list[Mask2FormerTileSegment],
    semantic: np.ndarray,
    panoptic: np.ndarray,
    args,
) -> list[dict[str, object]]:
    ordered = sorted(tile_segments, key=lambda item: item.score, reverse=True)
    min_area = min_segment_area(args)
    max_overlap = max_segment_overlap(args)
    segments: list[dict[str, object]] = []
    next_segment_id = 1
    for segment in ordered:
        wx0, wy0, wx1, wy1 = segment.write_box
        region_semantic = semantic[wy0:wy1, wx0:wx1]
        region_panoptic = panoptic[wy0:wy1, wx0:wx1]
        original_area = int(segment.mask.sum())
        if original_area < min_area:
            continue
        overlap_ratio = float((segment.mask & (region_panoptic > 0)).sum() / max(1, original_area))
        if overlap_ratio > max_overlap:
            continue
        instance_mask = segment.mask & (region_panoptic == 0)
        area_px = int(instance_mask.sum())
        if area_px < min_area:
            continue
        region_semantic[instance_mask] = segment.train_id
        region_panoptic[instance_mask] = next_segment_id
        segments.append(
            {
                "id": next_segment_id,
                "category_id": segment.train_id,
                "train_id": segment.train_id,
                "model_class_id": segment.model_id,
                "class_name": segment.class_name,
                "score": segment.score,
                "area_px": area_px,
                "bbox": [mask_bbox(instance_mask)[0] + wx0, mask_bbox(instance_mask)[1] + wy0, mask_bbox(instance_mask)[2], mask_bbox(instance_mask)[3]],
            }
        )
        next_segment_id += 1
    return segments


def predict_tiled(
    model: torch.nn.Module,
    checkpoint: dict,
    image,
    device: torch.device,
    image_size_arg: int,
    processor: Any,
    tile_config,
    args,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    semantic = np.zeros((image.height, image.width), dtype=np.uint8)
    panoptic = np.zeros((image.height, image.width), dtype=np.int32)
    collected_segments: list[Mask2FormerTileSegment] = []
    windows = list(iter_tile_windows(image.width, image.height, tile_config))
    for window in tqdm(windows, desc="tiles-mask2former", leave=False):
        tile = image.crop(window.box)
        _, tile_panoptic, tile_segments = predict(model, checkpoint, tile, device, image_size_arg, processor, prepared=True)
        collected_segments.extend(collect_tile_segments(tile_panoptic, tile_segments, window.write_box, window.relative_write_box, args))
    segments = paste_tile_segments(collected_segments, semantic, panoptic, args)
    return semantic, panoptic, segments


def run_inference(args, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    tile_config = parse_tile_args(args)
    output_dir = Path(args.output_dir)
    model.to(device)
    model.eval()
    processor = build_processor(checkpoint.get("model_name_or_path"))
    results: list[dict[str, object]] = []
    images = iter_images(Path(args.input))
    for image_path in tqdm(images, desc="infer-mask2former"):
        image = load_rgb_image(image_path)
        if should_use_tiles(image, tile_config):
            semantic, panoptic, segments = predict_tiled(model, checkpoint, image, device, args.image_size, processor, tile_config, args)
        else:
            semantic, panoptic, segments = predict(model, checkpoint, image, device, args.image_size, processor, prepared=True)
            min_area = min_segment_area(args)
            min_score = min_segment_score(args)
            if min_area > 0 or min_score > 0:
                keep_ids = {
                    int(segment["id"])
                    for segment in segments
                    if int(segment.get("area_px", 0)) >= min_area and float(segment.get("score", 0.0)) >= min_score
                }
                panoptic = np.where(np.isin(panoptic, list(keep_ids)), panoptic, 0).astype(np.int32)
                semantic = np.where(panoptic > 0, semantic, 0).astype(np.uint8)
                segments = [segment for segment in segments if int(segment["id"]) in keep_ids]
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir, output_crs, gsd, args.reference_label_root))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
