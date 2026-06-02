from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import MODEL_ID_TO_NAME, MODEL_ID_TO_TRAIN_ID
from data import load_rgb_image
from infer_common import (
    iter_images,
    iter_tile_windows,
    mask_bbox,
    parse_gsd_args,
    parse_tile_args,
    require_output_crs,
    save_panoptic_outputs,
    should_use_tiles,
)
from model.YOLO26.model import build_yolo_model


def offset_bbox(bbox: list[int], x_offset: int, y_offset: int) -> list[int]:
    return [int(bbox[0] + x_offset), int(bbox[1] + y_offset), int(bbox[2]), int(bbox[3])]


def paste_yolo_result(
    result,
    image_size: tuple[int, int],
    semantic: np.ndarray,
    panoptic: np.ndarray,
    segments: list[dict[str, object]],
    next_segment_id: int,
    write_box: tuple[int, int, int, int],
    relative_write_box: tuple[int, int, int, int],
) -> int:
    if result.masks is None or result.boxes is None:
        return next_segment_id
    masks = result.masks.data.detach().cpu()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    wx0, wy0, wx1, wy1 = write_box
    rx0, ry0, rx1, ry1 = relative_write_box
    region_panoptic = panoptic[wy0:wy1, wx0:wx1]
    region_semantic = semantic[wy0:wy1, wx0:wx1]
    for mask_tensor, model_id, confidence in zip(masks, classes, confidences):
        mask_image = Image.fromarray((mask_tensor.numpy() > 0.5).astype(np.uint8), mode="L").resize(image_size, Image.NEAREST)
        mask_crop = np.asarray(mask_image, dtype=bool)[ry0:ry1, rx0:rx1]
        instance_mask = mask_crop & (region_panoptic == 0)
        area_px = int(instance_mask.sum())
        if area_px == 0:
            continue
        train_id = MODEL_ID_TO_TRAIN_ID.get(int(model_id), 0)
        region_semantic[instance_mask] = train_id
        region_panoptic[instance_mask] = next_segment_id
        segments.append(
            {
                "id": next_segment_id,
                "category_id": train_id,
                "train_id": train_id,
                "model_class_id": int(model_id),
                "class_name": MODEL_ID_TO_NAME.get(int(model_id), str(model_id)),
                "confidence": float(confidence),
                "area_px": area_px,
                "bbox": offset_bbox(mask_bbox(instance_mask), wx0, wy0),
            }
        )
        next_segment_id += 1
    return next_segment_id


def run_inference(args) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    tile_config = parse_tile_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_yolo_model(args.checkpoint)
    images = iter_images(Path(args.input))
    predict_kwargs = {
        "task": "segment",
        "imgsz": args.image_size,
        "conf": args.threshold,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    results: list[dict[str, object]] = []
    for source_image in tqdm(images, desc="infer-yolo26"):
        image_path = Path(source_image)
        image = load_rgb_image(image_path).convert("RGB")
        semantic = np.zeros((image.height, image.width), dtype=np.uint8)
        panoptic = np.zeros((image.height, image.width), dtype=np.int32)
        segments: list[dict[str, object]] = []
        next_segment_id = 1
        if should_use_tiles(image, tile_config):
            windows = list(iter_tile_windows(image.width, image.height, tile_config))
            for window in tqdm(windows, desc="tiles-yolo26", leave=False):
                tile = image.crop(window.box)
                predictions = model.predict(source=tile, stream=False, **predict_kwargs)
                if len(predictions) != 1:
                    raise RuntimeError(f"Expected one YOLO prediction for {source_image} tile {window.box}, got {len(predictions)}.")
                next_segment_id = paste_yolo_result(
                    predictions[0],
                    tile.size,
                    semantic,
                    panoptic,
                    segments,
                    next_segment_id,
                    window.write_box,
                    window.relative_write_box,
                )
        else:
            predictions = model.predict(source=image, stream=False, **predict_kwargs)
            if len(predictions) != 1:
                raise RuntimeError(f"Expected one YOLO prediction for {source_image}, got {len(predictions)}.")
            next_segment_id = paste_yolo_result(
                predictions[0],
                image.size,
                semantic,
                panoptic,
                segments,
                next_segment_id,
                (0, 0, image.width, image.height),
                (0, 0, image.width, image.height),
            )
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir, output_crs, gsd, args.reference_label_root))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
