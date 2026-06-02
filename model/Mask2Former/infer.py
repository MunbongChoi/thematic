from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from config import MODEL_ID_TO_NAME, MODEL_ID_TO_TRAIN_ID
from data import load_rgb_image
from infer_common import image_to_tensor, iter_images, mask_bbox, parse_gsd_args, require_output_crs, save_panoptic_outputs
from model.Mask2Former.model import build_processor


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


def run_inference(args, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    output_dir = Path(args.output_dir)
    model.to(device)
    model.eval()
    processor = build_processor(checkpoint.get("model_name_or_path"))
    results: list[dict[str, object]] = []
    images = iter_images(Path(args.input))
    for image_path in tqdm(images, desc="infer-mask2former"):
        image = load_rgb_image(image_path)
        semantic, panoptic, segments = predict(model, checkpoint, image, device, args.image_size, processor, prepared=True)
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir, output_crs, gsd, args.reference_label_root))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
