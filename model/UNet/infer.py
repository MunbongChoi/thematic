from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_rgb_image
from infer_common import (
    image_to_tensor,
    iter_images,
    iter_tile_windows,
    panoptic_from_semantic,
    parse_gsd_args,
    parse_tile_args,
    require_output_crs,
    save_panoptic_outputs,
    should_use_tiles,
)


def predict_semantic(model: torch.nn.Module, image, image_size: int, device: torch.device) -> np.ndarray:
    tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        logits = model(tensor)
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].detach().cpu().numpy().astype("uint8")


def predict_semantic_tiled(model: torch.nn.Module, image, image_size: int, device: torch.device, tile_config) -> np.ndarray:
    semantic = np.zeros((image.height, image.width), dtype=np.uint8)
    windows = list(iter_tile_windows(image.width, image.height, tile_config))
    for window in tqdm(windows, desc="tiles-unet", leave=False):
        tile = image.crop(window.box)
        tile_semantic = predict_semantic(model, tile, image_size, device)
        rx0, ry0, rx1, ry1 = window.relative_write_box
        wx0, wy0, wx1, wy1 = window.write_box
        semantic[wy0:wy1, wx0:wx1] = tile_semantic[ry0:ry1, rx0:rx1]
    return semantic


def run_inference(args, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    tile_config = parse_tile_args(args)
    output_dir = Path(args.output_dir)
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", args.image_size))
    results: list[dict[str, object]] = []
    images = iter_images(Path(args.input))
    for image_path in tqdm(images, desc="infer-unet"):
        image = load_rgb_image(image_path)
        if should_use_tiles(image, tile_config):
            semantic = predict_semantic_tiled(model, image, image_size, device, tile_config)
        else:
            semantic = predict_semantic(model, image, image_size, device)
        panoptic, segments = panoptic_from_semantic(semantic, min_area_px=int(getattr(args, "min_mask_area_px", 0) or 0), split_stuff=False)
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir, output_crs, gsd, args.reference_label_root))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
