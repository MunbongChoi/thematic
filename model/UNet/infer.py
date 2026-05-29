from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_rgb_image
from infer_common import image_to_tensor, iter_images, panoptic_from_semantic, parse_gsd_args, require_output_crs, save_panoptic_outputs


def run_inference(args, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    output_dir = Path(args.output_dir)
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", args.image_size))
    results: list[dict[str, object]] = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode():
            logits = model(tensor)
            logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            semantic = logits.argmax(dim=1)[0].detach().cpu().numpy().astype("uint8")
        panoptic, segments = panoptic_from_semantic(semantic)
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir, output_crs, gsd))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
