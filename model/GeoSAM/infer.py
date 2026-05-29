from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import load_rgb_image
from infer_common import iter_images, mask_bbox, parse_gsd_args, require_output_crs, save_panoptic_outputs
from model.GeoSAM.model import build_samgeo
from model.Mask2Former.infer import predict as mask2former_predict
from model.Mask2Former.model import build_processor
from model.common import load_checkpoint


def read_samgeo_mask(mask_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    import rasterio

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    if mask.shape != (image_size[1], image_size[0]):
        mask_image = Image.fromarray((mask > 0).astype(np.uint8), mode="L").resize(image_size, Image.NEAREST)
        return np.asarray(mask_image, dtype=np.uint8) > 0
    return mask > 0


def samgeo_predict_mask(sam: object, image_path: Path, image_size: tuple[int, int], bbox: list[int], temp_dir: Path) -> np.ndarray:
    x, y, width, height = bbox
    box_xyxy = [float(x), float(y), float(x + width), float(y + height)]
    output_path = temp_dir / f"{image_path.stem}_{x}_{y}_{width}_{height}_samgeo.tif"
    if hasattr(sam, "set_image"):
        sam.predict(boxes=box_xyxy, point_crs=None, output=str(output_path), dtype="uint8", multimask_output=False)
    else:
        sam.predict(
            image=str(image_path),
            boxes=box_xyxy,
            point_crs=None,
            output=str(output_path),
            dtype="uint8",
            multimask_output=False,
        )
    return read_samgeo_mask(output_path, image_size)


def run_inference(args, device: torch.device) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    source_model, source_checkpoint = load_checkpoint(args.prompt_source_mask2former_checkpoint, map_location=device)
    source_model.to(device)
    source_model.eval()
    source_processor = build_processor(source_checkpoint.get("model_name_or_path"))
    sam = build_samgeo(args, device)
    output_dir = Path(args.output_dir)
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="samgeo_") as temp_name:
        temp_dir = Path(temp_name)
        for image_path in iter_images(Path(args.input)):
            image = load_rgb_image(image_path)
            _, _, prompt_segments = mask2former_predict(
                source_model,
                source_checkpoint,
                image,
                device,
                args.image_size,
                source_processor,
                prepared=True,
            )
            if hasattr(sam, "set_image"):
                sam.set_image(str(image_path))
            semantic = np.zeros((image.height, image.width), dtype=np.uint8)
            panoptic = np.zeros((image.height, image.width), dtype=np.int32)
            refined_segments: list[dict[str, object]] = []
            next_segment_id = 1
            for segment in prompt_segments:
                mask = samgeo_predict_mask(sam, image_path, image.size, segment["bbox"], temp_dir) & (panoptic == 0)
                area_px = int(mask.sum())
                if area_px == 0:
                    continue
                train_id = int(segment["train_id"])
                semantic[mask] = train_id
                panoptic[mask] = next_segment_id
                refined_segments.append(
                    {
                        "id": next_segment_id,
                        "category_id": train_id,
                        "train_id": train_id,
                        "class_name": segment["class_name"],
                        "area_px": area_px,
                        "bbox": mask_bbox(mask),
                    }
                )
                next_segment_id += 1
            results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, refined_segments, output_dir, output_crs, gsd))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
