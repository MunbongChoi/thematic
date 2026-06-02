from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import MODEL_ID_TO_NAME, MODEL_ID_TO_TRAIN_ID
from infer_common import mask_bbox, parse_gsd_args, require_output_crs, save_panoptic_outputs
from model.YOLO26.model import build_yolo_model


def run_inference(args) -> None:
    output_crs = require_output_crs(args.output_crs)
    gsd = parse_gsd_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_yolo_model(args.checkpoint)
    predict_kwargs = {
        "source": args.input,
        "task": "segment",
        "imgsz": args.image_size,
        "conf": args.threshold,
        "stream": False,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    predictions = model.predict(**predict_kwargs)
    results: list[dict[str, object]] = []
    for result in predictions:
        image_path = Path(result.path)
        if image_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"YOLO inference output came from unsupported input {image_path}. Expected .tif or .tiff.")
        image = Image.fromarray(result.orig_img[:, :, ::-1]).convert("RGB")
        semantic = np.zeros((image.height, image.width), dtype=np.uint8)
        panoptic = np.zeros((image.height, image.width), dtype=np.int32)
        segments: list[dict[str, object]] = []
        next_segment_id = 1
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for mask_tensor, model_id, confidence in zip(masks, classes, confidences):
                mask_image = Image.fromarray((mask_tensor.numpy() > 0.5).astype(np.uint8), mode="L").resize(image.size, Image.NEAREST)
                instance_mask = np.asarray(mask_image, dtype=bool) & (panoptic == 0)
                area_px = int(instance_mask.sum())
                if area_px == 0:
                    continue
                train_id = MODEL_ID_TO_TRAIN_ID.get(int(model_id), 0)
                semantic[instance_mask] = train_id
                panoptic[instance_mask] = next_segment_id
                segments.append(
                    {
                        "id": next_segment_id,
                        "category_id": train_id,
                        "train_id": train_id,
                        "model_class_id": int(model_id),
                        "class_name": MODEL_ID_TO_NAME.get(int(model_id), str(model_id)),
                        "confidence": float(confidence),
                        "area_px": area_px,
                        "bbox": mask_bbox(instance_mask),
                    }
                )
                next_segment_id += 1
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir, output_crs, gsd, args.reference_label_root))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
