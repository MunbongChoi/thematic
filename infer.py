from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    DEFAULT_IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    MODEL_ID_TO_NAME,
    MODEL_ID_TO_TRAIN_ID,
    OUTPUT_CRS,
    OUTPUT_ROOT,
    TRAIN_ID_TO_COLOR,
    TRAIN_ID_TO_NAME,
)
from model import build_mask2former_processor, build_sam_processor, build_yolo_model, load_checkpoint, resolve_torch_device
from train import IMAGE_MEAN, IMAGE_STD, load_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run panoptic segmentation inference and save PNG/JSON/GeoJSON outputs.")
    parser.add_argument("--checkpoint", default=str(OUTPUT_ROOT / "unet" / "best.pt"))
    parser.add_argument("--architecture", default="auto", choices=["auto", "yolo", "unet", "mask2former", "sam"])
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--threshold", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prompt-source-mask2former-checkpoint",
        default=str(OUTPUT_ROOT / "mask2former" / "best.pt"),
        help="Mask2Former checkpoint used to create SAM bbox prompts and class ids.",
    )
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No supported images found under: {path}")
    return images


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def colorize_mask(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in TRAIN_ID_TO_COLOR.items():
        rgb[mask == int(class_id)] = color
    return Image.fromarray(rgb, mode="RGB")


def save_overlay(image: Image.Image, mask: np.ndarray, output_path: Path, alpha: int = 115) -> None:
    image_rgba = image.convert("RGBA")
    color = np.asarray(colorize_mask(mask), dtype=np.uint8)
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    foreground = mask > 0
    overlay[..., :3] = color
    overlay[..., 3] = np.where(foreground, alpha, 0).astype(np.uint8)
    Image.alpha_composite(image_rgba, Image.fromarray(overlay, mode="RGBA")).save(output_path)


def panoptic_id_to_rgb(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = mask % 256
    rgb[:, :, 1] = (mask // 256) % 256
    rgb[:, :, 2] = (mask // 65536) % 256
    return Image.fromarray(rgb, mode="RGB")


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    return [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]


def class_summary(mask: np.ndarray) -> dict[str, int]:
    summary: dict[str, int] = {}
    for class_id, class_name in TRAIN_ID_TO_NAME.items():
        count = int((mask == int(class_id)).sum())
        if count:
            summary[class_name] = count
    return summary


def connected_components(binary: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(binary.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            component = np.zeros(binary.shape, dtype=bool)
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component[cy, cx] = True
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(component)
    return components


def panoptic_from_semantic(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    panoptic = np.zeros(mask.shape, dtype=np.int32)
    segments: list[dict[str, object]] = []
    next_segment_id = 1
    for train_id in sorted(int(value) for value in np.unique(mask) if int(value) != 0):
        for component in connected_components(mask == train_id):
            area = int(component.sum())
            if area == 0:
                continue
            panoptic[component] = next_segment_id
            segments.append(
                {
                    "id": next_segment_id,
                    "category_id": train_id,
                    "train_id": train_id,
                    "class_name": TRAIN_ID_TO_NAME.get(train_id, str(train_id)),
                    "area": area,
                    "bbox": mask_bbox(component),
                }
            )
            next_segment_id += 1
    return panoptic, segments


def bbox_polygon_feature(segment: dict[str, object]) -> dict[str, object]:
    x, y, width, height = [int(value) for value in segment["bbox"]]
    return {
        "type": "Feature",
        "properties": segment,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]]],
        },
    }


def polygonize_panoptic(panoptic: np.ndarray, segments: list[dict[str, object]]) -> dict[str, object]:
    features: list[dict[str, object]] = []
    try:
        from affine import Affine
        from rasterio.features import shapes
    except ImportError:
        features = [bbox_polygon_feature(segment) for segment in segments]
    else:
        by_id = {int(segment["id"]): segment for segment in segments}
        for geometry, value in shapes(panoptic.astype(np.int32), mask=panoptic > 0, transform=Affine.identity()):
            segment_id = int(value)
            segment = by_id.get(segment_id)
            if segment is None:
                continue
            features.append({"type": "Feature", "properties": segment, "geometry": geometry})
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": OUTPUT_CRS}},
        "properties": {
            "crs_note": "Pixel-space GeoJSON. No authoritative metric area/distance operation is performed.",
        },
        "features": features,
    }


def save_panoptic_outputs(
    image_path: Path,
    image: Image.Image,
    semantic: np.ndarray,
    panoptic: np.ndarray,
    segments: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_semantic_mask.png"
    color_path = output_dir / f"{image_path.stem}_color.png"
    panoptic_path = output_dir / f"{image_path.stem}_panoptic.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    geojson_path = output_dir / f"{image_path.stem}_segments.geojson"
    Image.fromarray(semantic.astype(np.uint8), mode="L").save(mask_path)
    colorize_mask(semantic).save(color_path)
    panoptic_id_to_rgb(panoptic.astype(np.int32)).save(panoptic_path)
    save_overlay(image, semantic, overlay_path)
    geojson_path.write_text(json.dumps(polygonize_panoptic(panoptic, segments), indent=2), encoding="utf-8")
    return {
        "image": str(image_path),
        "semantic_mask": str(mask_path),
        "panoptic_mask": str(panoptic_path),
        "color_mask": str(color_path),
        "overlay": str(overlay_path),
        "geojson": str(geojson_path),
        "pixel_counts": class_summary(semantic),
        "segments_info": segments,
        "crs_note": "Output is pixel-space segmentation. No distance/area CRS operation is performed.",
    }


def run_unet(args: argparse.Namespace, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", args.image_size))
    results: list[dict[str, object]] = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            semantic = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        panoptic, segments = panoptic_from_semantic(semantic)
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_yolo(args: argparse.Namespace) -> None:
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
                area = int(instance_mask.sum())
                if area == 0:
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
                        "area": area,
                        "bbox": mask_bbox(instance_mask),
                    }
                )
                next_segment_id += 1
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def mask2former_predict(
    model: torch.nn.Module,
    checkpoint: dict,
    image: Image.Image,
    device: torch.device,
    image_size_arg: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", image_size_arg))
    processor = build_mask2former_processor(checkpoint.get("model_name_or_path"))
    tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
    panoptic = processed["segmentation"].detach().cpu().numpy().astype(np.int32)
    semantic = np.zeros((image.height, image.width), dtype=np.uint8)
    segments: list[dict[str, object]] = []
    for info in processed["segments_info"]:
        model_id = int(info.get("label_id", info.get("category_id", 0)))
        train_id = MODEL_ID_TO_TRAIN_ID.get(model_id, 0)
        segment_mask = panoptic == int(info["id"])
        area = int(segment_mask.sum())
        if area == 0:
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
                "area": area,
                "bbox": mask_bbox(segment_mask),
            }
        )
    return semantic, panoptic, segments


def run_mask2former(args: argparse.Namespace, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    results: list[dict[str, object]] = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        semantic, panoptic, segments = mask2former_predict(model, checkpoint, image, device, args.image_size)
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def sam_predict_mask(model: torch.nn.Module, processor: object, image: Image.Image, bbox: list[int], device: torch.device) -> np.ndarray:
    x, y, width, height = bbox
    input_box = [[float(x), float(y), float(x + width), float(y + height)]]
    inputs = processor(images=image, input_boxes=[input_box], return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    pred_masks = outputs.pred_masks
    while pred_masks.ndim > 3:
        pred_masks = pred_masks[:, 0]
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "post_process_masks"):
        try:
            masks = processor.image_processor.post_process_masks(
                pred_masks[:, None],
                inputs.get("original_sizes"),
                inputs.get("reshaped_input_sizes"),
            )[0]
            return masks[0].detach().cpu().numpy().astype(bool)
        except Exception:
            pass
    mask = pred_masks.sigmoid()[0].detach().cpu().numpy()
    mask_image = Image.fromarray((mask > 0.5).astype(np.uint8), mode="L").resize(image.size, Image.NEAREST)
    return np.asarray(mask_image, dtype=bool)


def run_sam(args: argparse.Namespace, sam_model: torch.nn.Module, sam_checkpoint: dict, device: torch.device) -> None:
    source_model, source_checkpoint = load_checkpoint(args.prompt_source_mask2former_checkpoint, map_location=device)
    processor = build_sam_processor(sam_checkpoint.get("model_name_or_path"))
    sam_model.to(device)
    sam_model.eval()
    output_dir = Path(args.output_dir)
    results: list[dict[str, object]] = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        _, _, prompt_segments = mask2former_predict(source_model, source_checkpoint, image, device, args.image_size)
        semantic = np.zeros((image.height, image.width), dtype=np.uint8)
        panoptic = np.zeros((image.height, image.width), dtype=np.int32)
        refined_segments: list[dict[str, object]] = []
        next_segment_id = 1
        for segment in prompt_segments:
            mask = sam_predict_mask(sam_model, processor, image, segment["bbox"], device) & (panoptic == 0)
            area = int(mask.sum())
            if area == 0:
                continue
            train_id = int(segment["train_id"])
            semantic[mask] = train_id
            panoptic[mask] = next_segment_id
            refined = dict(segment)
            refined.update({"id": next_segment_id, "area": area, "bbox": mask_bbox(mask), "prompt_source": "mask2former"})
            refined_segments.append(refined)
            next_segment_id += 1
        results.append(save_panoptic_outputs(image_path, image, semantic, panoptic, refined_segments, output_dir))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.architecture == "yolo":
        run_yolo(args)
        return

    device = resolve_torch_device(args.device)
    try:
        model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    except Exception:
        if args.architecture == "auto":
            run_yolo(args)
            return
        raise

    architecture = str(checkpoint["architecture"])
    if args.architecture != "auto" and args.architecture != architecture:
        raise ValueError(f"Checkpoint architecture is {architecture!r}, but --architecture={args.architecture!r}.")
    if architecture == "unet":
        run_unet(args, model, checkpoint, device)
    elif architecture == "mask2former":
        run_mask2former(args, model, checkpoint, device)
    elif architecture == "sam":
        run_sam(args, model, checkpoint, device)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")


if __name__ == "__main__":
    main()
