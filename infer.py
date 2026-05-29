from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    DEFAULT_IMAGE_SIZE,
    IMAGE_EXTENSIONS,
    MASK2FORMER_ID_TO_NAME,
    OUTPUT_ROOT,
    TRAIN_ID_TO_COLOR,
    TRAIN_ID_TO_NAME,
    YOLO_ID_TO_NAME,
    YOLO_ID_TO_TRAIN_ID,
)
from model import build_mask2former_processor, build_yolo_model, load_checkpoint, resolve_torch_device
from train import IMAGE_MEAN, IMAGE_STD, load_rgb_image, logits_from_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation inference and save masks/overlays/results JSON.")
    parser.add_argument("--checkpoint", default=str(OUTPUT_ROOT / "unet" / "best.pt"))
    parser.add_argument("--architecture", default="auto", choices=["auto", "yolo", "unet", "segformer", "mask2former"])
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--threshold", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default=None)
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


def panoptic_from_semantic(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    panoptic = np.zeros(mask.shape, dtype=np.int32)
    segments: list[dict[str, object]] = []
    next_segment_id = 1
    for train_id in sorted(int(value) for value in np.unique(mask) if int(value) != 0):
        segment_mask = mask == train_id
        area = int(segment_mask.sum())
        if area == 0:
            continue
        panoptic[segment_mask] = next_segment_id
        segments.append(
            {
                "id": next_segment_id,
                "category_id": train_id,
                "train_id": train_id,
                "class_name": TRAIN_ID_TO_NAME.get(train_id, str(train_id)),
                "area": area,
                "bbox": mask_bbox(segment_mask),
            }
        )
        next_segment_id += 1
    return panoptic, segments


def save_panoptic_outputs(
    image_path: Path,
    image: Image.Image,
    semantic: np.ndarray,
    panoptic: np.ndarray,
    segments: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    mask_path = output_dir / f"{image_path.stem}_semantic_mask.png"
    color_path = output_dir / f"{image_path.stem}_color.png"
    panoptic_path = output_dir / f"{image_path.stem}_panoptic.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    Image.fromarray(semantic.astype(np.uint8), mode="L").save(mask_path)
    colorize_mask(semantic).save(color_path)
    panoptic_id_to_rgb(panoptic.astype(np.int32)).save(panoptic_path)
    save_overlay(image, semantic, overlay_path)
    return {
        "image": str(image_path),
        "semantic_mask": str(mask_path),
        "panoptic_mask": str(panoptic_path),
        "color_mask": str(color_path),
        "overlay": str(overlay_path),
        "pixel_counts": class_summary(semantic),
        "segments_info": segments,
        "crs_note": "Output is a pixel-space panoptic segmentation. No distance/area CRS operation is performed.",
    }


def run_torch_semantic(args: argparse.Namespace, architecture: str, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", args.image_size))
    results: list[dict[str, object]] = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = logits_from_model(model, architecture, tensor)
            logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            mask = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        panoptic, segments = panoptic_from_semantic(mask)
        results.append(save_panoptic_outputs(image_path, image, mask, panoptic, segments, output_dir))
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
            for mask_tensor, yolo_id, confidence in zip(masks, classes, confidences):
                mask_image = Image.fromarray((mask_tensor.numpy() > 0.5).astype(np.uint8), mode="L").resize(image.size, Image.NEAREST)
                instance_mask = np.asarray(mask_image, dtype=bool) & (panoptic == 0)
                area = int(instance_mask.sum())
                if area == 0:
                    continue
                train_id = YOLO_ID_TO_TRAIN_ID.get(int(yolo_id), 0)
                semantic[instance_mask] = train_id
                panoptic[instance_mask] = next_segment_id
                segments.append(
                    {
                        "id": next_segment_id,
                        "category_id": train_id,
                        "train_id": train_id,
                        "yolo_class_id": int(yolo_id),
                        "class_name": YOLO_ID_TO_NAME.get(int(yolo_id), str(yolo_id)),
                        "confidence": float(confidence),
                        "area": area,
                        "bbox": mask_bbox(instance_mask),
                    }
                )
                next_segment_id += 1
        row = save_panoptic_outputs(image_path, image, semantic, panoptic, segments, output_dir)
        results.append(row)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_mask2former(args: argparse.Namespace, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", args.image_size))
    processor = build_mask2former_processor(checkpoint.get("model_name_or_path"))
    label_ids_to_fuse = set(MASK2FORMER_ID_TO_NAME.keys())
    results: list[dict[str, object]] = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=tensor)
            processed = processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[(image.height, image.width)],
                label_ids_to_fuse=label_ids_to_fuse,
            )[0]
        panoptic = processed["segmentation"].detach().cpu().numpy()
        semantic = np.zeros((image.height, image.width), dtype=np.uint8)
        segments: list[dict[str, object]] = []
        for info in processed["segments_info"]:
            label_id = int(info.get("label_id", info.get("category_id", 0)))
            train_id = YOLO_ID_TO_TRAIN_ID.get(label_id, 0)
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
                    "label_id": label_id,
                    "class_name": MASK2FORMER_ID_TO_NAME.get(label_id, str(label_id)),
                    "score": float(info.get("score", 0.0)),
                    "area": area,
                    "bbox": mask_bbox(segment_mask),
                }
            )
        row = save_panoptic_outputs(image_path, image, semantic, panoptic.astype(np.int32), segments, output_dir)
        results.append(row)
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
    if architecture == "mask2former":
        run_mask2former(args, model, checkpoint, device)
    elif architecture in {"unet", "segformer"}:
        run_torch_semantic(args, architecture, model, checkpoint, device)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")


if __name__ == "__main__":
    main()
