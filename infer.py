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


def class_summary(mask: np.ndarray) -> dict[str, int]:
    summary: dict[str, int] = {}
    for class_id, class_name in TRAIN_ID_TO_NAME.items():
        count = int((mask == int(class_id)).sum())
        if count:
            summary[class_name] = count
    return summary


def save_semantic_outputs(image_path: Path, image: Image.Image, mask: np.ndarray, output_dir: Path) -> dict[str, object]:
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    color_path = output_dir / f"{image_path.stem}_color.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
    colorize_mask(mask).save(color_path)
    save_overlay(image, mask, overlay_path)
    return {
        "image": str(image_path),
        "mask": str(mask_path),
        "color_mask": str(color_path),
        "overlay": str(overlay_path),
        "pixel_counts": class_summary(mask),
        "crs_note": "Output is a pixel-space segmentation mask. No distance/area CRS operation is performed.",
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
        results.append(save_semantic_outputs(image_path, image, mask, output_dir))
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
        instances: list[dict[str, object]] = []
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for idx, (mask_tensor, yolo_id, confidence) in enumerate(zip(masks, classes, confidences), start=1):
                mask_image = Image.fromarray((mask_tensor.numpy() > 0.5).astype(np.uint8), mode="L").resize(image.size, Image.NEAREST)
                instance_mask = np.asarray(mask_image, dtype=bool)
                train_id = YOLO_ID_TO_TRAIN_ID.get(int(yolo_id), 0)
                semantic[instance_mask] = train_id
                instances.append(
                    {
                        "id": idx,
                        "yolo_class_id": int(yolo_id),
                        "class_name": YOLO_ID_TO_NAME.get(int(yolo_id), str(yolo_id)),
                        "confidence": float(confidence),
                        "pixel_count": int(instance_mask.sum()),
                    }
                )
        row = save_semantic_outputs(image_path, image, semantic, output_dir)
        row["instances"] = instances
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
            semantic[segment_mask] = train_id
            segments.append(
                {
                    "id": int(info["id"]),
                    "label_id": label_id,
                    "class_name": MASK2FORMER_ID_TO_NAME.get(label_id, str(label_id)),
                    "score": float(info.get("score", 0.0)),
                    "pixel_count": int(segment_mask.sum()),
                }
            )
        row = save_semantic_outputs(image_path, image, semantic, output_dir)
        panoptic_path = output_dir / f"{image_path.stem}_panoptic_ids.png"
        Image.fromarray(np.clip(panoptic, 0, 255).astype(np.uint8), mode="L").save(panoptic_path)
        row["panoptic_ids"] = str(panoptic_path)
        row["segments"] = segments
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

