from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import PANOPTIC_CATEGORIES
from model import build_mask2former_processor, build_yolo_model, load_checkpoint, resolve_torch_device
from panoptic import panoptic_id_to_rgb
from train import IMAGE_MEAN, IMAGE_STD, load_rgb_image, logits_from_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run road extraction inference.")
    parser.add_argument("--checkpoint", default="runs/road_extraction/best_model.pt")
    parser.add_argument("--architecture", default="auto", type=str.lower, choices=["auto", "segformer", "unet", "yolo", "mask2former"])
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--image-size", type=int, default=512, help="Required for YOLO inference.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional road probability threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help="GPU device for inference. Use '0' or 'cuda:0' for torch models, and '0,1,2,3' for YOLO.",
    )
    return parser.parse_args()


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def iter_images(path: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if path.is_file():
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"Input file is not a supported image: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(f"No supported images found under: {path}")
    return images


def save_overlay(image: Image.Image, mask: Image.Image, output_path: Path) -> None:
    image_rgba = image.convert("RGBA")
    road = np.asarray(mask) > 0
    overlay = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    overlay[road] = [255, 40, 40, 110]
    overlay_image = Image.fromarray(overlay, mode="RGBA")
    Image.alpha_composite(image_rgba, overlay_image).save(output_path)


def save_panoptic_overlay(
    image: Image.Image,
    segmentation: torch.Tensor,
    segments_info: list[dict],
    output_path: Path,
) -> None:
    image_rgba = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    category_colors = {category.train_id: category.color for category in PANOPTIC_CATEGORIES}
    for info in segments_info:
        label_id = int(info.get("label_id", info.get("category_id", 0)))
        color = category_colors.get(label_id, (255, 40, 40))
        mask = segmentation.cpu().numpy() == int(info["id"])
        overlay[mask] = [color[0], color[1], color[2], 115]
    Image.alpha_composite(image_rgba, Image.fromarray(overlay, mode="RGBA")).save(output_path)


def json_scalar(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_panoptic_inference(args: argparse.Namespace, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(checkpoint.get("image_size", args.image_size))
    processor = build_mask2former_processor(checkpoint["model_name_or_path"])
    label_ids_to_fuse = {int(idx) for idx, label in checkpoint["id2label"].items() if int(idx) != 1 and label != "building"}
    results = []

    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        original_size = image.size
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=tensor)
            processed = processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[(original_size[1], original_size[0])],
                label_ids_to_fuse=label_ids_to_fuse,
            )[0]

        segmentation = processed["segmentation"].detach().cpu()
        segments_info = processed["segments_info"]
        panoptic_path = output_dir / f"{image_path.stem}_panoptic.png"
        overlay_path = output_dir / f"{image_path.stem}_panoptic_overlay.png"
        panoptic_id_to_rgb(segmentation.numpy()).save(panoptic_path)
        save_panoptic_overlay(image, segmentation, segments_info, overlay_path)

        instance_paths = []
        summary: dict[str, int] = {}
        serializable_segments = []
        for idx, info in enumerate(segments_info, start=1):
            label_id = int(info.get("label_id", info.get("category_id", 0)))
            label = checkpoint["id2label"].get(label_id, checkpoint["id2label"].get(str(label_id), str(label_id)))
            summary[label] = summary.get(label, 0) + 1
            segment_row = {key: json_scalar(value) for key, value in info.items()}
            segment_row["label"] = label
            serializable_segments.append(segment_row)
            if label == "building":
                instance_mask = (segmentation.numpy() == int(info["id"])).astype(np.uint8) * 255
                instance_path = output_dir / f"{image_path.stem}_building_{idx:03d}.png"
                Image.fromarray(instance_mask, mode="L").save(instance_path)
                instance_paths.append(str(instance_path))

        result = {
            "image": str(image_path),
            "panoptic": str(panoptic_path),
            "overlay": str(overlay_path),
            "building_instance_masks": instance_paths,
            "category_summary": summary,
            "segments_info": serializable_segments,
            "crs_note": "Output is pixel-space. Source CRS is not modified or used for measurement.",
        }
        results.append(result)

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def run_yolo_inference(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_yolo_model(args.checkpoint)
    confidence = args.threshold if args.threshold is not None else 0.25
    predict_kwargs = {
        "source": args.input,
        "task": "segment",
        "imgsz": args.image_size,
        "conf": confidence,
        "stream": False,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    predictions = model.predict(**predict_kwargs)

    results = []
    for result in predictions:
        image_path = Path(result.path)
        image = Image.fromarray(result.orig_img[:, :, ::-1]).convert("RGB")
        if result.masks is None:
            mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            mask_tensor = result.masks.data.detach().cpu()
            combined = torch.any(mask_tensor > 0.5, dim=0).numpy().astype(np.uint8)
            mask_array = np.asarray(
                Image.fromarray(combined * 255, mode="L").resize(image.size, Image.NEAREST),
                dtype=np.uint8,
            )

        mask_image = Image.fromarray(mask_array, mode="L")
        mask_path = output_dir / f"{image_path.stem}_road_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        mask_image.save(mask_path)
        save_overlay(image, mask_image, overlay_path)
        results.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "crs_note": "Output is pixel-space. Source CRS is not modified or used for measurement.",
            }
        )

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main() -> None:
    args = parse_args()
    if args.architecture == "yolo":
        run_yolo_inference(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    try:
        model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    except Exception:
        if args.architecture == "auto":
            run_yolo_inference(args)
            return
        raise
    model.to(device)
    model.eval()
    architecture = checkpoint["architecture"]
    if args.architecture != "auto" and args.architecture != architecture:
        raise ValueError(f"Checkpoint architecture is {architecture!r}, but --architecture={args.architecture!r}.")
    if architecture == "mask2former":
        run_panoptic_inference(args, model, checkpoint, device)
        return
    image_size = int(checkpoint.get("image_size", 512))

    results = []
    for image_path in iter_images(Path(args.input)):
        image = load_rgb_image(image_path)
        original_size = image.size
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = logits_from_model(model, architecture, tensor)
            logits = F.interpolate(logits, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)[0, 1]
            if args.threshold is None:
                mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            else:
                mask = (probs.cpu().numpy() >= args.threshold).astype(np.uint8)

        mask_image = Image.fromarray(mask * 255, mode="L")
        mask_path = output_dir / f"{image_path.stem}_road_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        mask_image.save(mask_path)
        save_overlay(image, mask_image, overlay_path)
        results.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "crs_note": "Output is pixel-space. Source CRS is not modified or used for measurement.",
            }
        )

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
