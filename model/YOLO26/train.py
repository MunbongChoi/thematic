from __future__ import annotations

import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from config import ANN_CODE_TO_MODEL_ID, MODEL_ID_TO_NAME
from data import (
    ensure_dir,
    feature_ann_code,
    geo_to_pixel_transform,
    image_size,
    load_label,
    load_rgb_image,
    polygons_from_geometry,
    split_samples,
)
from model.YOLO26.model import build_yolo_model


def polygon_to_yolo_line(ann_code: int, polygon: list[list[tuple[float, float]]], size: tuple[int, int]) -> str | None:
    width, height = size
    points = [(max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))) for x, y in polygon[0]]
    if len({(round(x, 6), round(y, 6)) for x, y in points}) < 3:
        return None
    coords = " ".join(coord for point in points for coord in (f"{point[0]:.6f}", f"{point[1]:.6f}"))
    return f"{ANN_CODE_TO_MODEL_ID[ann_code]} {coords}"


def write_yolo_label(label_path: Path, output_path: Path, size: tuple[int, int]) -> None:
    from config import FEATURES_FIELD, GEOMETRY_FIELD

    data = load_label(label_path)
    transform = geo_to_pixel_transform(data, label_path, size)
    lines: list[str] = []
    for feature_idx, feature in enumerate(data.get(FEATURES_FIELD, []), start=1):
        ann_code = feature_ann_code(feature, label_path, feature_idx)
        for polygon in polygons_from_geometry(feature.get(GEOMETRY_FIELD, {}), transform, size):
            line = polygon_to_yolo_line(ann_code, polygon, size)
            if line is not None:
                lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yolo_rgb_image(source: Path, target: Path, args) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        image = load_rgb_image(source).convert("RGB")
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(target, quality=args.yolo_jpeg_quality, subsampling=0)
        else:
            image.save(target)


def prepare_yolo_dataset(args) -> Path:
    if args.yolo_data:
        data_yaml = Path(args.yolo_data)
        if not data_yaml.is_file():
            raise FileNotFoundError(f"--yolo-data was provided but does not exist: {data_yaml}")
        print(f"Using existing YOLO dataset: {data_yaml}")
        return data_yaml

    prepared_root = Path(args.prepared_dir)
    if prepared_root.exists() and not args.force_prepare:
        candidates = sorted(prepared_root.rglob("data.yaml"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            print(f"Using existing prepared YOLO dataset: {candidates[0]}")
            return candidates[0]

    train_samples, valid_samples = split_samples(args)
    yolo_root = ensure_dir(Path(args.prepared_dir) / f"yolo_rgb_{args.yolo_image_format}")
    data_yaml = yolo_root / "data.yaml"
    if data_yaml.is_file() and not args.force_prepare:
        return data_yaml
    for split_name, samples in (("train", train_samples), ("val", valid_samples)):
        for image_path, label_path in tqdm(samples, desc=f"prepare-yolo-{split_name}", leave=False):
            image_output = yolo_root / "images" / split_name / f"{image_path.stem}.{args.yolo_image_format}"
            label_output = yolo_root / "labels" / split_name / f"{image_path.stem}.txt"
            write_yolo_rgb_image(image_path, image_output, args)
            write_yolo_label(label_path, label_output, image_size(image_path))
    lines = [f"path: {yolo_root.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {idx}: {name}" for idx, name in MODEL_ID_TO_NAME.items())
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def normalize_yolo_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "cpu":
        return "cpu"
    return normalized.replace("cuda:", "")


def train(args) -> None:
    arch_output_dir = ensure_dir(Path(args.output_dir) / "yolo")
    data_yaml = prepare_yolo_dataset(args)
    model = build_yolo_model(args.model_name_or_path)
    train_kwargs = {
        "data": str(data_yaml.resolve()),
        "task": "segment",
        "imgsz": args.image_size,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "lr0": args.lr,
        "project": str(arch_output_dir.resolve()),
        "name": "train",
        "exist_ok": True,
        "workers": args.num_workers,
        "cache": False,
        "amp": args.yolo_amp,
    }
    yolo_device = normalize_yolo_device(args.device)
    if yolo_device:
        train_kwargs["device"] = yolo_device
    print(
        "YOLO training config: "
        f"device={train_kwargs.get('device', 'auto')}, "
        f"batch={args.batch_size}, workers={args.num_workers}, "
        f"cache=False, amp={args.yolo_amp}, "
        f"torch_cuda={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}"
    )
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", arch_output_dir / "train"))
    weights_dir = save_dir / "weights"
    for name in ("best.pt", "last.pt"):
        source = weights_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"YOLO training finished but {source} was not found.")
        shutil.copy2(source, arch_output_dir / name)
