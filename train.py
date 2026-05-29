from __future__ import annotations

import argparse
import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    ANALYSIS_CRS_EPSG,
    ANN_CODE_FIELD,
    ANN_CODE_TO_MODEL_ID,
    ANN_CODE_TO_TRAIN_ID,
    BACKGROUND_ID,
    CLASSES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    FEATURES_FIELD,
    GEOMETRY_FIELD,
    GEOMETRY_TYPES,
    IMAGE_EXTENSIONS,
    LABEL_CRS_EPSG,
    MODEL_ID_TO_NAME,
    OUTPUT_ROOT,
    PREPARED_ROOT,
    PROPERTIES_FIELD,
    RGB_RASTER_EXTENSIONS,
    TRAIN_ID_TO_NAME,
)
from model import ModelAPI, ModelConfig, build_yolo_model

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SUPPORTED_TRAIN_ARCHITECTURES = ("all", "yolo", "unet", "mask2former")


class LabelSchemaError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build panoptic segmentation datasets and train models.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--prepared-dir", default=str(PREPARED_ROOT))
    parser.add_argument("--architecture", default="all", choices=SUPPORTED_TRAIN_ARCHITECTURES)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers. For YOLO DDP this is per GPU process.")
    parser.add_argument("--yolo-cache", default="ram", choices=["false", "ram", "disk"], help="Ultralytics image cache mode for YOLO.")
    parser.add_argument("--yolo-amp", action=argparse.BooleanOptionalAction, default=True, help="Enable Ultralytics AMP for YOLO.")
    parser.add_argument("--device", default=None, help="Use cpu, 0, cuda:0, or 0,1 where supported.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def array_to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB array with shape HxWx3, got {array.shape}.")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scaled = array.astype(np.float32) / max(1, info.max)
    else:
        scaled = array.astype(np.float32)
        max_value = float(np.nanmax(scaled)) if scaled.size else 1.0
        if max_value > 1.0:
            scaled /= max(max_value, 1.0)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_rgb_image(image_path: Path) -> Image.Image:
    # Source imagery is RGB. TIFF files are still read with rasterio because
    # some tiled/compressed GeoTIFFs fail when PIL tries to load pixel data.
    if image_path.suffix.lower() in RGB_RASTER_EXTENSIONS:
        try:
            import rasterio
            from rasterio.errors import NotGeoreferencedWarning
        except ImportError as exc:
            raise ImportError("rasterio is required to read RGB TIFF inputs. Install requirements.txt.") from exc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(image_path) as src:
                if src.count < 3:
                    raise ValueError(f"{image_path} is expected to be RGB but has {src.count} band(s).")
                bands = src.read([1, 2, 3])
        return Image.fromarray(array_to_uint8_rgb(np.moveaxis(bands, 0, -1)), mode="RGB")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def image_size(image_path: Path) -> tuple[int, int]:
    if image_path.suffix.lower() in RGB_RASTER_EXTENSIONS:
        try:
            import rasterio
            from rasterio.errors import NotGeoreferencedWarning
        except ImportError:
            with Image.open(image_path) as image:
                return image.size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(image_path) as src:
                return src.width, src.height
    with Image.open(image_path) as image:
        return image.size


def load_label(label_path: Path) -> dict:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("type") != "FeatureCollection":
        raise LabelSchemaError(f"{label_path} must be a GeoJSON FeatureCollection.")
    crs = data.get("crs")
    if not crs:
        raise LabelSchemaError(f"{label_path} is missing CRS metadata. Expected EPSG:{LABEL_CRS_EPSG}.")
    name = str(crs.get("properties", {}).get("name", ""))
    if f"EPSG::{LABEL_CRS_EPSG}" not in name and f"EPSG:{LABEL_CRS_EPSG}" not in name:
        raise LabelSchemaError(f"{label_path} CRS must be EPSG:{LABEL_CRS_EPSG}, got {name!r}.")
    return data


def iter_raw_rings(geometry: dict) -> Iterable[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield from coordinates
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def label_bounds(data: dict, label_path: Path) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in data.get(FEATURES_FIELD, []):
        geometry = feature.get(GEOMETRY_FIELD, {})
        if geometry.get("type") not in GEOMETRY_TYPES:
            raise LabelSchemaError(f"{label_path} has unsupported geometry type {geometry.get('type')!r}.")
        for ring in iter_raw_rings(geometry):
            for point in ring:
                if len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
    if not xs or not ys:
        raise LabelSchemaError(f"{label_path} does not contain polygon coordinates.")
    return min(xs), min(ys), max(xs), max(ys)


def geo_to_pixel_transform(data: dict, label_path: Path, size: tuple[int, int]) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = label_bounds(data, label_path)
    width, height = size
    if width <= 0 or height <= 0 or max_x <= min_x or max_y <= min_y:
        raise LabelSchemaError(f"{label_path} has invalid image size or label bounds.")
    # EPSG:5186 tile coordinates are mapped into pixel space for mask creation.
    return min_x, max_y, (max_x - min_x) / width, (max_y - min_y) / height


def repaired_geometry(geometry: dict) -> dict:
    if geometry.get("type") not in GEOMETRY_TYPES:
        return geometry
    try:
        from shapely.geometry import mapping, shape
    except ImportError:
        return geometry
    geom = shape(geometry)
    if geom.is_valid:
        return geometry
    repaired = geom.buffer(0)
    if repaired.is_empty:
        return geometry
    return mapping(repaired)


def geo_ring_to_pixels(
    ring: list[list[float]],
    transform: tuple[float, float, float, float],
    size: tuple[int, int],
) -> list[tuple[float, float]]:
    width, height = size
    min_x, max_y, x_res, y_res = transform
    pixels: list[tuple[float, float]] = []
    for point in ring:
        if len(point) < 2:
            continue
        x = (float(point[0]) - min_x) / x_res
        y = (max_y - float(point[1])) / y_res
        pixels.append((max(0.0, min(width - 1.0, x)), max(0.0, min(height - 1.0, y))))
    return pixels


def polygons_from_geometry(
    geometry: dict,
    transform: tuple[float, float, float, float],
    size: tuple[int, int],
) -> list[list[list[tuple[float, float]]]]:
    geometry = repaired_geometry(geometry)
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if geometry_type == "Polygon" else coordinates if geometry_type == "MultiPolygon" else []
    pixel_polygons: list[list[list[tuple[float, float]]]] = []
    for polygon in polygons:
        rings: list[list[tuple[float, float]]] = []
        for ring in polygon:
            pixel_ring = geo_ring_to_pixels(ring, transform, size)
            if len({(round(x, 3), round(y, 3)) for x, y in pixel_ring}) >= 3:
                rings.append(pixel_ring)
        if rings:
            pixel_polygons.append(rings)
    return pixel_polygons


def feature_ann_code(feature: dict, label_path: Path, feature_idx: int) -> int:
    raw_ann_code = feature.get(PROPERTIES_FIELD, {}).get(ANN_CODE_FIELD)
    if raw_ann_code is None:
        raise LabelSchemaError(f"{label_path} feature {feature_idx} is missing {ANN_CODE_FIELD}.")
    try:
        ann_code = int(raw_ann_code)
    except (TypeError, ValueError) as exc:
        raise LabelSchemaError(f"{label_path} feature {feature_idx} has invalid ANN_CD={raw_ann_code!r}.") from exc
    if ann_code not in ANN_CODE_TO_TRAIN_ID:
        raise LabelSchemaError(f"{label_path} feature {feature_idx} contains unmapped ANN_CD={ann_code}.")
    return ann_code


def rasterize_feature_mask(polygons: list[list[list[tuple[float, float]]]], size: tuple[int, int]) -> np.ndarray:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon[0], fill=1)
        for hole in polygon[1:]:
            draw.polygon(hole, fill=0)
    return np.asarray(mask, dtype=bool)


def render_semantic_mask(label_path: Path, size: tuple[int, int]) -> np.ndarray:
    data = load_label(label_path)
    transform = geo_to_pixel_transform(data, label_path, size)
    width, height = size
    semantic = np.full((height, width), BACKGROUND_ID, dtype=np.uint8)
    for feature_idx, feature in enumerate(data.get(FEATURES_FIELD, []), start=1):
        ann_code = feature_ann_code(feature, label_path, feature_idx)
        polygons = polygons_from_geometry(feature.get(GEOMETRY_FIELD, {}), transform, size)
        if not polygons:
            continue
        feature_mask = rasterize_feature_mask(polygons, size)
        if feature_mask.any():
            semantic[feature_mask] = ANN_CODE_TO_TRAIN_ID[ann_code]
    return semantic


def render_instance_targets(label_path: Path, size: tuple[int, int], output_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = load_label(label_path)
    transform = geo_to_pixel_transform(data, label_path, size)
    masks: list[torch.Tensor] = []
    class_labels: list[int] = []
    boxes: list[list[float]] = []
    for feature_idx, feature in enumerate(data.get(FEATURES_FIELD, []), start=1):
        ann_code = feature_ann_code(feature, label_path, feature_idx)
        polygons = polygons_from_geometry(feature.get(GEOMETRY_FIELD, {}), transform, size)
        if not polygons:
            continue
        mask = rasterize_feature_mask(polygons, size).astype(np.uint8)
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        mask_image = Image.fromarray(mask, mode="L").resize((output_size, output_size), Image.NEAREST)
        masks.append(torch.from_numpy(np.asarray(mask_image, dtype=np.float32)))
        class_labels.append(ANN_CODE_TO_MODEL_ID[ann_code])
    if not masks:
        return (
            torch.zeros((0, output_size, output_size), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, 4), dtype=torch.float32),
        )
    return torch.stack(masks), torch.tensor(class_labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32)


def build_image_index(image_dir: Path) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        by_stem.setdefault(image_path.stem, image_path)
    return by_stem


def find_image_for_label(label_path: Path, image_index: dict[str, Path]) -> Path:
    if label_path.stem in image_index:
        return image_index[label_path.stem]
    raise FileNotFoundError(f"No image matched label: {label_path}")


def collect_samples(split_dir: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    image_dir = split_dir / "image"
    label_dir = split_dir / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected image/ and label/ under {split_dir}")
    image_index = build_image_index(image_dir)
    label_paths = sorted(path for path in label_dir.rglob("*") if path.suffix.lower() == ".json")
    if limit:
        label_paths = label_paths[:limit]
    if not label_paths:
        raise FileNotFoundError(f"No JSON labels found under {label_dir}")
    return [(find_image_for_label(label_path, image_index), label_path) for label_path in label_paths]


def has_labeled_split(split_dir: Path) -> bool:
    return (split_dir / "image").exists() and (split_dir / "label").exists() and any((split_dir / "label").rglob("*.json"))


def split_samples(args: argparse.Namespace) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    dataset_root = Path(args.dataset_root)
    train_samples = collect_samples(dataset_root / "train", args.limit)
    if has_labeled_split(dataset_root / "valid"):
        return train_samples, collect_samples(dataset_root / "valid", args.limit)
    if has_labeled_split(dataset_root / "test"):
        return train_samples, collect_samples(dataset_root / "test", args.limit)
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    shuffled = train_samples[:]
    random.Random(args.seed).shuffle(shuffled)
    val_size = max(1, min(len(shuffled) - 1, int(len(shuffled) * args.val_ratio)))
    return shuffled[val_size:], shuffled[:val_size]


class SemanticSegmentationDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], image_size_value: int) -> None:
        self.samples = samples
        self.image_size = image_size_value

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, label_path = self.samples[index]
        image = load_rgb_image(image_path)
        semantic = render_semantic_mask(label_path, image.size)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.fromarray(semantic, mode="L").resize((self.image_size, self.image_size), Image.NEAREST)
        image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        return {
            "pixel_values": (image_tensor - IMAGE_MEAN) / IMAGE_STD,
            "labels": torch.from_numpy(np.asarray(mask, dtype=np.int64)),
            "image_path": str(image_path),
            "label_path": str(label_path),
        }


class InstanceSegmentationDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], image_size_value: int) -> None:
        self.samples = samples
        self.image_size = image_size_value

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, label_path = self.samples[index]
        image = load_rgb_image(image_path)
        mask_labels, class_labels, boxes = render_instance_targets(label_path, image.size, self.image_size)
        resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        scale_x = self.image_size / max(1, image.width)
        scale_y = self.image_size / max(1, image.height)
        if boxes.numel():
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        return {
            "pixel_values": (image_tensor - IMAGE_MEAN) / IMAGE_STD,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "boxes": boxes,
            "pil_image": resized,
            "image_path": str(image_path),
            "label_path": str(label_path),
        }


def collate_semantic(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch if isinstance(item["pixel_values"], torch.Tensor)]),
        "labels": torch.stack([item["labels"] for item in batch if isinstance(item["labels"], torch.Tensor)]),
        "image_path": [str(item["image_path"]) for item in batch],
        "label_path": [str(item["label_path"]) for item in batch],
    }


def collate_instance(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch if isinstance(item["pixel_values"], torch.Tensor)]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "pil_images": [item["pil_image"] for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
        "label_path": [str(item["label_path"]) for item in batch],
    }


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, collate_fn: Callable) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def make_semantic_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_samples, valid_samples = split_samples(args)
    return (
        make_loader(SemanticSegmentationDataset(train_samples, args.image_size), args, True, collate_semantic),
        make_loader(SemanticSegmentationDataset(valid_samples, args.image_size), args, False, collate_semantic),
    )


def make_instance_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_samples, valid_samples = split_samples(args)
    return (
        make_loader(InstanceSegmentationDataset(train_samples, args.image_size), args, True, collate_instance),
        make_loader(InstanceSegmentationDataset(valid_samples, args.image_size), args, False, collate_instance),
    )


def compute_semantic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    ious: list[float] = []
    for class_id in TRAIN_ID_TO_NAME:
        if class_id == BACKGROUND_ID:
            continue
        pred = preds == class_id
        target = labels == class_id
        union = (pred | target).sum().item()
        if union:
            ious.append(float((pred & target).sum().item() / union))
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": float((preds == labels).sum().item() / max(1, labels.numel())),
    }


def average_metrics(items: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(items)
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def run_unet_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False):
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            metric_items.append(compute_semantic_metrics(logits.detach(), labels.detach()))
    metrics = average_metrics(metric_items)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def run_mask2former_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False):
            images = batch["pixel_values"].to(device)
            mask_labels = [mask.to(device) for mask in batch["mask_labels"]]
            class_labels = [labels.to(device) for labels in batch["class_labels"]]
            outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


def train_torch_model(
    args: argparse.Namespace,
    architecture: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epoch_runner: Callable,
    monitor_metric: str,
    maximize: bool,
) -> None:
    arch_output_dir = ensure_dir(Path(args.output_dir) / architecture)
    config = ModelConfig(architecture=architecture, model_name_or_path=args.model_name_or_path).normalized()
    model_api = ModelAPI.create(config).prepare_for_training(args.device)
    optimizer = torch.optim.AdamW(model_api.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = -float("inf") if maximize else float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_runner(model_api.module, train_loader, model_api.device, optimizer)
        valid_metrics = epoch_runner(model_api.module, valid_loader, model_api.device, None)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        model_api.save(arch_output_dir / "last.pt", args.image_size, valid_metrics)
        current = valid_metrics[monitor_metric]
        improved = current > best_value if maximize else current < best_value
        if improved:
            best_value = current
            model_api.save(arch_output_dir / "best.pt", args.image_size, valid_metrics)
    (arch_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def polygon_to_yolo_line(ann_code: int, polygon: list[list[tuple[float, float]]], size: tuple[int, int]) -> str | None:
    width, height = size
    points = [(max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))) for x, y in polygon[0]]
    if len({(round(x, 6), round(y, 6)) for x, y in points}) < 3:
        return None
    coords = " ".join(coord for point in points for coord in (f"{point[0]:.6f}", f"{point[1]:.6f}"))
    return f"{ANN_CODE_TO_MODEL_ID[ann_code]} {coords}"


def write_yolo_label(label_path: Path, output_path: Path, size: tuple[int, int]) -> None:
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


def write_yolo_rgb_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        load_rgb_image(source).convert("RGB").save(target)


def prepare_yolo_dataset(args: argparse.Namespace) -> Path:
    train_samples, valid_samples = split_samples(args)
    yolo_root = ensure_dir(Path(args.prepared_dir) / "yolo_rgb")
    data_yaml = yolo_root / "data.yaml"
    if data_yaml.is_file() and not args.force_prepare:
        return data_yaml
    for split_name, samples in (("train", train_samples), ("val", valid_samples)):
        for image_path, label_path in tqdm(samples, desc=f"prepare-yolo-{split_name}", leave=False):
            image_output = yolo_root / "images" / split_name / f"{image_path.stem}.png"
            label_output = yolo_root / "labels" / split_name / f"{image_path.stem}.txt"
            write_yolo_rgb_image(image_path, image_output)
            write_yolo_label(label_path, label_output, image_size(image_path))
    lines = [f"path: {yolo_root.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {idx}: {name}" for idx, name in MODEL_ID_TO_NAME.items())
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def export_semantic_masks(args: argparse.Namespace) -> None:
    train_samples, valid_samples = split_samples(args)
    root = ensure_dir(Path(args.prepared_dir) / "semantic_masks")
    for split_name, samples in (("train", train_samples), ("val", valid_samples)):
        for image_path, label_path in tqdm(samples, desc=f"prepare-semantic-{split_name}", leave=False):
            mask = render_semantic_mask(label_path, image_size(image_path))
            output_path = root / split_name / f"{image_path.stem}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(output_path)
    metadata = {
        "input_crs": f"EPSG:{LABEL_CRS_EPSG}",
        "analysis_crs": f"EPSG:{ANALYSIS_CRS_EPSG}",
        "output_crs": "pixel",
        "classes": TRAIN_ID_TO_NAME,
        "note": "GeoJSON coordinates are mapped to image pixel space from label tile bounds. No metric CRS operation is performed.",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_all_datasets(args: argparse.Namespace) -> None:
    export_semantic_masks(args)
    prepare_yolo_dataset(args)
    root = ensure_dir(args.prepared_dir)
    metadata = {
        "classes": [item.__dict__ for item in CLASSES],
        "label_crs": f"EPSG:{LABEL_CRS_EPSG}",
        "output_crs": "pixel",
        "spatial_operation": "pixel-space rasterization only",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def normalize_yolo_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "cpu":
        return "cpu"
    return normalized.replace("cuda:", "")


def normalize_yolo_cache(cache_mode: str) -> bool | str:
    if cache_mode == "false":
        return False
    return cache_mode


def train_yolo(args: argparse.Namespace) -> None:
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
        "cache": normalize_yolo_cache(args.yolo_cache),
        "amp": args.yolo_amp,
    }
    yolo_device = normalize_yolo_device(args.device)
    if yolo_device:
        train_kwargs["device"] = yolo_device
    print(
        "YOLO training config: "
        f"device={train_kwargs.get('device', 'auto')}, "
        f"batch={args.batch_size}, workers={args.num_workers}, "
        f"cache={train_kwargs['cache']}, amp={args.yolo_amp}, "
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


def train_unet(args: argparse.Namespace) -> None:
    train_loader, valid_loader = make_semantic_loaders(args)
    train_torch_model(args, "unet", train_loader, valid_loader, run_unet_epoch, "mean_iou", True)


def train_mask2former(args: argparse.Namespace) -> None:
    train_loader, valid_loader = make_instance_loaders(args)
    train_torch_model(args, "mask2former", train_loader, valid_loader, run_mask2former_epoch, "loss", False)


TRAIN_DISPATCH = {
    "yolo": train_yolo,
    "unet": train_unet,
    "mask2former": train_mask2former,
}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.prepare_only:
        prepare_all_datasets(args)
        print(f"Prepared datasets written to {Path(args.prepared_dir).resolve()}")
        return
    if args.architecture == "all" and args.model_name_or_path:
        raise ValueError("--model-name-or-path can target only one architecture. Run each architecture separately when overriding it.")
    architectures = ("yolo", "unet", "mask2former") if args.architecture == "all" else (args.architecture,)
    for architecture in architectures:
        print(f"Training {architecture} -> {Path(args.output_dir) / architecture}")
        TRAIN_DISPATCH[architecture](args)


if __name__ == "__main__":
    main()
