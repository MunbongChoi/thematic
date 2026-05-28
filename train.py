from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config import DEFAULT_MODEL_NAME, LABEL_CRS_EPSG, NUM_CLASSES, OUTPUT_ROOT, PANOPTIC_DATASET_DIR, panoptic_categories_as_coco
from model import (
    DEFAULT_YOLO_SEG_MODEL,
    ModelConfig,
    build_model,
    build_yolo_model,
    resolve_torch_device,
    resolve_torch_device_ids,
    save_checkpoint,
)
from panoptic import (
    category_summary,
    masks_and_classes_from_panoptic,
    panoptic_id_to_rgb,
    polygons_from_geometry,
    render_panoptic_label,
    segment_infos_to_json,
)


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
RASTER_SUFFIXES = {".tif", ".tiff"}
DEFAULT_OUTPUT_DIR = os.environ.get("SATSEG_OUTPUT_DIR", "runs/road_extraction")
SUPPORTED_TRAIN_ARCHITECTURES = ("segformer", "unet", "yolo", "mask2former")


def normalize_architecture(value: str) -> str:
    architecture = value.strip().lower()
    if architecture not in SUPPORTED_TRAIN_ARCHITECTURES:
        choices = ", ".join(SUPPORTED_TRAIN_ARCHITECTURES)
        raise argparse.ArgumentTypeError(f"Unsupported architecture: {value!r}. Choose one of: {choices}")
    return architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train satellite segmentation model.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root containing train/image and train/label.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and metrics. Can also be set with SATSEG_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--architecture",
        default="segformer",
        type=normalize_architecture,
        choices=SUPPORTED_TRAIN_ARCHITECTURES,
        help="Model architecture.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Hugging Face model id/path for SegFormer, or an arbitrary label for UNet checkpoints.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Square training size in pixels.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="GPU device for training. Use '0' or 'cuda:0' for torch models, and '0,1,2,3' for YOLO multi-GPU.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Used when dataset/valid is empty.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick checks.")
    parser.add_argument("--panoptic-data-dir", default=str(PANOPTIC_DATASET_DIR), help="COCO panoptic export directory.")
    parser.add_argument("--export-panoptic-only", action="store_true", help="Export COCO panoptic labels and exit.")
    parser.add_argument(
        "--target-ann-codes",
        default="30",
        help="Comma-separated ANN_CD values to rasterize as road for GeoJSON geometry labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_output_dir(path: str | Path, purpose: str = "output") -> Path:
    output_dir = Path(path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create {purpose} directory: {output_dir}\n"
            "The current location is not writable. Use a writable path, for example:\n"
            f"  python train.py --output-dir /tmp/satellite_runs/{output_dir.name} ...\n"
            "or set a default for this shell:\n"
            "  export SATSEG_OUTPUT_DIR=/tmp/satellite_runs/road_extraction"
        ) from exc
    if not output_dir.is_dir():
        raise NotADirectoryError(f"{purpose} path exists but is not a directory: {output_dir}")
    write_test_path = output_dir / f".write_test_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        write_test_path.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise PermissionError(
            f"Cannot write to {purpose} directory: {output_dir.resolve()}\n"
            "Use a writable absolute output path, for example:\n"
            f"  python train.py --output-dir /tmp/satellite_runs/{output_dir.name} ...\n"
            "or set:\n"
            "  export SATSEG_OUTPUT_DIR=/tmp/satellite_runs/road_extraction"
        ) from exc
    try:
        write_test_path.unlink()
    except OSError:
        pass
    return output_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_torch_model_for_training(config: ModelConfig, device_arg: str | None) -> tuple[nn.Module, torch.device]:
    device = resolve_torch_device(device_arg)
    model = build_model(config).to(device)
    device_ids = resolve_torch_device_ids(device_arg)
    if device.type == "cuda" and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        print(f"Using DataParallel on CUDA devices: {device_ids}")
    return model, device


def parse_road_coords(value: str) -> list[tuple[float, float]]:
    if not value or value == "EMPTY":
        return []
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) < 6 or len(values) % 2 != 0:
        return []
    return [(values[idx], values[idx + 1]) for idx in range(0, len(values), 2)]


def parse_ann_codes(value: str | Iterable[int]) -> set[int]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("--target-ann-codes must contain at least one ANN_CD value.")
        try:
            return {int(item) for item in items}
        except ValueError as exc:
            raise ValueError(f"Invalid ANN_CD list: {value}") from exc
    return {int(item) for item in value}


def array_to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[:, :, :3]
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scaled = array.astype(np.float32) / max(1, info.max)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    max_value = float(np.nanmax(array)) if array.size else 1.0
    scaled = array.astype(np.float32)
    if max_value <= 1.0:
        scaled *= 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def load_rgb_image(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() in RASTER_SUFFIXES:
        try:
            import rasterio
            from rasterio.errors import NotGeoreferencedWarning
        except ImportError:
            warnings.warn(
                "rasterio is not installed; falling back to PIL for TIF reading. "
                "Install rasterio for more reliable geospatial raster handling.",
                RuntimeWarning,
                stacklevel=2,
            )
            return Image.open(image_path).convert("RGB")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(image_path) as src:
                band_indexes = list(range(1, min(src.count, 3) + 1))
                array = src.read(band_indexes)
        array = np.moveaxis(array, 0, -1)
        return Image.fromarray(array_to_uint8_rgb(array), mode="RGB")

    return Image.open(image_path).convert("RGB")


def image_size(image_path: Path) -> tuple[int, int]:
    if image_path.suffix.lower() in RASTER_SUFFIXES:
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


def validate_label_crs(data: dict, label_path: Path) -> None:
    crs = data.get("crs")
    if not crs:
        raise ValueError(f"{label_path} is missing CRS metadata. Expected EPSG:{LABEL_CRS_EPSG} for geometry labels.")
    name = str(crs.get("properties", {}).get("name", ""))
    if f"EPSG::{LABEL_CRS_EPSG}" not in name and f"EPSG:{LABEL_CRS_EPSG}" not in name:
        raise ValueError(f"{label_path} CRS must be EPSG:{LABEL_CRS_EPSG}, got {name!r}.")


def feature_is_target(feature: dict, target_ann_codes: set[int], label_path: Path, feature_idx: int) -> bool:
    properties = feature.get("properties", {})
    ann_cd = properties.get("ANN_CD")
    if ann_cd is None:
        raise ValueError(f"{label_path} feature {feature_idx} is missing ANN_CD; cannot select target class.")
    try:
        return int(ann_cd) in target_ann_codes
    except (TypeError, ValueError):
        raise ValueError(f"{label_path} feature {feature_idx} has invalid ANN_CD={ann_cd!r}.")


def iter_polygon_rings(geometry: dict) -> Iterable[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield from coordinates
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def label_bounds(data: dict) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in data.get("features", []):
        for ring in iter_polygon_rings(feature.get("geometry", {})):
            for point in ring:
                if len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
    if not xs or not ys:
        raise ValueError("GeoJSON label does not contain polygon coordinates.")
    return min(xs), min(ys), max(xs), max(ys)


def geo_to_pixel_transform(data: dict, size: tuple[int, int], label_path: Path) -> tuple[float, float, float, float]:
    """Return minx, maxy, x_res, y_res for labels stored in map coordinates.

    The changed dataset stores EPSG:5186 coordinates in JSON while the source TIF files
    are not georeferenced. Bounds are therefore inferred from the label tile extent.
    No distance or area is calculated here; the CRS coordinates are only mapped into
    image pixel space for supervised mask generation.
    """

    validate_label_crs(data, label_path)
    width, height = size
    min_x, min_y, max_x, max_y = label_bounds(data)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Invalid GeoJSON label bounds.")
    return min_x, max_y, (max_x - min_x) / width, (max_y - min_y) / height


def geo_ring_to_pixels(
    ring: list[list[float]],
    transform: tuple[float, float, float, float],
    size: tuple[int, int],
) -> list[tuple[float, float]]:
    width, height = size
    min_x, max_y, x_res, y_res = transform
    points: list[tuple[float, float]] = []
    for point in ring:
        if len(point) < 2:
            continue
        x = (float(point[0]) - min_x) / x_res
        y = (max_y - float(point[1])) / y_res
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    return points


def road_polygons_from_label(
    label_path: Path,
    image_size: tuple[int, int] | None = None,
    target_ann_codes: set[int] | None = None,
) -> list[list[tuple[float, float]]]:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    polygons: list[list[tuple[float, float]]] = []
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        coords = parse_road_coords(properties.get("road_imcoords", "EMPTY"))
        if len(coords) >= 3:
            polygons.append(coords)

    if polygons:
        return polygons

    if image_size is None:
        raise ValueError(f"{label_path} uses geometry coordinates; image_size is required.")

    target_ann_codes = target_ann_codes or {30}
    transform = geo_to_pixel_transform(data, image_size, label_path)
    for feature_idx, feature in enumerate(data.get("features", []), start=1):
        if not feature_is_target(feature, target_ann_codes, label_path, feature_idx):
            continue
        for polygon in polygons_from_geometry(feature.get("geometry", {}), transform, image_size):
            exterior = polygon[0]
            if len(exterior) >= 3:
                polygons.append(exterior)
    return polygons


def rasterize_road_mask(label_path: Path, size: tuple[int, int], target_ann_codes: set[int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    has_pixel_coords = False
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        coords = parse_road_coords(properties.get("road_imcoords", "EMPTY"))
        if len(coords) < 3:
            continue
        has_pixel_coords = True
        clipped = [(max(0, min(width - 1, x)), max(0, min(height - 1, y))) for x, y in coords]
        draw.polygon(clipped, fill=1)
    if has_pixel_coords:
        return mask

    transform = geo_to_pixel_transform(data, size, label_path)
    for feature_idx, feature in enumerate(data.get("features", []), start=1):
        if not feature_is_target(feature, target_ann_codes, label_path, feature_idx):
            continue
        for polygon in polygons_from_geometry(feature.get("geometry", {}), transform, size):
            exterior = polygon[0]
            if len(exterior) < 3:
                continue
            draw.polygon(exterior, fill=1)
            for hole in polygon[1:]:
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)
    return mask


def build_image_index(image_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    suffixes = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in suffixes):
        by_name.setdefault(image_path.name, image_path)
        by_stem.setdefault(image_path.stem, image_path)
    return by_name, by_stem


def find_image_for_label(image_dir: Path, label_path: Path, image_index: tuple[dict[str, Path], dict[str, Path]]) -> Path:
    by_name, by_stem = image_index
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    for feature in data.get("features", []):
        image_id = feature.get("properties", {}).get("image_id")
        if image_id:
            image_id_path = Path(str(image_id))
            if image_id_path.name in by_name:
                return by_name[image_id_path.name]

    stem = label_path.stem
    if stem in by_stem:
        return by_stem[stem]
    raise FileNotFoundError(f"No image matched label: {label_path}")


def collect_samples(split_dir: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")
    image_dir = split_dir / "image"
    label_dir = split_dir / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected image/ and label/ under {split_dir}")

    image_index = build_image_index(image_dir)
    if not image_index[0]:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    label_paths = sorted(label_dir.rglob("*.json"))
    if limit:
        label_paths = label_paths[:limit]
    if not label_paths:
        raise FileNotFoundError(f"No label JSON files found in {label_dir}")
    return [(find_image_for_label(image_dir, label_path, image_index), label_path) for label_path in label_paths]


def has_labeled_split(split_dir: Path) -> bool:
    return (split_dir / "image").exists() and (split_dir / "label").exists() and any((split_dir / "label").rglob("*.json"))


def split_dataset_for_validation(dataset: Dataset, args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if len(dataset) < 2:
        raise ValueError("At least two training samples are required when dataset/valid is unavailable.")
    val_size = max(1, int(len(dataset) * args.val_ratio))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1
    train_size = len(dataset) - val_size
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    collate_fn: Callable[[list], object],
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def split_samples(args: argparse.Namespace) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    dataset_root = Path(args.dataset_root)
    train_samples = collect_samples(dataset_root / "train", args.limit)
    valid_dir = dataset_root / "valid"
    if has_labeled_split(valid_dir):
        return train_samples, collect_samples(valid_dir, args.limit)

    rng = random.Random(args.seed)
    shuffled = train_samples[:]
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("At least two training samples are required when dataset/valid is unavailable.")
    val_size = max(1, int(len(shuffled) * args.val_ratio))
    if val_size >= len(shuffled):
        val_size = len(shuffled) - 1
    return shuffled[val_size:], shuffled[:val_size]


def write_yolo_seg_label(label_path: Path, image_size: tuple[int, int], output_path: Path, target_ann_codes: set[int]) -> None:
    width, height = image_size
    lines: list[str] = []
    for polygon in road_polygons_from_label(label_path, image_size, target_ann_codes):
        normalized_pairs: list[tuple[float, float]] = []
        for x, y in polygon:
            x_norm = max(0.0, min(1.0, x / width))
            y_norm = max(0.0, min(1.0, y / height))
            normalized_pairs.append((x_norm, y_norm))
        unique_points = {(round(x, 6), round(y, 6)) for x, y in normalized_pairs}
        if len(unique_points) < 3 or abs(polygon_area(normalized_pairs)) < 1e-8:
            continue
        normalized = [coord for point in normalized_pairs for coord in (f"{point[0]:.6f}", f"{point[1]:.6f}")]
        if len(normalized) >= 6:
            lines.append("0 " + " ".join(normalized))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def prepare_yolo_dataset(args: argparse.Namespace, yolo_root: Path) -> Path:
    train_samples, valid_samples = split_samples(args)
    target_ann_codes = parse_ann_codes(args.target_ann_codes)

    for split_name, samples in (("train", train_samples), ("val", valid_samples)):
        for image_path, label_path in samples:
            image_output = yolo_root / "images" / split_name / image_path.name
            label_output = yolo_root / "labels" / split_name / f"{image_path.stem}.txt"
            image_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_output)
            write_yolo_seg_label(label_path, image_size(image_path), label_output, target_ann_codes)

    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: road",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def train_yolo(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir, "YOLO output")
    yolo_root = output_dir / "yolo_dataset"
    data_yaml = prepare_yolo_dataset(args, yolo_root)
    model_name = args.model_name_or_path
    if model_name == "nvidia/segformer-b0-finetuned-ade-512-512":
        model_name = DEFAULT_YOLO_SEG_MODEL

    model = build_yolo_model(model_name)
    train_kwargs = {
        "data": str(data_yaml.resolve()),
        "task": "segment",
        "imgsz": args.image_size,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "lr0": args.lr,
        "project": str(output_dir.resolve()),
        "name": "yolo",
        "exist_ok": True,
        "workers": args.num_workers,
    }
    if args.device:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)
    best_candidates = [output_dir / "yolo" / "weights" / "best.pt"]
    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is not None:
        best_candidates.insert(0, Path(save_dir) / "weights" / "best.pt")
    for best_path in best_candidates:
        if best_path.exists():
            shutil.copy2(best_path, output_dir / "best_yolo.pt")
            break
    else:
        searched = ", ".join(str(path) for path in best_candidates)
        raise FileNotFoundError(f"YOLO training finished but best.pt was not found. Searched: {searched}")

    result_dict = getattr(results, "results_dict", None)
    print(json.dumps(result_dict, indent=2) if result_dict else results)


def write_panoptic_split(samples: list[tuple[Path, Path]], split_name: str, output_root: Path) -> dict[str, object]:
    images_dir = output_root / "images" / split_name
    panoptic_dir = output_root / "panoptic" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    panoptic_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []

    for image_id, (image_path, label_path) in enumerate(samples, start=1):
        image = load_rgb_image(image_path)
        label = render_panoptic_label(label_path, image.size)
        image_file_name = f"{image_path.stem}.png"
        panoptic_file_name = f"{image_path.stem}_panoptic.png"
        image.save(images_dir / image_file_name)
        panoptic_id_to_rgb(label.panoptic_id_mask).save(panoptic_dir / panoptic_file_name)
        images.append(
            {
                "id": image_id,
                "file_name": image_file_name,
                "width": image.width,
                "height": image.height,
                "source_image": str(image_path),
                "source_label": str(label_path),
            }
        )
        annotations.append(
            {
                "image_id": image_id,
                "file_name": panoptic_file_name,
                "segments_info": segment_infos_to_json(label.segments_info),
                "category_summary": category_summary(label.segments_info),
                "crs_note": "EPSG:5186 coordinates are mapped into pixel space from label tile bounds; no CRS measurement is performed.",
            }
        )

    return {"images": images, "annotations": annotations, "categories": panoptic_categories_as_coco()}


def prepare_panoptic_dataset(args: argparse.Namespace, output_root: Path) -> Path:
    train_samples, valid_samples = split_samples(args)
    output_root = ensure_output_dir(output_root, "panoptic export")
    for split_name, samples in (("train", train_samples), ("val", valid_samples)):
        dataset = write_panoptic_split(samples, split_name, output_root)
        with (output_root / f"panoptic_{split_name}.json").open("w", encoding="utf-8") as file:
            json.dump(dataset, file, ensure_ascii=False, indent=2)
    return output_root


class PanopticSegmentationDataset(Dataset):
    """Mask2Former panoptic dataset.

    Labels are GeoJSON polygons in EPSG:5186. They are converted to pixel-space masks
    from the label tile bounds because the source TIFs are not georeferenced.
    """

    def __init__(self, split_dir: Path, image_size: int, limit: int | None = None) -> None:
        self.samples = collect_samples(split_dir, limit)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, label_path = self.samples[index]
        image = load_rgb_image(image_path)
        label = render_panoptic_label(label_path, image.size)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_labels, class_labels = masks_and_classes_from_panoptic(label, (self.image_size, self.image_size))
        semantic = Image.fromarray(label.semantic_mask.astype(np.uint8), mode="L").resize(
            (self.image_size, self.image_size),
            Image.NEAREST,
        )

        image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - IMAGE_MEAN) / IMAGE_STD
        return {
            "pixel_values": image_tensor,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "semantic_labels": torch.from_numpy(np.asarray(semantic, dtype=np.int64)),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "building_instances": sum(1 for segment in label.segments_info if segment.ann_code == 10),
        }


def collate_panoptic_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch if isinstance(item["pixel_values"], torch.Tensor)]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "semantic_labels": torch.stack([item["semantic_labels"] for item in batch if isinstance(item["semantic_labels"], torch.Tensor)]),
        "image_path": [str(item["image_path"]) for item in batch],
        "label_path": [str(item["label_path"]) for item in batch],
        "building_instances": [int(item["building_instances"]) for item in batch],
    }


def make_panoptic_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataset_root = Path(args.dataset_root)
    train_dataset = PanopticSegmentationDataset(dataset_root / "train", args.image_size, args.limit)
    valid_dir = dataset_root / "valid"
    if has_labeled_split(valid_dir):
        valid_dataset = PanopticSegmentationDataset(valid_dir, args.image_size, args.limit)
    else:
        train_dataset, valid_dataset = split_dataset_for_validation(train_dataset, args)
    return (
        make_loader(train_dataset, args, shuffle=True, collate_fn=collate_panoptic_batch),
        make_loader(valid_dataset, args, shuffle=False, collate_fn=collate_panoptic_batch),
    )


def run_panoptic_epoch(
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
            mask_labels = [masks.to(device) for masks in batch["mask_labels"]]
            class_labels = [labels.to(device) for labels in batch["class_labels"]]
            outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Mask2Former did not produce a training loss.")

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.detach().cpu()))

    return {"loss": float(np.mean(losses)) if losses else 0.0}


def train_panoptic(args: argparse.Namespace) -> None:
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = str(OUTPUT_ROOT)
    if args.export_panoptic_only:
        path = prepare_panoptic_dataset(args, Path(args.panoptic_data_dir))
        print(f"Exported COCO panoptic dataset to {path}")
        return

    train_loader, valid_loader = make_panoptic_dataloaders(args)
    model_name = args.model_name_or_path
    if model_name == "nvidia/segformer-b0-finetuned-ade-512-512":
        model_name = DEFAULT_MODEL_NAME
    model_config = ModelConfig(architecture="mask2former", model_name_or_path=model_name, num_labels=NUM_CLASSES)
    device_ids = resolve_torch_device_ids(args.device)
    if len(device_ids) > 1:
        raise ValueError(
            "Mask2Former training does not support --device with multiple GPUs in this script. "
            "Use a single device such as --device 0, or add a torchrun/DDP training path."
        )
    model, device = build_torch_model_for_training(model_config, args.device)
    train_model_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        args=args,
        model_config=model_config,
        epoch_runner=run_panoptic_epoch,
        checkpoint_name="best_mask2former.pt",
        monitor_metric="loss",
        maximize=False,
        pretrained_subdir="mask2former_model",
    )


class RoadSegmentationDataset(Dataset):
    """Pixel-space road segmentation dataset.

    Current labels store EPSG:5186 geometries and older labels may store
    road_imcoords pixel polygons. Training masks are always generated in image
    pixel space; no distance, area, or CRS-dependent measurement is performed.
    """

    def __init__(self, split_dir: Path, image_size: int, limit: int | None = None, target_ann_codes: set[int] | None = None) -> None:
        self.samples = collect_samples(split_dir, limit)
        self.image_size = image_size
        self.target_ann_codes = target_ann_codes or {30}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, label_path = self.samples[index]
        image = load_rgb_image(image_path)
        mask = rasterize_road_mask(label_path, image.size, self.target_ann_codes)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - IMAGE_MEAN) / IMAGE_STD
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
            "image_path": str(image_path),
        }


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch if isinstance(item["pixel_values"], torch.Tensor)]),
        "labels": torch.stack([item["labels"] for item in batch if isinstance(item["labels"], torch.Tensor)]),
        "image_path": [str(item["image_path"]) for item in batch],
    }


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    dataset_root = Path(args.dataset_root)
    target_ann_codes = parse_ann_codes(args.target_ann_codes)
    train_dataset = RoadSegmentationDataset(dataset_root / "train", args.image_size, args.limit, target_ann_codes)

    valid_dir = dataset_root / "valid"
    if has_labeled_split(valid_dir):
        valid_dataset = RoadSegmentationDataset(valid_dir, args.image_size, args.limit, target_ann_codes)
    else:
        train_dataset, valid_dataset = split_dataset_for_validation(train_dataset, args)
    return (
        make_loader(train_dataset, args, shuffle=True, collate_fn=collate_batch),
        make_loader(valid_dataset, args, shuffle=False, collate_fn=collate_batch),
    )


def logits_from_model(model: nn.Module, architecture: str, images: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if architecture == "segformer":
        outputs = model(pixel_values=images, labels=labels)
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
    else:
        logits = model(images)
        loss = F.cross_entropy(logits, labels) if labels is not None else None

    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits, loss


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    road_pred = preds == 1
    road_true = labels == 1
    intersection = (road_pred & road_true).sum().item()
    union = (road_pred | road_true).sum().item()
    pred_sum = road_pred.sum().item()
    true_sum = road_true.sum().item()
    total = labels.numel()
    correct = (preds == labels).sum().item()
    return {
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) else 1.0,
        "pixel_accuracy": correct / total if total else 0.0,
    }


def average_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        raise RuntimeError("No metrics were produced; check that the dataloader contains at least one batch.")
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    architecture: str,
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
            logits, loss = logits_from_model(model, architecture, images, labels)
            if loss is None:
                raise RuntimeError("Training/evaluation loss was not produced.")

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.detach().cpu()))
            metric_items.append(compute_metrics(logits.detach(), labels.detach()))

    results = average_metrics(metric_items)
    results["loss"] = float(np.mean(losses))
    return results


def unwrapped_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_pretrained_model(model: nn.Module, output_dir: Path, subdir: str) -> None:
    model_to_save = unwrapped_model(model)
    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(output_dir / subdir)


def train_model_loop(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    model_config: ModelConfig,
    epoch_runner: Callable[[nn.Module, DataLoader, torch.device, torch.optim.Optimizer | None], dict[str, float]],
    checkpoint_name: str,
    monitor_metric: str,
    maximize: bool,
    pretrained_subdir: str | None = None,
) -> None:
    output_dir = ensure_output_dir(args.output_dir, "training output")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = -float("inf") if maximize else float("inf")
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_runner(model, train_loader, device, optimizer)
        valid_metrics = epoch_runner(model, valid_loader, device, None)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        current_value = valid_metrics[monitor_metric]
        improved = current_value > best_value if maximize else current_value < best_value
        if improved:
            best_value = current_value
            save_checkpoint(str(output_dir / checkpoint_name), model, model_config, args.image_size, valid_metrics)
            if pretrained_subdir:
                save_pretrained_model(model, output_dir, pretrained_subdir)

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def train_semantic(args: argparse.Namespace) -> None:
    train_loader, valid_loader = make_dataloaders(args)
    config = ModelConfig(architecture=args.architecture, model_name_or_path=args.model_name_or_path)
    model, device = build_torch_model_for_training(config, args.device)
    pretrained_subdir = "hf_model" if args.architecture == "segformer" else None
    semantic_epoch = lambda m, l, d, o=None: run_epoch(m, l, args.architecture, d, o)
    train_model_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        args=args,
        model_config=config,
        epoch_runner=semantic_epoch,
        checkpoint_name="best_model.pt",
        monitor_metric="iou",
        maximize=True,
        pretrained_subdir=pretrained_subdir,
    )


TRAIN_DISPATCH = {
    "segformer": train_semantic,
    "unet": train_semantic,
    "yolo": train_yolo,
    "mask2former": train_panoptic,
}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    TRAIN_DISPATCH[args.architecture](args)



if __name__ == "__main__":
    main()
