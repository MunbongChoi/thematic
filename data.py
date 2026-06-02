from __future__ import annotations

import json
import random
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from config import (
    ANALYSIS_CRS_EPSG,
    ANN_CODE_FIELD,
    ANN_CODE_TO_MODEL_ID,
    ANN_CODE_TO_TRAIN_ID,
    BACKGROUND_ID,
    CLASSES,
    FEATURES_FIELD,
    GEOMETRY_FIELD,
    GEOMETRY_TYPES,
    IMAGE_EXTENSIONS,
    LABEL_CRS_EPSG,
    PROPERTIES_FIELD,
    RGB_RASTER_EXTENSIONS,
    TRAIN_ID_TO_NAME,
)

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LabelSchemaError(ValueError):
    pass


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
    if array.ndim != 3:
        raise ValueError(f"Expected RGB array with shape HxWxC, got {array.shape}.")
    if array.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 image channels for RGB input, got {array.shape[-1]} channel(s).")
    if array.shape[-1] > 3:
        array = array[..., :3]
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
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported source image extension for {image_path}. Expected .tif or .tiff.")
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError as exc:
        raise ImportError("rasterio is required to read TIFF inputs. Install requirements.txt.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(image_path) as src:
            if src.count < 3:
                raise ValueError(f"{image_path} is expected to be RGB but has {src.count} band(s).")
            bands = src.read([1, 2, 3])
    return Image.fromarray(array_to_uint8_rgb(np.moveaxis(bands, 0, -1)), mode="RGB")


@lru_cache(maxsize=4096)
def image_size(image_path: Path) -> tuple[int, int]:
    if image_path.suffix.lower() not in RGB_RASTER_EXTENSIONS:
        raise ValueError(f"Unsupported source image extension for {image_path}. Expected .tif or .tiff.")
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError as exc:
        raise ImportError("rasterio is required to read TIFF inputs. Install requirements.txt.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(image_path) as src:
            return src.width, src.height


@lru_cache(maxsize=4096)
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
    min_x = min_y = float("inf")
    max_x = max_y = -float("inf")
    has_coordinate = False
    for feature in data.get(FEATURES_FIELD, []):
        geometry = feature.get(GEOMETRY_FIELD, {})
        if geometry.get("type") not in GEOMETRY_TYPES:
            raise LabelSchemaError(f"{label_path} has unsupported geometry type {geometry.get('type')!r}.")
        for ring in iter_raw_rings(geometry):
            for point in ring:
                if len(point) >= 2:
                    x = float(point[0])
                    y = float(point[1])
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    has_coordinate = True
    if not has_coordinate:
        raise LabelSchemaError(f"{label_path} does not contain polygon coordinates.")
    return min_x, min_y, max_x, max_y


def geo_to_pixel_transform(data: dict, label_path: Path, size: tuple[int, int]) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = label_bounds(data, label_path)
    width, height = size
    if width <= 0 or height <= 0 or max_x <= min_x or max_y <= min_y:
        raise LabelSchemaError(f"{label_path} has invalid image size or label bounds.")
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
    coordinates = np.asarray([point[:2] for point in ring if len(point) >= 2], dtype=np.float64)
    if coordinates.size == 0:
        return []
    x = np.clip((coordinates[:, 0] - min_x) / x_res, 0.0, width - 1.0)
    y = np.clip((max_y - coordinates[:, 1]) / y_res, 0.0, height - 1.0)
    return list(zip(x.tolist(), y.tolist()))


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


def render_instance_targets(label_path: Path, size: tuple[int, int], output_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = load_label(label_path)
    transform = geo_to_pixel_transform(data, label_path, size)
    masks: list[torch.Tensor] = []
    class_labels: list[int] = []
    for feature_idx, feature in enumerate(data.get(FEATURES_FIELD, []), start=1):
        ann_code = feature_ann_code(feature, label_path, feature_idx)
        polygons = polygons_from_geometry(feature.get(GEOMETRY_FIELD, {}), transform, size)
        if not polygons:
            continue
        mask = rasterize_feature_mask(polygons, size).astype(np.uint8)
        if not mask.any():
            continue
        mask_image = Image.fromarray(mask, mode="L").resize((output_size, output_size), Image.NEAREST)
        masks.append(torch.from_numpy(np.asarray(mask_image, dtype=np.float32)))
        class_labels.append(ANN_CODE_TO_MODEL_ID[ann_code])
    if not masks:
        return (
            torch.zeros((0, output_size, output_size), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )
    return torch.stack(masks), torch.tensor(class_labels, dtype=torch.long)


def build_image_index(image_dir: Path) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        by_stem.setdefault(image_path.stem, image_path)
    return by_stem


def find_image_for_label(label_path: Path, image_index: dict[str, Path]) -> Path:
    if label_path.stem in image_index:
        return image_index[label_path.stem]
    raise FileNotFoundError(f"No TIFF image matched label: {label_path}")


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


def split_samples(args) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
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
        mask_labels, class_labels = render_instance_targets(label_path, image.size, self.image_size)
        resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        return {
            "pixel_values": (image_tensor - IMAGE_MEAN) / IMAGE_STD,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
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
        "image_path": [str(item["image_path"]) for item in batch],
        "label_path": [str(item["label_path"]) for item in batch],
    }


def make_loader(dataset: Dataset, args, shuffle: bool, collate_fn: Callable) -> DataLoader:
    use_workers = args.num_workers > 0
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": use_workers,
    }
    if use_workers:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def make_semantic_loaders(args) -> tuple[DataLoader, DataLoader]:
    train_samples, valid_samples = split_samples(args)
    return (
        make_loader(SemanticSegmentationDataset(train_samples, args.image_size), args, True, collate_semantic),
        make_loader(SemanticSegmentationDataset(valid_samples, args.image_size), args, False, collate_semantic),
    )


def make_instance_loaders(args) -> tuple[DataLoader, DataLoader]:
    train_samples, valid_samples = split_samples(args)
    return (
        make_loader(InstanceSegmentationDataset(train_samples, args.image_size), args, True, collate_instance),
        make_loader(InstanceSegmentationDataset(valid_samples, args.image_size), args, False, collate_instance),
    )


def average_metrics(items: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(items)
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def export_semantic_masks(args) -> None:
    from tqdm import tqdm

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


def write_prepared_metadata(prepared_dir: str | Path) -> None:
    root = ensure_dir(prepared_dir)
    metadata = {
        "classes": [item.__dict__ for item in CLASSES],
        "label_crs": f"EPSG:{LABEL_CRS_EPSG}",
        "output_crs": "pixel",
        "spatial_operation": "pixel-space rasterization only",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
