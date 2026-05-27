from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from config import (
    ANN_CODE_FIELD,
    ANN_CODE_TO_CATEGORY_ID,
    BACKGROUND_LABEL,
    CATEGORY_ID_TO_NAME,
    CATEGORY_ID_TO_TRAIN_ID,
    GEOMETRY_FIELD,
    GEOMETRY_TYPES,
    LABEL_CRS_EPSG,
    LABEL_FEATURES_FIELD,
    LABEL_PROPERTIES_FIELD,
    PANOPTIC_CATEGORIES,
)

@dataclass(frozen=True)
class SegmentInfo:
    id: int
    category_id: int
    train_id: int
    ann_code: int
    isthing: bool
    area: int
    bbox: list[int]


@dataclass(frozen=True)
class PanopticLabel:
    semantic_mask: np.ndarray
    panoptic_id_mask: np.ndarray
    segments_info: list[SegmentInfo]


class PanopticLabelError(ValueError):
    pass


def id_to_rgb(segment_id: int) -> tuple[int, int, int]:
    return segment_id % 256, (segment_id // 256) % 256, (segment_id // 65536) % 256


def rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0].astype(np.int64) + 256 * rgb[:, :, 1].astype(np.int64) + 65536 * rgb[:, :, 2].astype(np.int64)


def panoptic_id_to_rgb(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = mask % 256
    rgb[:, :, 1] = (mask // 256) % 256
    rgb[:, :, 2] = (mask // 65536) % 256
    return Image.fromarray(rgb, mode="RGB")


def segment_infos_to_json(segments_info: Iterable[SegmentInfo]) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "category_id": item.category_id,
            "train_id": item.train_id,
            "ann_code": item.ann_code,
            "isthing": int(item.isthing),
            "area": item.area,
            "bbox": item.bbox,
        }
        for item in segments_info
    ]


def validate_label_crs(data: dict, label_path: Path) -> None:
    crs = data.get("crs")
    if not crs:
        raise PanopticLabelError(f"{label_path} is missing CRS metadata. Expected EPSG:{LABEL_CRS_EPSG} for geometry labels.")
    name = str(crs.get("properties", {}).get("name", ""))
    if f"EPSG::{LABEL_CRS_EPSG}" not in name and f"EPSG:{LABEL_CRS_EPSG}" not in name:
        raise PanopticLabelError(f"{label_path} CRS must be EPSG:{LABEL_CRS_EPSG}, got {name!r}.")


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
    for feature in data.get(LABEL_FEATURES_FIELD, []):
        for ring in iter_polygon_rings(feature.get(GEOMETRY_FIELD, {})):
            for point in ring:
                if len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
    if not xs or not ys:
        raise PanopticLabelError("GeoJSON label does not contain polygon coordinates.")
    return min(xs), min(ys), max(xs), max(ys)


def geo_to_pixel_transform(data: dict, size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = size
    min_x, min_y, max_x, max_y = label_bounds(data)
    if max_x <= min_x or max_y <= min_y:
        raise PanopticLabelError("Invalid GeoJSON label bounds.")
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
        raise PanopticLabelError("Invalid geometry could not be repaired.")
    return mapping(repaired)


def polygons_from_geometry(
    geometry: dict,
    transform: tuple[float, float, float, float],
    size: tuple[int, int],
) -> list[list[list[tuple[float, float]]]]:
    geometry = repaired_geometry(geometry)
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        return []

    pixel_polygons: list[list[list[tuple[float, float]]]] = []
    for polygon in polygons:
        rings: list[list[tuple[float, float]]] = []
        for ring in polygon:
            pixel_ring = geo_ring_to_pixels(ring, transform, size)
            if len(pixel_ring) >= 3:
                rings.append(pixel_ring)
        if rings:
            pixel_polygons.append(rings)
    return pixel_polygons


def rasterize_polygons(polygons: list[list[list[tuple[float, float]]]], size: tuple[int, int]) -> np.ndarray:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        exterior = polygon[0]
        draw.polygon(exterior, fill=1)
        for hole in polygon[1:]:
            draw.polygon(hole, fill=0)
    return np.asarray(mask, dtype=bool)


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    return [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]


def render_panoptic_label(label_path: Path, size: tuple[int, int]) -> PanopticLabel:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    validate_label_crs(data, label_path)
    width, height = size
    semantic = np.full((height, width), BACKGROUND_LABEL, dtype=np.int64)
    panoptic = np.zeros((height, width), dtype=np.int32)
    segments_info: list[SegmentInfo] = []
    transform = geo_to_pixel_transform(data, size)
    thing_by_category = {category.id: category.isthing for category in PANOPTIC_CATEGORIES}
    stuff_masks: dict[int, np.ndarray] = {}
    stuff_ann_codes: dict[int, int] = {}
    next_segment_id = 1

    for feature_idx, feature in enumerate(data.get(LABEL_FEATURES_FIELD, []), start=1):
        properties = feature.get(LABEL_PROPERTIES_FIELD, {})
        raw_ann_code = properties.get(ANN_CODE_FIELD)
        if raw_ann_code is None:
            raise PanopticLabelError(f"{label_path} feature {feature_idx} is missing {ANN_CODE_FIELD}.")
        ann_code = int(raw_ann_code)
        if ann_code not in ANN_CODE_TO_CATEGORY_ID:
            raise PanopticLabelError(f"{label_path} contains unmapped ANN_CD={ann_code}.")

        category_id = ANN_CODE_TO_CATEGORY_ID[ann_code]
        polygons = polygons_from_geometry(feature.get(GEOMETRY_FIELD, {}), transform, size)
        if not polygons:
            continue
        mask = rasterize_polygons(polygons, size)
        if not mask.any():
            continue

        if thing_by_category[category_id]:
            visible = mask & (panoptic == 0)
            area = int(visible.sum())
            if area == 0:
                continue
            train_id = CATEGORY_ID_TO_TRAIN_ID[category_id]
            segment_id = next_segment_id
            next_segment_id += 1
            semantic[visible] = train_id
            panoptic[visible] = segment_id
            segments_info.append(
                SegmentInfo(
                    id=segment_id,
                    category_id=category_id,
                    train_id=train_id,
                    ann_code=ann_code,
                    isthing=True,
                    area=area,
                    bbox=mask_bbox(visible),
                )
            )
        else:
            if category_id not in stuff_masks:
                stuff_masks[category_id] = mask.copy()
                stuff_ann_codes[category_id] = ann_code
            else:
                stuff_masks[category_id] |= mask

    for category_id, mask in stuff_masks.items():
        visible = mask & (panoptic == 0)
        area = int(visible.sum())
        if area == 0:
            continue
        train_id = CATEGORY_ID_TO_TRAIN_ID[category_id]
        segment_id = next_segment_id
        next_segment_id += 1
        semantic[visible] = train_id
        panoptic[visible] = segment_id
        segments_info.append(
            SegmentInfo(
                id=segment_id,
                category_id=category_id,
                train_id=train_id,
                ann_code=stuff_ann_codes[category_id],
                isthing=False,
                area=area,
                bbox=mask_bbox(visible),
            )
        )

    return PanopticLabel(semantic_mask=semantic, panoptic_id_mask=panoptic, segments_info=segments_info)


def masks_and_classes_from_panoptic(label: PanopticLabel, output_size: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    masks: list[torch.Tensor] = []
    class_labels: list[int] = []
    for segment in label.segments_info:
        mask = Image.fromarray((label.panoptic_id_mask == segment.id).astype(np.uint8), mode="L")
        if output_size is not None:
            mask = mask.resize(output_size, Image.NEAREST)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))
        if mask_tensor.sum() == 0:
            continue
        masks.append(mask_tensor)
        class_labels.append(segment.train_id)
    if not class_labels:
        if output_size is None:
            height, width = label.semantic_mask.shape
            output_size = (width, height)
        width, height = output_size
        return torch.zeros((0, height, width), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.stack(masks), torch.tensor(class_labels, dtype=torch.long)


def category_summary(segments_info: Iterable[SegmentInfo]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for segment in segments_info:
        name = CATEGORY_ID_TO_NAME[segment.category_id]
        summary[name] = summary.get(name, 0) + 1
    return summary
