from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import IMAGE_EXTENSIONS, LABEL_CRS_EPSG, TRAIN_ID_TO_COLOR, TRAIN_ID_TO_NAME
from data import IMAGE_MEAN, IMAGE_STD, label_bounds, load_label


@dataclass(frozen=True)
class GsdConfig:
    x_m: float | None = None
    y_m: float | None = None

    @property
    def has_metric_area(self) -> bool:
        return self.x_m is not None and self.y_m is not None


def parse_gsd_args(args) -> GsdConfig:
    gsd_m = getattr(args, "gsd_m", None)
    gsd_x_m = getattr(args, "gsd_x_m", None)
    gsd_y_m = getattr(args, "gsd_y_m", None)
    if gsd_m is not None and (gsd_x_m is not None or gsd_y_m is not None):
        raise ValueError("Use either --gsd-m or both --gsd-x-m/--gsd-y-m, not both forms.")
    if gsd_m is not None:
        if gsd_m <= 0:
            raise ValueError("--gsd-m must be positive.")
        return GsdConfig(float(gsd_m), float(gsd_m))
    if gsd_x_m is None and gsd_y_m is None:
        return GsdConfig()
    if gsd_x_m is None or gsd_y_m is None:
        raise ValueError("Both --gsd-x-m and --gsd-y-m are required for non-square pixels.")
    if gsd_x_m <= 0 or gsd_y_m <= 0:
        raise ValueError("--gsd-x-m and --gsd-y-m must be positive.")
    return GsdConfig(float(gsd_x_m), float(gsd_y_m))


def require_output_crs(value: str | None) -> str:
    if not value or not str(value).strip():
        raise ValueError("--output-crs is required before GeoJSON output can be written, for example --output-crs EPSG:5186.")
    return str(value).strip()


def normalize_crs(value: str) -> str:
    return value.strip().upper().replace("::", ":")


def reference_label_for_image(image_path: Path, reference_label_root: str | Path | None) -> Path | None:
    if reference_label_root is None:
        return None
    root = Path(reference_label_root)
    if root.is_file():
        if root.stem != image_path.stem:
            raise ValueError(f"Reference label {root} does not match image stem {image_path.stem!r}.")
        return root
    if not root.exists():
        raise FileNotFoundError(f"--reference-label-root does not exist: {root}")
    matches = sorted(root.rglob(f"{image_path.stem}.json"))
    if not matches:
        raise FileNotFoundError(f"No reference GeoJSON label matching {image_path.stem}.json found under {root}.")
    return matches[0]


def infer_reference_label_root(image_path: Path) -> Path | None:
    parts = image_path.parts
    lowered = [part.lower() for part in parts]
    if "image" not in lowered:
        return None
    image_idx = len(lowered) - 1 - lowered[::-1].index("image")
    if image_idx == 0:
        return None
    candidate = Path(*parts[:image_idx]) / "label"
    return candidate if candidate.exists() else None


def transform_from_reference_label(image_path: Path, output_crs: str, reference_label_root: str | Path):
    from affine import Affine
    import rasterio

    requested = normalize_crs(output_crs)
    if requested != f"EPSG:{LABEL_CRS_EPSG}":
        raise ValueError(
            f"Reference labels are validated as EPSG:{LABEL_CRS_EPSG}, but --output-crs is {output_crs}. "
            "Use a georeferenced raster for other output CRS values."
        )
    label_path = reference_label_for_image(image_path, reference_label_root)
    if label_path is None:
        raise ValueError(f"{image_path} has no raster CRS and no --reference-label-root was provided.")
    label = load_label(label_path)
    min_x, min_y, max_x, max_y = label_bounds(label, label_path)
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
    if width <= 0 or height <= 0 or max_x <= min_x or max_y <= min_y:
        raise ValueError(f"{label_path} has invalid bounds for {image_path}.")
    x_res = (max_x - min_x) / width
    y_res = (max_y - min_y) / height
    return Affine(x_res, 0.0, min_x, 0.0, -y_res, max_y)


def raster_transform_for_output(image_path: Path, output_crs: str, reference_label_root: str | Path | None = None):
    try:
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning
    except ImportError as exc:
        raise ImportError("rasterio is required to write coordinate-aware GeoJSON outputs.") from exc
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(image_path) as src:
            if src.crs is None:
                reference_label_root = reference_label_root or infer_reference_label_root(image_path)
                if reference_label_root is not None:
                    return transform_from_reference_label(image_path, output_crs, reference_label_root)
                raise ValueError(
                    f"{image_path} has no raster CRS. Cannot write GeoJSON coordinates for {output_crs}. "
                    "Provide georeferenced TIFFs or pass --reference-label-root pointing to matching EPSG:5186 GeoJSON labels. "
                    "For the default dataset layout, expected a sibling label folder such as dataset/test/label."
                )
            src_crs = src.crs
            requested = normalize_crs(output_crs)
            requested_epsg = requested.removeprefix("EPSG:")
            src_epsg = src_crs.to_epsg()
            if src_epsg is not None and requested_epsg.isdigit() and int(requested_epsg) == src_epsg:
                return src.transform
            if normalize_crs(src_crs.to_string()) == requested:
                return src.transform
            raise ValueError(
                f"{image_path} CRS is {src_crs.to_string()}, but --output-crs is {output_crs}. "
                "Reproject the raster first or pass the matching CRS."
            )


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {path}. Expected .tif or .tiff.")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No TIFF images found under: {path}")
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
            area_px = int(component.sum())
            if area_px == 0:
                continue
            panoptic[component] = next_segment_id
            segments.append(
                {
                    "id": next_segment_id,
                    "category_id": train_id,
                    "train_id": train_id,
                    "class_name": TRAIN_ID_TO_NAME.get(train_id, str(train_id)),
                    "area_px": area_px,
                    "bbox": mask_bbox(component),
                }
            )
            next_segment_id += 1
    return panoptic, segments


def public_segment_properties(segment: dict[str, object], geometry: dict, output_crs: str, gsd: GsdConfig) -> dict[str, object]:
    area_px = int(segment.get("area_px", segment.get("area", 0)))
    class_id = int(segment.get("category_id", segment.get("train_id", 0)))
    props: dict[str, object] = {
        "id": int(segment["id"]),
        "class": str(segment.get("class_name", TRAIN_ID_TO_NAME.get(class_id, str(class_id)))),
        "class_id": class_id,
        "area_px": area_px,
        "coordinate_crs": output_crs,
        "coordinates": geometry.get("coordinates", []),
    }
    if gsd.has_metric_area:
        props["gsd_x_m"] = float(gsd.x_m)
        props["gsd_y_m"] = float(gsd.y_m)
        props["area_m2"] = float(area_px * float(gsd.x_m) * float(gsd.y_m))
    for score_key in ("confidence", "score"):
        if score_key in segment:
            props[score_key] = float(segment[score_key])
    return props


def polygonize_panoptic(
    panoptic: np.ndarray,
    segments: list[dict[str, object]],
    image_path: Path,
    output_crs: str,
    gsd: GsdConfig,
    reference_label_root: str | Path | None = None,
) -> dict[str, object]:
    from rasterio.features import shapes

    transform = raster_transform_for_output(image_path, output_crs, reference_label_root)
    by_id = {int(segment["id"]): segment for segment in segments}
    features: list[dict[str, object]] = []
    for geometry, value in shapes(panoptic.astype(np.int32), mask=panoptic > 0, transform=transform):
        segment = by_id.get(int(value))
        if segment is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": public_segment_properties(segment, geometry, output_crs, gsd),
                "geometry": geometry,
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": output_crs}},
        "properties": {
            "coordinate_crs": output_crs,
            "area_unit": "m2" if gsd.has_metric_area else None,
            "area_note": "area_m2 is computed from GSD when GSD is provided; area_px is always pixel count.",
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
    output_crs: str,
    gsd: GsdConfig,
    reference_label_root: str | Path | None = None,
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
    geojson = polygonize_panoptic(panoptic, segments, image_path, output_crs, gsd, reference_label_root)
    geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    return {
        "image": str(image_path),
        "semantic_mask": str(mask_path),
        "panoptic_mask": str(panoptic_path),
        "color_mask": str(color_path),
        "overlay": str(overlay_path),
        "geojson": str(geojson_path),
        "pixel_counts": class_summary(semantic),
        "coordinate_crs": output_crs,
    }
