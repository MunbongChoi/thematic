from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "satellite_panoptic_segmentation"

DATASET_ROOT = Path("dataset")
TRAIN_IMAGE_DIR = DATASET_ROOT / "train" / "image"
TRAIN_LABEL_DIR = DATASET_ROOT / "train" / "label"
VALID_IMAGE_DIR = DATASET_ROOT / "valid" / "image"
VALID_LABEL_DIR = DATASET_ROOT / "valid" / "label"

OUTPUT_ROOT = Path("runs") / "panoptic_segmentation"
PANOPTIC_DATASET_DIR = Path("outputs") / "panoptic_dataset"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RASTER_EXTENSIONS = {".tif", ".tiff"}
LABEL_EXTENSION = ".json"

# The changed labels store map-space GeoJSON geometries in EPSG:5186.
# Source TIFs in this workspace do not expose geotransforms, so later dataset
# code should map coordinates into pixel space from the label tile bounds.
LABEL_CRS_EPSG = 5186
ANALYSIS_CRS_EPSG = 5186
OUTPUT_CRS_EPSG = 4326
GEOMETRY_FIELD = "geometry"
GEOMETRY_TYPES = {"Polygon", "MultiPolygon"}
LABEL_FEATURES_FIELD = "features"
LABEL_PROPERTIES_FIELD = "properties"
ANN_CODE_FIELD = "ANN_CD"

PANOPTIC_VOID_LABEL = 255
BACKGROUND_LABEL = 0
DEFAULT_IMAGE_SIZE = 512
DEFAULT_SEED = 42

DEFAULT_MODEL_NAME = "facebook/mask2former-swin-tiny-coco-panoptic"
DEFAULT_ARCHITECTURE = "mask2former"


@dataclass(frozen=True)
class PanopticCategory:
    id: int
    train_id: int
    name: str
    ann_codes: tuple[int, ...]
    isthing: bool
    color: tuple[int, int, int]


PANOPTIC_CATEGORIES: tuple[PanopticCategory, ...] = (
    PanopticCategory(1, 1, "building", (10,), True, (220, 70, 70)),
    PanopticCategory(2, 2, "parking_lot", (20,), False, (235, 170, 60)),
    PanopticCategory(3, 3, "road", (30,), False, (90, 90, 90)),
    PanopticCategory(4, 4, "street_tree", (40,), False, (80, 170, 80)),
    PanopticCategory(5, 5, "paddy_field", (50,), False, (100, 180, 90)),
    PanopticCategory(6, 6, "greenhouse", (55,), False, (120, 200, 185)),
    PanopticCategory(7, 7, "field", (60,), False, (185, 155, 110)),
    PanopticCategory(8, 8, "broadleaf_forest", (71,), False, (45, 135, 70)),
    PanopticCategory(9, 9, "coniferous_forest", (75,), False, (30, 110, 80)),
    PanopticCategory(10, 10, "bare_ground", (80,), False, (175, 155, 135)),
    PanopticCategory(11, 11, "water", (95,), False, (65, 120, 200)),
    PanopticCategory(12, 12, "non_target", (100,), False, (125, 125, 145)),
)

ANN_CODE_TO_CATEGORY_ID = {
    ann_code: category.id
    for category in PANOPTIC_CATEGORIES
    for ann_code in category.ann_codes
}
CATEGORY_ID_TO_TRAIN_ID = {category.id: category.train_id for category in PANOPTIC_CATEGORIES}
CATEGORY_ID_TO_NAME = {category.id: category.name for category in PANOPTIC_CATEGORIES}
ID2LABEL = {BACKGROUND_LABEL: "background"} | {
    category.train_id: category.name for category in PANOPTIC_CATEGORIES
}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
NUM_CLASSES = len(ID2LABEL)


def panoptic_categories_as_coco() -> list[dict[str, object]]:
    return [
        {
            "id": category.id,
            "name": category.name,
            "isthing": int(category.isthing),
            "color": list(category.color),
        }
        for category in PANOPTIC_CATEGORIES
    ]
