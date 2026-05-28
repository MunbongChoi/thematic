from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = Path("runs") / "segmentation"
PREPARED_ROOT = Path("outputs") / "prepared"

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
RASTER_EXTENSIONS = {".tif", ".tiff"}
LABEL_EXTENSION = ".json"

# Labels are GeoJSON polygons in Korea 2000 Central Belt coordinates.
# The training code maps CRS coordinates to tile pixel space only; it does not
# compute distance, area, density, or buffer values from these coordinates.
LABEL_CRS_EPSG = 5186
ANALYSIS_CRS_EPSG = 5186
OUTPUT_CRS = "pixel"
GEOMETRY_FIELD = "geometry"
GEOMETRY_TYPES = {"Polygon", "MultiPolygon"}
FEATURES_FIELD = "features"
PROPERTIES_FIELD = "properties"
ANN_CODE_FIELD = "ANN_CD"

BACKGROUND_ID = 0
IGNORE_INDEX = 255
DEFAULT_IMAGE_SIZE = 512
DEFAULT_SEED = 42

DEFAULT_YOLO_MODEL = "yolo11n-seg.pt"
DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_MASK2FORMER_MODEL = "facebook/mask2former-swin-tiny-coco-panoptic"


@dataclass(frozen=True)
class SegmentationClass:
    train_id: int
    yolo_id: int
    name: str
    korean_name: str
    ann_codes: tuple[int, ...]
    color: tuple[int, int, int]
    isthing: bool = False


CLASSES: tuple[SegmentationClass, ...] = (
    SegmentationClass(1, 0, "building", "building", (10,), (220, 70, 70), True),
    SegmentationClass(2, 1, "parking_lot", "parking_lot", (20,), (235, 170, 60)),
    SegmentationClass(3, 2, "road", "road", (30,), (90, 90, 90)),
    SegmentationClass(4, 3, "street_tree", "street_tree", (40,), (80, 170, 80)),
    SegmentationClass(5, 4, "paddy_field", "paddy_field", (50,), (100, 180, 90)),
    SegmentationClass(6, 5, "greenhouse", "greenhouse", (55,), (120, 200, 185)),
    SegmentationClass(7, 6, "field", "field", (60,), (185, 155, 110)),
    SegmentationClass(8, 7, "broadleaf_forest", "broadleaf_forest", (71,), (45, 135, 70)),
    SegmentationClass(9, 8, "coniferous_forest", "coniferous_forest", (75,), (30, 110, 80)),
    SegmentationClass(10, 9, "bare_ground", "bare_ground", (80,), (175, 155, 135)),
    SegmentationClass(11, 10, "water", "water", (95,), (65, 120, 200)),
    SegmentationClass(12, 11, "non_cultivated", "non_cultivated", (100,), (125, 125, 145)),
)

ANN_CODE_TO_TRAIN_ID = {ann_code: item.train_id for item in CLASSES for ann_code in item.ann_codes}
ANN_CODE_TO_YOLO_ID = {ann_code: item.yolo_id for item in CLASSES for ann_code in item.ann_codes}
TRAIN_ID_TO_NAME = {BACKGROUND_ID: "background"} | {item.train_id: item.name for item in CLASSES}
NAME_TO_TRAIN_ID = {name: idx for idx, name in TRAIN_ID_TO_NAME.items()}
TRAIN_ID_TO_COLOR = {BACKGROUND_ID: (0, 0, 0)} | {item.train_id: item.color for item in CLASSES}
YOLO_ID_TO_NAME = {item.yolo_id: item.name for item in CLASSES}
YOLO_ID_TO_TRAIN_ID = {item.yolo_id: item.train_id for item in CLASSES}
MASK2FORMER_ID_TO_NAME = {item.yolo_id: item.name for item in CLASSES}
MASK2FORMER_NAME_TO_ID = {item.name: item.yolo_id for item in CLASSES}
NUM_SEMANTIC_CLASSES = len(TRAIN_ID_TO_NAME)
NUM_SEGMENT_CLASSES = len(CLASSES)

