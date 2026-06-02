from __future__ import annotations

from pathlib import Path
from typing import Any

from config import DEFAULT_YOLO_MODEL


def build_yolo_model(model_name_or_path: str | Path | None = None) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for YOLO segmentation. Install requirements.txt.") from exc
    return YOLO(str(model_name_or_path or DEFAULT_YOLO_MODEL))
