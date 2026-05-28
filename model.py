from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# This Windows environment fails OpenMP initialization when torch is loaded
# before numpy, while the training/inference entrypoints already load numpy first.
import numpy as np  # noqa: F401
import torch
from torch import nn

from config import DEFAULT_MODEL_NAME, ID2LABEL as PANOPTIC_LABELS, LABEL2ID as PANOPTIC_LABEL2ID, NUM_CLASSES


ROAD_LABELS = {0: "background", 1: "road"}
DEFAULT_YOLO_SEG_MODEL = "yolo11n-seg.pt"
CHECKPOINT_CANDIDATES = (
    "best_model.pt",
    "best_sam.pt",
    "best_mask2former.pt",
    "best_yolo.pt",
    "yolo/weights/best.pt",
)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "segformer"
    model_name_or_path: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_labels: int = 2


def cuda_diagnostic_message(requested_device: str) -> str:
    details = [
        f"CUDA device {requested_device!r} was requested, but PyTorch cannot use CUDA.",
        f"torch={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
        f"torch.cuda.is_available()={torch.cuda.is_available()}",
        f"torch.cuda.device_count()={torch.cuda.device_count()}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        details.append(f"nvidia-smi=unavailable ({exc})")
    else:
        smi_output = result.stdout.strip() or result.stderr.strip()
        details.append(f"nvidia-smi={smi_output if smi_output else 'no output'}")

    details.append(
        "If nvidia-smi shows a GPU but torch.version.cuda is None or "
        "torch.cuda.is_available() is False, install a CUDA-enabled PyTorch "
        "build in this same Python environment."
    )
    return "\n".join(details)


def resolve_torch_device_ids(device: str | None = None) -> list[int]:
    normalized = "" if device is None else str(device).strip().lower()
    if normalized in {"", "cuda"}:
        return [0] if torch.cuda.is_available() else []
    if normalized == "cpu":
        return []
    raw_ids = normalized.replace("cuda:", "").split(",")
    device_ids: list[int] = []
    for raw_id in raw_ids:
        raw_id = raw_id.strip()
        if not raw_id:
            continue
        try:
            device_ids.append(int(raw_id))
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device list: {device!r}. Use '0' or '0,1,2,3'.") from exc
    if device_ids and not torch.cuda.is_available():
        raise RuntimeError(cuda_diagnostic_message(str(device)))
    available = torch.cuda.device_count()
    invalid = [idx for idx in device_ids if idx < 0 or idx >= available]
    if invalid:
        raise RuntimeError(f"CUDA device index(es) {invalid} unavailable. Found {available} CUDA device(s).")
    return device_ids


def resolve_torch_device(device: str | None = None) -> torch.device:
    normalized = "" if device is None else str(device).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized and not (normalized == "cuda" or normalized.startswith("cuda:") or normalized[0].isdigit()):
        return torch.device(normalized)

    device_ids = resolve_torch_device_ids(device)
    if device_ids:
        return torch.device(f"cuda:{device_ids[0]}")
    if normalized:
        raise RuntimeError(cuda_diagnostic_message(normalized))
    return torch.device("cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_labels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_labels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        x = self.bottleneck(self.pool(enc3))
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        return self.head(x)


def build_model(config: ModelConfig) -> nn.Module:
    architecture = config.architecture.lower()
    if architecture == "unet":
        return UNet(num_labels=config.num_labels)
    if architecture in {"mask2former", "panoptic"}:
        return build_mask2former_model(config.model_name_or_path, config.num_labels)
    if architecture == "segformer":
        try:
            from transformers import AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for architecture='segformer'. "
                "Install requirements.txt or use --architecture unet."
            ) from exc

        return AutoModelForSemanticSegmentation.from_pretrained(
            config.model_name_or_path,
            num_labels=config.num_labels,
            id2label=ROAD_LABELS,
            label2id={label: idx for idx, label in ROAD_LABELS.items()},
            ignore_mismatched_sizes=True,
        )
    if architecture == "yolo":
        raise ValueError("Use build_yolo_model() for architecture='yolo'.")
    raise ValueError(f"Unsupported architecture: {config.architecture}")


def build_mask2former_model(model_name_or_path: str = DEFAULT_MODEL_NAME, num_labels: int = NUM_CLASSES) -> nn.Module:
    try:
        from transformers import Mask2FormerForUniversalSegmentation
    except ImportError as exc:
        raise ImportError(
            "transformers is required for architecture='mask2former'. "
            "Install requirements.txt first."
        ) from exc

    return Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=PANOPTIC_LABELS,
        label2id=PANOPTIC_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

def build_mask2former_processor(model_name_or_path: str = DEFAULT_MODEL_NAME) -> Any:
    try:
        from transformers import Mask2FormerImageProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for Mask2Former image processing. "
            "Install requirements.txt first."
        ) from exc
    return Mask2FormerImageProcessor.from_pretrained(model_name_or_path)


def build_yolo_model(model_name_or_path: str = DEFAULT_YOLO_SEG_MODEL) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for architecture='yolo'. "
            "Install requirements.txt first."
        ) from exc
    return YOLO(str(resolve_checkpoint_path(model_name_or_path, required=False)))


def resolve_checkpoint_path(path: str | Path, required: bool = True) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        for candidate in CHECKPOINT_CANDIDATES:
            candidate_path = checkpoint_path / candidate
            if candidate_path.is_file():
                return candidate_path
        candidates = ", ".join(CHECKPOINT_CANDIDATES)
        raise FileNotFoundError(
            f"Checkpoint path points to a directory: {checkpoint_path}. "
            f"Pass a checkpoint file directly, or place one of these files inside it: {candidates}"
        )
    if required and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def save_checkpoint(
    path: str,
    model: nn.Module,
    config: ModelConfig,
    image_size: int,
    metrics: dict[str, float] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    checkpoint = {
        "architecture": config.architecture,
        "model_name_or_path": config.model_name_or_path,
        "num_labels": config.num_labels,
        "image_size": image_size,
        "state_dict": model_to_save.state_dict(),
        "id2label": PANOPTIC_LABELS if config.architecture in {"mask2former", "panoptic"} else ROAD_LABELS,
        "metrics": metrics or {},
    }
    try:
        torch.save(checkpoint, temp_path)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to write checkpoint temporary file: {temp_path.resolve()}\n"
            f"Target checkpoint: {checkpoint_path.resolve()}\n"
            "The output directory is not writable by this Python process, "
            "the filesystem is full, or the path is on a restricted/mounted volume. "
            "Use a writable absolute --output-dir such as /tmp/satellite_runs/road_extraction "
            "or /home/jovyan/work/thematic/runs/road_extraction."
        ) from exc
    try:
        temp_path.replace(checkpoint_path)
    except OSError:
        try:
            shutil.copy2(temp_path, checkpoint_path)
        except OSError as exc:
            raise RuntimeError(
                f"Checkpoint was written to temporary file but could not be moved to: {checkpoint_path.resolve()}\n"
                "Check write permission, file locks, available disk space, and mounted filesystem behavior."
            ) from exc
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint_path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to open checkpoint: {checkpoint_path.resolve()}. "
            "Check that the file exists, is not empty, and is readable. "
            "If training was interrupted while saving, delete the partial file and train again."
        ) from exc
    config = ModelConfig(
        architecture=checkpoint["architecture"],
        model_name_or_path=checkpoint["model_name_or_path"],
        num_labels=int(checkpoint.get("num_labels", 2)),
    )
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
