from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np  # noqa: F401
import torch
from torch import nn

from config import (
    DEFAULT_MASK2FORMER_MODEL,
    DEFAULT_SEGFORMER_MODEL,
    DEFAULT_YOLO_MODEL,
    MASK2FORMER_ID_TO_NAME,
    MASK2FORMER_NAME_TO_ID,
    NUM_SEGMENT_CLASSES,
    NUM_SEMANTIC_CLASSES,
    TRAIN_ID_TO_NAME,
)


SUPPORTED_ARCHITECTURES = ("yolo", "unet", "segformer", "mask2former")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    model_name_or_path: str | None = None
    num_labels: int | None = None

    def normalized(self) -> "ModelConfig":
        architecture = self.architecture.strip().lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            choices = ", ".join(SUPPORTED_ARCHITECTURES)
            raise ValueError(f"Unsupported architecture {self.architecture!r}. Choose one of: {choices}")
        if architecture == "yolo":
            model_name = self.model_name_or_path or DEFAULT_YOLO_MODEL
            num_labels = NUM_SEGMENT_CLASSES
        elif architecture == "mask2former":
            model_name = self.model_name_or_path or DEFAULT_MASK2FORMER_MODEL
            num_labels = NUM_SEGMENT_CLASSES
        elif architecture == "segformer":
            model_name = self.model_name_or_path or DEFAULT_SEGFORMER_MODEL
            num_labels = NUM_SEMANTIC_CLASSES
        else:
            model_name = self.model_name_or_path or "unet"
            num_labels = NUM_SEMANTIC_CLASSES
        return ModelConfig(architecture=architecture, model_name_or_path=model_name, num_labels=num_labels)


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
        output = result.stdout.strip() or result.stderr.strip()
        details.append(f"nvidia-smi={output if output else 'no output'}")
    details.append("Install a CUDA-enabled PyTorch build in this same environment if GPU training is required.")
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
    def __init__(self, in_channels: int = 3, num_labels: int = NUM_SEMANTIC_CLASSES, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
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
        enc4 = self.enc4(self.pool(enc3))
        x = self.bottleneck(self.pool(enc4))
        x = self.dec4(torch.cat([self.up4(x), enc4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), enc3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), enc2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), enc1], dim=1))
        return self.head(x)


def build_model(config: ModelConfig) -> nn.Module:
    cfg = config.normalized()
    if cfg.architecture == "unet":
        return UNet(num_labels=int(cfg.num_labels or NUM_SEMANTIC_CLASSES))
    if cfg.architecture == "segformer":
        try:
            from transformers import AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("transformers is required for SegFormer. Install requirements.txt.") from exc
        return AutoModelForSemanticSegmentation.from_pretrained(
            str(cfg.model_name_or_path),
            num_labels=int(cfg.num_labels or NUM_SEMANTIC_CLASSES),
            id2label=TRAIN_ID_TO_NAME,
            label2id={name: idx for idx, name in TRAIN_ID_TO_NAME.items()},
            ignore_mismatched_sizes=True,
        )
    if cfg.architecture == "mask2former":
        try:
            from transformers import Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise ImportError("transformers is required for Mask2Former. Install requirements.txt.") from exc
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            str(cfg.model_name_or_path),
            num_labels=int(cfg.num_labels or NUM_SEGMENT_CLASSES),
            id2label=MASK2FORMER_ID_TO_NAME,
            label2id=MASK2FORMER_NAME_TO_ID,
            ignore_mismatched_sizes=True,
        )
    raise ValueError("Use build_yolo_model() for architecture='yolo'.")


def build_mask2former_processor(model_name_or_path: str | None = None) -> Any:
    try:
        from transformers import Mask2FormerImageProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for Mask2Former inference. Install requirements.txt.") from exc
    return Mask2FormerImageProcessor.from_pretrained(model_name_or_path or DEFAULT_MASK2FORMER_MODEL)


def build_yolo_model(model_name_or_path: str | Path | None = None) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for YOLO segmentation. Install requirements.txt.") from exc
    return YOLO(str(model_name_or_path or DEFAULT_YOLO_MODEL))


class Mask2FormerDataParallel(nn.Module):
    """DataParallel wrapper that keeps per-image mask targets aligned.

    Hugging Face Mask2Former receives `mask_labels` and `class_labels` as lists
    with one item per image. PyTorch's generic DataParallel recursively scatters
    list items, which can split an instance-mask tensor along the wrong axis.
    This wrapper slices those lists by image batch before dispatching replicas.
    """

    def __init__(self, module: nn.Module, device_ids: list[int], output_device: int | None = None) -> None:
        super().__init__()
        if len(device_ids) < 2:
            raise ValueError("Mask2FormerDataParallel requires at least two CUDA devices.")
        self.module = module
        self.device_ids = device_ids
        self.output_device = device_ids[0] if output_device is None else output_device

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.device_ids or pixel_values.size(0) == 0:
            return self.module(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels, **kwargs)

        active_device_ids = self.device_ids[: min(len(self.device_ids), pixel_values.size(0))]
        chunks = list(torch.chunk(pixel_values, len(active_device_ids), dim=0))
        replicas = nn.parallel.replicate(self.module, active_device_ids)
        kwargs_per_device: list[dict[str, Any]] = []
        start = 0
        for device_id, chunk in zip(active_device_ids, chunks):
            end = start + chunk.size(0)
            device = torch.device(f"cuda:{device_id}")
            item = {"pixel_values": chunk.to(device, non_blocking=True)}
            if mask_labels is not None:
                item["mask_labels"] = [mask.to(device, non_blocking=True) for mask in mask_labels[start:end]]
            if class_labels is not None:
                item["class_labels"] = [labels.to(device, non_blocking=True) for labels in class_labels[start:end]]
            item.update(kwargs)
            kwargs_per_device.append(item)
            start = end

        outputs = nn.parallel.parallel_apply(replicas, [()] * len(replicas), kwargs_per_device, active_device_ids)
        output_device = torch.device(f"cuda:{self.output_device}")
        losses = [output.loss.to(output_device) * chunk.size(0) for output, chunk in zip(outputs, chunks)]
        total = sum(chunk.size(0) for chunk in chunks)
        return SimpleNamespace(loss=sum(losses) / total)


@dataclass
class ModelAPI:
    config: ModelConfig
    module: nn.Module
    device: torch.device = torch.device("cpu")

    @classmethod
    def create(cls, config: ModelConfig) -> "ModelAPI":
        cfg = config.normalized()
        return cls(config=cfg, module=build_model(cfg))

    def prepare_for_training(self, device_arg: str | None = None) -> "ModelAPI":
        self.device = resolve_torch_device(device_arg)
        self.module = self.module.to(self.device)
        device_ids = resolve_torch_device_ids(device_arg)
        if self.device.type == "cuda" and len(device_ids) > 1:
            if self.config.architecture == "mask2former":
                self.module = Mask2FormerDataParallel(self.module, device_ids=device_ids, output_device=device_ids[0])
                print(f"Using Mask2FormerDataParallel on CUDA devices: {device_ids}")
            else:
                self.module = nn.DataParallel(self.module, device_ids=device_ids, output_device=device_ids[0])
                print(f"Using DataParallel on CUDA devices: {device_ids}")
        return self

    def save(self, path: str | Path, image_size: int, metrics: dict[str, float] | None = None) -> None:
        save_checkpoint(path, self.module, self.config, image_size, metrics)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: ModelConfig,
    image_size: int,
    metrics: dict[str, float] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, (nn.DataParallel, Mask2FormerDataParallel)) else model
    checkpoint = {
        "architecture": config.architecture,
        "model_name_or_path": config.model_name_or_path,
        "num_labels": config.num_labels,
        "image_size": image_size,
        "state_dict": model_to_save.state_dict(),
        "semantic_id2label": TRAIN_ID_TO_NAME,
        "mask2former_id2label": MASK2FORMER_ID_TO_NAME,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, checkpoint_path)


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        for candidate in ("best.pt", "last.pt", "weights/best.pt", "weights/last.pt"):
            candidate_path = checkpoint_path / candidate
            if candidate_path.is_file():
                return candidate_path
        raise FileNotFoundError(f"No best.pt or last.pt checkpoint found under: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint_path(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    config = ModelConfig(
        architecture=str(checkpoint["architecture"]),
        model_name_or_path=checkpoint.get("model_name_or_path"),
        num_labels=int(checkpoint.get("num_labels") or NUM_SEMANTIC_CLASSES),
    ).normalized()
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
