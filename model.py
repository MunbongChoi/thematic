from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from config import DEFAULT_MODEL_NAME, ID2LABEL as PANOPTIC_LABELS, LABEL2ID as PANOPTIC_LABEL2ID, NUM_CLASSES


ROAD_LABELS = {0: "background", 1: "road"}
DEFAULT_YOLO_SEG_MODEL = "yolo11n-seg.pt"


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "segformer"
    model_name_or_path: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_labels: int = 2


def resolve_torch_device(device: str | None = None) -> torch.device:
    if device is None or device == "":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if "," in normalized:
        first_device = normalized.split(",", maxsplit=1)[0].strip()
        normalized = first_device
    if normalized.isdigit():
        return torch.device(f"cuda:{normalized}")
    if normalized == "cuda":
        return torch.device("cuda:0")
    return torch.device(normalized)


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
    return YOLO(model_name_or_path)


def save_checkpoint(
    path: str,
    model: nn.Module,
    config: ModelConfig,
    image_size: int,
    metrics: dict[str, float] | None = None,
) -> None:
    torch.save(
        {
            "architecture": config.architecture,
            "model_name_or_path": config.model_name_or_path,
            "num_labels": config.num_labels,
            "image_size": image_size,
            "state_dict": model.state_dict(),
            "id2label": PANOPTIC_LABELS if config.architecture in {"mask2former", "panoptic"} else ROAD_LABELS,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    config = ModelConfig(
        architecture=checkpoint["architecture"],
        model_name_or_path=checkpoint["model_name_or_path"],
        num_labels=int(checkpoint.get("num_labels", 2)),
    )
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
