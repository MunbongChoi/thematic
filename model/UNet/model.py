from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from config import NUM_SEMANTIC_CLASSES
from model.common import resolve_torch_device, resolve_torch_device_ids, save_checkpoint


@dataclass(frozen=True)
class UNetConfig:
    architecture: str = "unet"
    model_name_or_path: str | None = "unet"
    num_labels: int = NUM_SEMANTIC_CLASSES


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
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
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


def build_model(config: UNetConfig | None = None) -> nn.Module:
    cfg = config or UNetConfig()
    return UNet(num_labels=int(cfg.num_labels))


@dataclass
class ModelAPI:
    config: UNetConfig
    module: nn.Module
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create(cls, model_name_or_path: str | None = None) -> "ModelAPI":
        config = UNetConfig(model_name_or_path=model_name_or_path or "unet")
        return cls(config=config, module=build_model(config))

    def prepare_for_training(self, device_arg: str | None = None) -> "ModelAPI":
        self.device = resolve_torch_device(device_arg)
        self.module = self.module.to(self.device)
        device_ids = resolve_torch_device_ids(device_arg)
        if self.device.type == "cuda" and len(device_ids) > 1:
            self.module = nn.DataParallel(self.module, device_ids=device_ids, output_device=device_ids[0])
        return self

    def save(self, path: str | Path, image_size: int, metrics: dict[str, float] | None = None) -> None:
        save_checkpoint(path, self.module, self.config, image_size, metrics)
