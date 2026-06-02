from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn

from config import DEFAULT_UNET_MODEL, NUM_SEMANTIC_CLASSES
from model.common import resolve_torch_device, resolve_torch_device_ids, save_checkpoint


@dataclass(frozen=True)
class UNetConfig:
    architecture: str = "unet"
    model_name_or_path: str | None = DEFAULT_UNET_MODEL
    num_labels: int = NUM_SEMANTIC_CLASSES
    base_channels: int = 48
    dropout: float = 0.1
    variant: str = DEFAULT_UNET_MODEL


def _normalize_variant(value: str | None) -> str:
    normalized = (value or DEFAULT_UNET_MODEL).strip().lower().replace("_", "-")
    aliases = {
        "unet": "unet-basic",
        "basic": "unet-basic",
        "basic-unet": "unet-basic",
        "resunet": "resattn-unet",
        "res-attn-unet": "resattn-unet",
        "attention-unet": "resattn-unet",
        "advanced": "resattn-unet",
    }
    return aliases.get(normalized, normalized)


def _group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (16, 8, 4, 2):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


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


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False), _group_norm(inter_channels))
        self.gate_proj = nn.Sequential(nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False), _group_norm(inter_channels))
        self.psi = nn.Sequential(nn.SiLU(inplace=True), nn.Conv2d(inter_channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * alpha


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(skip_channels, out_channels, out_channels)
        self.block = ResidualConvBlock(out_channels + skip_channels, out_channels, dropout)
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(skip, x)
        return self.channel_attention(self.block(torch.cat([x, skip], dim=1)))


class ResidualAttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_labels: int = NUM_SEMANTIC_CLASSES, base_channels: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = ResidualConvBlock(in_channels, channels[0], dropout=0.0)
        self.enc2 = ResidualConvBlock(channels[0], channels[1], dropout=dropout)
        self.enc3 = ResidualConvBlock(channels[1], channels[2], dropout=dropout)
        self.enc4 = ResidualConvBlock(channels[2], channels[3], dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(channels[3], channels[3] * 2, dropout=dropout),
            ChannelAttention(channels[3] * 2),
        )
        self.dec4 = DecoderBlock(channels[3] * 2, channels[3], channels[3], dropout)
        self.dec3 = DecoderBlock(channels[3], channels[2], channels[2], dropout)
        self.dec2 = DecoderBlock(channels[2], channels[1], channels[1], dropout)
        self.dec1 = DecoderBlock(channels[1], channels[0], channels[0], dropout=0.0)
        self.head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            _group_norm(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], num_labels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        x = self.bottleneck(self.pool(enc4))
        x = self.dec4(x, enc4)
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        return self.head(x)


def build_model(config: UNetConfig | None = None) -> nn.Module:
    cfg = config or UNetConfig()
    variant = _normalize_variant(cfg.variant or cfg.model_name_or_path)
    if variant == "unet-basic":
        return UNet(num_labels=int(cfg.num_labels), base_channels=int(cfg.base_channels or 32))
    if variant == "resattn-unet":
        return ResidualAttentionUNet(
            num_labels=int(cfg.num_labels),
            base_channels=int(cfg.base_channels or 48),
            dropout=float(cfg.dropout or 0.0),
        )
    raise ValueError(f"Unsupported UNet variant {cfg.variant!r}. Use 'resattn-unet' or 'unet-basic'.")


def config_from_model_name(model_name_or_path: str | None = None) -> UNetConfig:
    model_name = model_name_or_path or DEFAULT_UNET_MODEL
    variant = _normalize_variant(model_name)
    if variant == "unet-basic":
        return UNetConfig(model_name_or_path=model_name, variant=variant, base_channels=32, dropout=0.0)
    return UNetConfig(model_name_or_path=model_name, variant=variant)


@dataclass
class ModelAPI:
    config: UNetConfig
    module: nn.Module
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create(cls, model_name_or_path: str | None = None) -> "ModelAPI":
        config = config_from_model_name(model_name_or_path)
        return cls(config=config, module=build_model(config))

    def prepare_for_training(self, device_arg: str | None = None) -> "ModelAPI":
        self.device = resolve_torch_device(device_arg)
        device_ids = resolve_torch_device_ids(device_arg)
        if self.device.type == "cuda" and len(device_ids) > 1:
            raise ValueError(
                "UNet multi-GPU training uses DDP, not DataParallel. "
                "Run: torchrun --nproc_per_node=<num_gpus> -m model.UNet.train_ddp ..."
            )
        self.module = self.module.to(self.device)
        return self

    def save(self, path: str | Path, image_size: int, metrics: dict[str, float] | None = None) -> None:
        save_checkpoint(path, self.module, self.config, image_size, metrics)
