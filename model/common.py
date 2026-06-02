from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np  # noqa: F401
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from config import MODEL_ID_TO_NAME, NUM_SEGMENT_CLASSES, NUM_SEMANTIC_CLASSES, TRAIN_ID_TO_NAME


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
        if raw_id:
            device_ids.append(int(raw_id))
    if device_ids and not torch.cuda.is_available():
        raise RuntimeError(cuda_diagnostic_message(str(device)))
    invalid = [idx for idx in device_ids if idx < 0 or idx >= torch.cuda.device_count()]
    if invalid:
        raise RuntimeError(f"CUDA device index(es) {invalid} unavailable. Found {torch.cuda.device_count()} CUDA device(s).")
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


def config_to_checkpoint_dict(config: object) -> dict[str, object]:
    if is_dataclass(config):
        return asdict(config)
    return dict(getattr(config, "__dict__", {}))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: object,
    image_size: int,
    metrics: dict[str, float] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    payload = config_to_checkpoint_dict(config)
    torch.save(
        {
            "architecture": str(payload["architecture"]),
            "model_name_or_path": payload.get("model_name_or_path"),
            "num_labels": payload.get("num_labels"),
            "model_config": payload,
            "image_size": image_size,
            "state_dict": model_to_save.state_dict(),
            "semantic_id2label": TRAIN_ID_TO_NAME,
            "segment_id2label": MODEL_ID_TO_NAME,
            "metrics": metrics or {},
        },
        checkpoint_path,
    )


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        for candidate in (
            "best.pt",
            "last.pt",
            "best_model.pt",
            "last_model.pt",
            "best_mask2former.pt",
            "last_mask2former.pt",
            "best_yolo.pt",
            "last_yolo.pt",
            "weights/best.pt",
            "weights/last.pt",
        ):
            candidate_path = checkpoint_path / candidate
            if candidate_path.is_file():
                return candidate_path
        raise FileNotFoundError(f"No supported checkpoint found under: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def build_model_for_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    architecture = str(checkpoint["architecture"]).strip().lower()
    if architecture == "unet":
        from model.UNet.model import DEFAULT_UNET_MODEL, UNetConfig, build_model, _normalize_variant

        payload = checkpoint.get("model_config") or {}
        model_name = checkpoint.get("model_name_or_path")
        variant = payload.get("variant") or _normalize_variant(str(model_name or DEFAULT_UNET_MODEL))
        config = UNetConfig(
            model_name_or_path=model_name,
            num_labels=int(checkpoint.get("num_labels") or NUM_SEMANTIC_CLASSES),
            base_channels=int(payload.get("base_channels") or (32 if variant == "unet-basic" else 48)),
            dropout=float(payload.get("dropout") or 0.0 if variant == "unet-basic" else payload.get("dropout", 0.1)),
            variant=str(variant),
        )
        return build_model(config)
    if architecture == "mask2former":
        from model.Mask2Former.model import Mask2FormerConfig, build_model

        config = Mask2FormerConfig(
            model_name_or_path=checkpoint.get("model_name_or_path"),
            num_labels=int(checkpoint.get("num_labels") or NUM_SEGMENT_CLASSES),
        )
        return build_model(config)
    raise ValueError(f"Unsupported checkpoint architecture: {architecture}")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint_path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = build_model_for_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint
