from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np  # noqa: F401
import torch
from torch import nn

from config import DEFAULT_MASK2FORMER_MODEL, MODEL_ID_TO_NAME, MODEL_NAME_TO_ID, NUM_SEGMENT_CLASSES
from model.common import resolve_torch_device, resolve_torch_device_ids, save_checkpoint


@dataclass(frozen=True)
class Mask2FormerConfig:
    architecture: str = "mask2former"
    model_name_or_path: str | None = DEFAULT_MASK2FORMER_MODEL
    num_labels: int = NUM_SEGMENT_CLASSES


@lru_cache(maxsize=8)
def build_processor(model_name_or_path: str | None = None) -> Any:
    try:
        from transformers import Mask2FormerImageProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for Mask2Former. Install requirements.txt.") from exc
    return Mask2FormerImageProcessor.from_pretrained(model_name_or_path or DEFAULT_MASK2FORMER_MODEL)


def build_model(config: Mask2FormerConfig | None = None) -> nn.Module:
    cfg = config or Mask2FormerConfig()
    try:
        from transformers import Mask2FormerConfig as HfMask2FormerConfig
        from transformers import Mask2FormerForUniversalSegmentation
    except ImportError as exc:
        raise ImportError("transformers is required for Mask2Former. Install requirements.txt.") from exc
    model_name = str(cfg.model_name_or_path or DEFAULT_MASK2FORMER_MODEL)
    hf_config = HfMask2FormerConfig.from_pretrained(model_name)
    hf_config.num_labels = int(cfg.num_labels or NUM_SEGMENT_CLASSES)
    hf_config.id2label = dict(MODEL_ID_TO_NAME)
    hf_config.label2id = dict(MODEL_NAME_TO_ID)
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        config=hf_config,
        ignore_mismatched_sizes=True,
    )


@dataclass
class ModelAPI:
    config: Mask2FormerConfig
    module: nn.Module
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create(cls, model_name_or_path: str | None = None) -> "ModelAPI":
        config = Mask2FormerConfig(model_name_or_path=model_name_or_path or DEFAULT_MASK2FORMER_MODEL)
        return cls(config=config, module=build_model(config))

    def prepare_for_training(self, device_arg: str | None = None) -> "ModelAPI":
        self.device = resolve_torch_device(device_arg)
        device_ids = resolve_torch_device_ids(device_arg)
        if self.device.type == "cuda" and len(device_ids) > 1:
            raise ValueError(
                "Mask2Former multi-GPU training uses DDP, not DataParallel. "
                "Run: torchrun --nproc_per_node=<num_gpus> -m model.Mask2Former.train_ddp ..."
            )
        self.module = self.module.to(self.device)
        return self

    def save(self, path: str | Path, image_size: int, metrics: dict[str, float] | None = None) -> None:
        save_checkpoint(path, self.module, self.config, image_size, metrics)
