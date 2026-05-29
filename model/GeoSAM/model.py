from __future__ import annotations

import torch

from config import DEFAULT_SAMGEO_MODEL_TYPE


def build_samgeo(args, device: torch.device) -> object:
    try:
        from samgeo import SamGeo
    except ImportError as exc:
        raise ImportError("SAM inference requires segment-geospatial. Install it with: pip install segment-geospatial") from exc
    kwargs = {"model_type": getattr(args, "samgeo_model_type", DEFAULT_SAMGEO_MODEL_TYPE), "automatic": False}
    if getattr(args, "samgeo_checkpoint", None):
        kwargs["checkpoint"] = args.samgeo_checkpoint
    if device.type == "cuda":
        kwargs["device"] = str(device)
    try:
        return SamGeo(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return SamGeo(**kwargs)
