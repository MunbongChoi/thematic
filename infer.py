from __future__ import annotations

import argparse

from config import DEFAULT_IMAGE_SIZE, DEFAULT_SAMGEO_MODEL_TYPE, OUTPUT_ROOT

SUPPORTED_INFER_ARCHITECTURES = ("auto", "yolo", "yolo26", "unet", "mask2former", "sam", "geosam")


def normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "yolo26": "yolo",
        "geosam": "sam",
    }
    return aliases.get(normalized, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run satellite segmentation inference and write GeoJSON outputs.")
    parser.add_argument("--checkpoint", default=str(OUTPUT_ROOT / "unet" / "best.pt"))
    parser.add_argument("--architecture", default="auto", choices=SUPPORTED_INFER_ARCHITECTURES)
    parser.add_argument("--input", required=True, help="Input TIFF image file or directory.")
    parser.add_argument("--output-dir", default="outputs/infer")
    parser.add_argument("--output-crs", default=None, help="Required GeoJSON coordinate CRS, for example EPSG:5186.")
    parser.add_argument(
        "--reference-label-root",
        default=None,
        help="Optional GeoJSON label file/root used to recover coordinates when input TIFFs have no raster CRS.",
    )
    parser.add_argument("--gsd-m", type=float, default=None, help="Square-pixel GSD in meters. Enables area_m2.")
    parser.add_argument("--gsd-x-m", type=float, default=None, help="Horizontal GSD in meters for non-square pixels.")
    parser.add_argument("--gsd-y-m", type=float, default=None, help="Vertical GSD in meters for non-square pixels.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--min-mask-area-px", type=int, default=256, help="Drop predicted instances smaller than this pixel area.")
    parser.add_argument("--max-mask-overlap", type=float, default=0.35, help="Drop a predicted instance when this fraction is already covered by higher-confidence masks.")
    parser.add_argument("--min-segment-score", type=float, default=0.0, help="Drop Mask2Former panoptic segments below this score.")
    parser.add_argument("--tile-size", type=int, default=0, help="Enable high-resolution tiled inference with this tile size. Use 0 for whole-image inference.")
    parser.add_argument("--tile-overlap", type=int, default=128, help="Pixel overlap between inference tiles.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--samgeo-model-type", default=DEFAULT_SAMGEO_MODEL_TYPE, help="segment-geospatial SamGeo model_type.")
    parser.add_argument("--samgeo-checkpoint", default=None, help="Optional local SAM checkpoint for segment-geospatial.")
    parser.add_argument(
        "--prompt-source-mask2former-checkpoint",
        default=str(OUTPUT_ROOT / "mask2former" / "best.pt"),
        help="Mask2Former checkpoint used internally to create GeoSAM prompts and class ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.architecture = normalize_architecture(args.architecture)
    import numpy  # noqa: F401

    if args.architecture == "yolo":
        from model.YOLO26.infer import run_inference as run_yolo

        run_yolo(args)
        return

    from model.common import load_checkpoint, resolve_torch_device

    device = resolve_torch_device(args.device)
    if args.architecture == "sam":
        from model.GeoSAM.infer import run_inference as run_geosam

        run_geosam(args, device)
        return

    try:
        print(f"Loading checkpoint on CPU: {args.checkpoint}", flush=True)
        model, checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    except Exception:
        if args.architecture == "auto":
            run_yolo(args)
            return
        raise

    architecture = normalize_architecture(str(checkpoint["architecture"]))
    if args.architecture != "auto" and args.architecture != architecture:
        raise ValueError(f"Checkpoint architecture is {architecture!r}, but --architecture={args.architecture!r}.")
    if architecture == "unet":
        from model.UNet.infer import run_inference as run_unet

        print(f"Running UNet inference on {device}", flush=True)
        run_unet(args, model, checkpoint, device)
    elif architecture == "mask2former":
        from model.Mask2Former.infer import run_inference as run_mask2former

        print(f"Running Mask2Former inference on {device}", flush=True)
        run_mask2former(args, model, checkpoint, device)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")


if __name__ == "__main__":
    main()
