from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import CLASSES, DEFAULT_IMAGE_SIZE, DEFAULT_SEED, OUTPUT_ROOT, PREPARED_ROOT
from data import export_semantic_masks, seed_everything, write_prepared_metadata
from model.GeoSAM.train import train as train_geosam
from model.Mask2Former.train import train as train_mask2former
from model.UNet.train import train as train_unet
from model.YOLO26.train import prepare_yolo_dataset, train as train_yolo

SUPPORTED_TRAIN_ARCHITECTURES = ("all", "yolo", "yolo26", "unet", "mask2former", "sam", "geosam")
TRAINABLE_ARCHITECTURES = ("yolo", "unet", "mask2former")


def normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "yolo26": "yolo",
        "geosam": "sam",
    }
    return aliases.get(normalized, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets and train satellite segmentation models.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--prepared-dir", default=str(PREPARED_ROOT))
    parser.add_argument("--architecture", default="all", choices=SUPPORTED_TRAIN_ARCHITECTURES)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers. For YOLO DDP this is per GPU process.")
    parser.add_argument("--yolo-cache", default="false", help=argparse.SUPPRESS)
    parser.add_argument("--yolo-data", default=None, help="Existing YOLO data.yaml. If set, YOLO preparation is skipped.")
    parser.add_argument("--yolo-image-format", default="jpg", choices=["jpg", "png"], help="Prepared YOLO image format.")
    parser.add_argument("--yolo-jpeg-quality", type=int, default=95, help="JPEG quality when --yolo-image-format jpg.")
    parser.add_argument("--yolo-amp", action=argparse.BooleanOptionalAction, default=True, help="Enable Ultralytics AMP for YOLO.")
    parser.add_argument("--device", default=None, help="Use cpu, 0, or cuda:0. Use torchrun DDP entrypoints for multi-GPU PyTorch models.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def prepare_all_datasets(args: argparse.Namespace) -> None:
    export_semantic_masks(args)
    prepare_yolo_dataset(args)
    write_prepared_metadata(args.prepared_dir)
    classes_path = Path(args.prepared_dir) / "classes.json"
    classes_path.write_text(json.dumps([item.__dict__ for item in CLASSES], indent=2), encoding="utf-8")


TRAIN_DISPATCH = {
    "yolo": train_yolo,
    "unet": train_unet,
    "mask2former": train_mask2former,
    "sam": train_geosam,
}


def main() -> None:
    args = parse_args()
    args.architecture = normalize_architecture(args.architecture)
    seed_everything(args.seed)
    if args.prepare_only:
        prepare_all_datasets(args)
        print(f"Prepared datasets written to {Path(args.prepared_dir).resolve()}")
        return
    if args.architecture == "all" and args.model_name_or_path:
        raise ValueError("--model-name-or-path can target only one architecture. Run each architecture separately when overriding it.")
    architectures = TRAINABLE_ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    for architecture in architectures:
        print(f"Training {architecture} -> {Path(args.output_dir) / architecture}")
        TRAIN_DISPATCH[architecture](args)


if __name__ == "__main__":
    main()
