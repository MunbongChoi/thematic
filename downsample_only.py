from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.enums import Resampling


TIFF_EXTENSIONS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample TIFF files only.")
    parser.add_argument("--input", default="dataset/infer", help="Input TIFF file or directory.")
    parser.add_argument("--output", default="dataset/infer_downsampled", help="Output directory.")
    parser.add_argument("--factor", type=float, required=True, help="Downsample factor. Example: 2 means half width/height.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_tiffs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in TIFF_EXTENSIONS:
            raise ValueError(f"Expected .tif or .tiff file, got {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No TIFF files found under {path}")
    return files


def target_path(input_root: Path, source: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / source.name
    return output_root / source.relative_to(input_root)


def downsample_tiff(source: Path, target: Path, factor: float, overwrite: bool) -> None:
    if factor <= 1:
        raise ValueError("--factor must be greater than 1")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Use --overwrite.")

    with rasterio.open(source) as src:
        out_width = max(1, int(round(src.width / factor)))
        out_height = max(1, int(round(src.height / factor)))
        scale_x = src.width / out_width
        scale_y = src.height / out_height

        data = src.read(
            out_shape=(src.count, out_height, out_width),
            resampling=Resampling.bilinear,
        )

        profile = src.profile.copy()
        profile.update(
            width=out_width,
            height=out_height,
            transform=src.transform * src.transform.scale(scale_x, scale_y),
            compress=profile.get("compress") or "deflate",
            BIGTIFF="IF_SAFER",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)

    for source in iter_tiffs(input_path):
        target = target_path(input_path, source, output_root)
        downsample_tiff(source, target, args.factor, args.overwrite)
        print(f"{source} -> {target}")


if __name__ == "__main__":
    main()
