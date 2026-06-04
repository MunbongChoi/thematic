from __future__ import annotations

import argparse
from pathlib import Path

from config import IMAGE_EXTENSIONS


RESAMPLING_CHOICES = {
    "nearest",
    "bilinear",
    "cubic",
    "average",
    "lanczos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample TIFF files under dataset/infer while preserving raster metadata.")
    parser.add_argument("--input-dir", default="dataset/infer", help="Input TIFF file or directory. Defaults to dataset/infer.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <input>_downsampled_x<factor>.")
    parser.add_argument("--factor", type=float, required=True, help="Downsample factor. 2 means width/height become half.")
    parser.add_argument("--resampling", default="bilinear", choices=sorted(RESAMPLING_CHOICES))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional file limit for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files.")
    return parser.parse_args()


def iter_tiff_files(input_path: Path, limit: int | None = None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image: {input_path}. Expected .tif or .tiff.")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        files = files[: max(0, int(limit))]
    if not files:
        raise FileNotFoundError(f"No TIFF files found under: {input_path}")
    return files


def default_output_dir(input_path: Path, factor: float) -> Path:
    factor_text = f"{factor:g}".replace(".", "p")
    if input_path.is_file():
        return input_path.parent / f"{input_path.stem}_downsampled_x{factor_text}"
    return input_path.parent / f"{input_path.name}_downsampled_x{factor_text}"


def output_path_for(input_path: Path, source_path: Path, output_dir: Path) -> Path:
    if input_path.is_file():
        return output_dir / source_path.name
    return output_dir / source_path.relative_to(input_path)


def resampling_method(name: str):
    try:
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise ImportError("rasterio is required for GeoTIFF downsampling. Install requirements.txt.") from exc
    return getattr(Resampling, name)


def downsample_tiff(source_path: Path, target_path: Path, factor: float, resampling_name: str, overwrite: bool) -> None:
    if factor <= 1.0:
        raise ValueError("--factor must be greater than 1.0 for downsampling.")
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target_path}. Pass --overwrite to replace it.")

    import rasterio

    resampling = resampling_method(resampling_name)
    with rasterio.open(source_path) as src:
        output_width = max(1, int(round(src.width / factor)))
        output_height = max(1, int(round(src.height / factor)))
        actual_x_scale = src.width / output_width
        actual_y_scale = src.height / output_height
        data = src.read(
            out_shape=(src.count, output_height, output_width),
            resampling=resampling,
        )
        transform = src.transform * src.transform.scale(actual_x_scale, actual_y_scale)
        profile = src.profile.copy()
        profile.update(
            width=output_width,
            height=output_height,
            transform=transform,
            compress=profile.get("compress") or "deflate",
            BIGTIFF="IF_SAFER",
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target_path, "w", **profile) as dst:
        dst.write(data)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_path, args.factor)
    files = iter_tiff_files(input_path, args.limit)
    print(f"Downsample factor: {args.factor:g}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(files)}")
    for source_path in files:
        target_path = output_path_for(input_path, source_path, output_dir)
        print(f"{source_path} -> {target_path}")
        if not args.dry_run:
            downsample_tiff(source_path, target_path, args.factor, args.resampling, args.overwrite)


if __name__ == "__main__":
    main()
