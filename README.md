# Satellite Road Extraction

Binary road extraction project for satellite imagery.

The training labels are GeoJSON-like files. The current dataset stores labels
under nested `TL_*_Json` / `VL_*_Json` directories and uses
`geometry.coordinates` in EPSG:5186. Training masks rasterize `ANN_CD=30`
features as road by default. Older labels with `properties.road_imcoords`
pixel polygons are still supported.

The code does not calculate distance, area, density, or other CRS-dependent
values. EPSG:5186 coordinates are only transformed into image pixel space for
mask generation. The provided TIF files do not expose a geotransform, so pixel
mapping is inferred from the GeoJSON tile bounds.

## Structure

- `model.py`: model factory and checkpoint I/O
- `train.py`: dataset parsing, mask rasterization, training loop
- `test.py`: checkpoint evaluation
- `infer.py`: mask and overlay inference output
- `dataset/train/image`: source images, including nested `TS_*` directories
- `dataset/train/label`: JSON labels, including nested `TL_*_Json` directories
- `dataset/valid`: optional validation split

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If TIF loading fails because `rasterio` is missing, install it in the same
environment:

```bash
python -m pip install rasterio
```

For conda-based Jupyter images, `conda-forge` is often more reliable because it
installs GDAL-compatible binary dependencies:

```bash
conda install -c conda-forge rasterio
```

## CUDA Check

If `nvidia-smi` shows a GPU but training says CUDA is unavailable, the active
Python environment usually has a CPU-only PyTorch build. Check PyTorch from the
same shell used to run training:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY
```

For GPU training, install a CUDA-enabled PyTorch build in that same virtual
environment, then reinstall the project requirements:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Use `--device cpu` only when intentionally training without a GPU.

## Writable Output Directory

Training writes checkpoints and logs to `--output-dir`. In Linux/Jupyter
environments, the project directory may be mounted read-only. If
`runs/road_extraction` raises `PermissionError`, use a writable absolute path:

```bash
python train.py --architecture segformer --output-dir /tmp/satellite_runs/road_extraction
```

Or set a default output directory for the shell:

```bash
export SATSEG_OUTPUT_DIR=/tmp/satellite_runs/road_extraction
python train.py --architecture segformer
```

## Train with Hugging Face SegFormer

```bash
python train.py --architecture segformer --epochs 20 --batch-size 4 --image-size 512
```

Use `--target-ann-codes` to change which `ANN_CD` values are treated as road:

```bash
python train.py --architecture segformer --target-ann-codes 30
```

The default Hugging Face model is `nvidia/segformer-b0-finetuned-ade-512-512`.
Its segmentation head is resized to two labels: `background` and `road`.

## Train with local UNet

```bash
python train.py --architecture unet --epochs 50 --batch-size 8 --image-size 512
```

## Train with YOLO segmentation

```bash
python train.py --architecture yolo --model-name-or-path yolo11n-seg.pt --epochs 50 --batch-size 8 --image-size 640
```

For YOLO, the training script converts JSON road polygons into Ultralytics
segmentation labels under `runs/road_extraction/yolo_dataset`.
The best checkpoint is copied to `runs/road_extraction/best_yolo.pt`.

## Train with Mask2Former panoptic segmentation

```bash
python train.py --architecture mask2former --output-dir runs/panoptic_segmentation --epochs 20 --batch-size 2 --image-size 512
```

The panoptic path preserves all observed `ANN_CD` classes:
`10,20,30,40,50,55,60,71,75,80,95,100`. `ANN_CD=10` is treated as
building instances, and the remaining classes are treated as stuff classes.

To export COCO-style panoptic annotations without training:

```bash
python train.py --architecture mask2former --export-panoptic-only --limit 10 --panoptic-data-dir outputs/panoptic_dataset
```

## Evaluate

```bash
python test.py --checkpoint runs/road_extraction/best_model.pt
python test.py --architecture yolo --checkpoint runs/road_extraction/best_yolo.pt --image-size 640
python test.py --architecture mask2former --checkpoint runs/panoptic_segmentation/best_mask2former.pt --image-size 512
```

`--checkpoint` normally points to a checkpoint file, not only a run directory.
For example, use `runs/road_extraction/best_model.pt` instead of
`runs/road_extraction`. If a run directory is provided, the code tries to find
one of `best_model.pt`, `best_sam.pt`, `best_mask2former.pt`,
`best_yolo.pt`, or `yolo/weights/best.pt` inside it.


## Inference

```bash
python infer.py --checkpoint runs/road_extraction/best_model.pt --input infer_data --output-dir outputs/infer
python infer.py --architecture yolo --checkpoint runs/road_extraction/best_yolo.pt --input infer_data --output-dir outputs/yolo_infer --image-size 640
python infer.py --architecture mask2former --checkpoint runs/panoptic_segmentation/best_mask2former.pt --input infer_data --output-dir outputs/panoptic_infer --image-size 512
```

Outputs are pixel-space PNG masks and visual overlays. Use a GIS/raster export
step later if georeferenced GeoTIFF output is required.
