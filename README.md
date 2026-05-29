# Panoptic Segmentation Pipeline

This project trains and runs panoptic segmentation models on the local RGB image and GeoJSON dataset.

Supported models:

- YOLO segmentation
- U-Net semantic segmentation
- Mask2Former panoptic segmentation
- SAM/SAM2 mask refinement using Mask2Former bbox prompts

## Dataset Contract

- Images: `dataset/{train,test}/image/**/*` with RGB imagery (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`)
- Labels: `dataset/{train,test}/label/**/*.json`
- Label format: GeoJSON `FeatureCollection`
- Label CRS: `EPSG:5186`
- Geometry: `Polygon` or `MultiPolygon`
- Class field: `properties.ANN_CD`

The code maps EPSG:5186 label coordinates into image pixel coordinates using each label tile's bounds. It does not calculate metric distance, area, buffer, nearest-neighbor distance, or density.

Images are assumed to already contain the correct RGB visual bands. TIFF files are read as bands 1, 2, and 3 with rasterio; PNG/JPG files are read with PIL and converted to RGB. Geospatial CRS validation is performed on the GeoJSON labels, not the image pixels.

## Classes

| ANN_CD | Class |
|---:|---|
| 10 | building |
| 20 | parking_lot |
| 30 | road |
| 40 | street_tree |
| 50 | paddy_field |
| 55 | greenhouse |
| 60 | field |
| 71 | broadleaf_forest |
| 75 | coniferous_forest |
| 80 | bare_ground |
| 95 | water |
| 100 | non_cultivated |

## Install

```powershell
pip install -r requirements.txt
```

For CUDA training, install a CUDA-enabled PyTorch build first:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu-cuda122.txt
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Prepare Datasets

```powershell
python train.py --prepare-only --dataset-root dataset --prepared-dir outputs/prepared
```

Smoke test:

```powershell
python train.py --prepare-only --dataset-root dataset --prepared-dir outputs/prepared_smoke --limit 2
```

## Train

Train all supported models:

```powershell
python train.py --architecture all --dataset-root dataset --output-dir runs/segmentation --epochs 50 --batch-size 4
```

Train one model:

```powershell
python train.py --architecture yolo --dataset-root dataset --output-dir runs/segmentation --epochs 50
python train.py --architecture unet --dataset-root dataset --output-dir runs/segmentation --epochs 50
python train.py --architecture mask2former --dataset-root dataset --output-dir runs/segmentation --epochs 50
python train.py --architecture sam --dataset-root dataset --output-dir runs/segmentation --epochs 50
```

Each model writes:

- `runs/segmentation/<model>/best.pt`
- `runs/segmentation/<model>/last.pt`
- `runs/segmentation/<model>/history.json` for PyTorch models

## Inference

```powershell
python infer.py --architecture unet --checkpoint runs/segmentation/unet/best.pt --input dataset/test/image --output-dir outputs/infer/unet
python infer.py --architecture yolo --checkpoint runs/segmentation/yolo/best.pt --input dataset/test/image --output-dir outputs/infer/yolo
python infer.py --architecture mask2former --checkpoint runs/segmentation/mask2former/best.pt --input dataset/test/image --output-dir outputs/infer/mask2former
python infer.py --architecture sam --checkpoint runs/segmentation/sam/best.pt --prompt-source-mask2former-checkpoint runs/segmentation/mask2former/best.pt --input dataset/test/image --output-dir outputs/infer/sam
```

SAM/SAM2 inference uses Mask2Former segments as bbox prompts and class sources.

Inference writes:

- `*_semantic_mask.png`
- `*_panoptic.png`
- `*_color.png`
- `*_overlay.png`
- `*_segments.geojson`
- `results.json`

GeoJSON output is pixel-space by default. Do not treat its coordinates as authoritative metric GIS coordinates unless a georeferenced raster transform is explicitly added and verified.
