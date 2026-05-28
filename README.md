# Segmentation Training Pipeline

This project trains multiple segmentation models on the GeoJSON/TIF dataset.

Supported classes are building, parking lot, road, street tree, paddy field,
greenhouse, field, broadleaf forest, coniferous forest, bare ground, water, and
non-cultivated land.

## Dataset Assumptions

- Input images: `dataset/{train,test}/image/**/*.tif`
- Input labels: `dataset/{train,test}/label/**/*.json`
- Label format: GeoJSON `FeatureCollection`
- Label CRS: `EPSG:5186`
- Geometry: `Polygon` or `MultiPolygon`
- Spatial operation: pixel-space mask generation only
- Output masks: pixel-space class rasters, not GIS measurement layers

The code maps EPSG:5186 label coordinates into image pixel coordinates from
each label tile's bounds. It does not compute distance, area, buffer, density,
or nearest-neighbor values.

## Install

```powershell
pip install -r requirements.txt
```

## Prepare Datasets Only

```powershell
python train.py --prepare-only --dataset-root dataset --prepared-dir outputs/prepared
```

## Train

Train all supported models:

```powershell
python train.py --architecture all --dataset-root dataset --output-dir runs/segmentation --epochs 50 --batch-size 4
```

Train one model:

```powershell
python train.py --architecture unet --dataset-root dataset --output-dir runs/segmentation --epochs 50
python train.py --architecture yolo --dataset-root dataset --output-dir runs/segmentation --epochs 50
python train.py --architecture mask2former --dataset-root dataset --output-dir runs/segmentation --epochs 50
```

Each trained model writes:

- `runs/segmentation/<architecture>/best.pt`
- `runs/segmentation/<architecture>/last.pt`
- `runs/segmentation/<architecture>/history.json` for PyTorch models

## Inference

```powershell
python infer.py --architecture auto --checkpoint runs/segmentation/unet/best.pt --input dataset/test/image --output-dir outputs/infer/unet
python infer.py --architecture yolo --checkpoint runs/segmentation/yolo/best.pt --input dataset/test/image --output-dir outputs/infer/yolo
```

Inference writes class-id masks, color masks, overlays, and `results.json`.

