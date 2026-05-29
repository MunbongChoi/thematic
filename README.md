# Panoptic Segmentation Pipeline

This project trains and runs panoptic segmentation models on the local RGB image and GeoJSON dataset.

Supported training models:

- YOLO26 segmentation
- U-Net semantic segmentation
- Mask2Former panoptic segmentation

GeoSAM is inference-only and is run through `segment-geospatial` (`samgeo`) using Mask2Former prompts. Prompt boxes are internal and are not written to the GeoJSON contract.

The code is organized to match `AGENT.md`:

- `model/YOLO26/{model.py,train.py,infer.py}`
- `model/UNet/{model.py,train.py,infer.py}`
- `model/GeoSAM/{model.py,train.py,infer.py}`
- `model/Mask2Former/{model.py,train.py,infer.py}`
- root `train.py` and `infer.py` are CLI dispatchers.

## Dataset Contract

- Images: `dataset/{train,test}/image/**/*` with RGB TIFF imagery (`.tif`, `.tiff`)
- Labels: `dataset/{train,test}/label/**/*.json`
- Label format: GeoJSON `FeatureCollection`
- Label CRS: `EPSG:5186`
- Geometry: `Polygon` or `MultiPolygon`
- Class field: `properties.ANN_CD`

The code maps EPSG:5186 label coordinates into image pixel coordinates using each label tile's bounds. It does not calculate metric distance, area, buffer, nearest-neighbor distance, or density.

Images are assumed to already contain the correct RGB visual bands. TIFF files are read as bands 1, 2, and 3 with rasterio. Geospatial CRS validation is performed on GeoJSON labels during training and on raster CRS during coordinate-aware inference.

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

Train all supported training models:

```powershell
python train.py --architecture all --dataset-root dataset --output-dir runs/segmentation --epochs 50 --batch-size 4
```

Train one model:

```powershell
python train.py --architecture yolo --yolo-data outputs/prepared/yolo_rgb_jpg/data.yaml --output-dir runs/segmentation --epochs 50 --batch-size 64 --device 0,1,2,3 --num-workers 4
python train.py --architecture yolo26 --yolo-data outputs/prepared/yolo_rgb_jpg/data.yaml --output-dir runs/segmentation --epochs 50 --batch-size 64 --device 0,1,2,3 --num-workers 4
python train.py --architecture unet --dataset-root dataset --output-dir runs/segmentation --epochs 50
python train.py --architecture mask2former --dataset-root dataset --output-dir runs/segmentation --epochs 50
```

For RTX 4090 x4, YOLO should not be trained with `--batch-size 4`; that creates a very small per-GPU batch and usually leaves the GPUs underfed. Start with `--batch-size 64`, then reduce to `32` or `16` only if CUDA memory is exhausted. YOLO cache is forced to `False` in code to avoid RAM pressure. Reuse an existing prepared `data.yaml` with `--yolo-data`; otherwise the script reuses the newest `data.yaml` under `--prepared-dir` when one exists. `--num-workers` is per YOLO GPU process, so `--num-workers 4` creates up to 16 loader workers in 4-GPU DDP.

Each model writes:

- `runs/segmentation/<model>/best.pt`
- `runs/segmentation/<model>/last.pt`
- `runs/segmentation/<model>/history.json` for PyTorch models

## Inference

```powershell
python infer.py --architecture unet --checkpoint runs/segmentation/unet/best.pt --input dataset/test/image --output-dir outputs/infer/unet --output-crs EPSG:5186
python infer.py --architecture yolo --checkpoint runs/segmentation/yolo/best.pt --input dataset/test/image --output-dir outputs/infer/yolo --output-crs EPSG:5186 --gsd-m 0.5
python infer.py --architecture mask2former --checkpoint runs/segmentation/mask2former/best.pt --input dataset/test/image --output-dir outputs/infer/mask2former --output-crs EPSG:5186 --gsd-x-m 0.5 --gsd-y-m 0.5
python infer.py --architecture sam --prompt-source-mask2former-checkpoint runs/segmentation/mask2former/best.pt --input dataset/test/image --output-dir outputs/infer/sam --output-crs EPSG:5186 --samgeo-model-type vit_h
```

GeoSAM inference uses `segment-geospatial` with Mask2Former segments as internal prompts and class sources. GeoSAM is not trained by `train.py`; `python train.py --architecture sam` fails with an explicit inference-only error.

Inference writes:

- `*_semantic_mask.png`
- `*_panoptic.png`
- `*_color.png`
- `*_overlay.png`
- `*_segments.geojson`
- `results.json`

`--output-crs` is required before GeoJSON can be written. The input TIFF must have a raster CRS matching `--output-crs`; the code does not silently reproject or invent a coordinate system.

Each GeoJSON feature includes:

- `properties.id`
- `properties.class`
- `properties.class_id`
- `properties.area_px`
- `properties.coordinate_crs`
- `properties.coordinates`
- `properties.area_m2`, `properties.gsd_x_m`, and `properties.gsd_y_m` only when GSD is provided

Metric area is calculated only from explicit GSD:

- `--gsd-m 0.5` for square pixels
- `--gsd-x-m 0.5 --gsd-y-m 0.75` for non-square pixels
- `area_m2 = area_px * gsd_x_m * gsd_y_m`
