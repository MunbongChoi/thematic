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

```powershell
pip install -r requirements.txt
```

## Docker GPU Run

Build the image from the project root:

```bash
docker build -f dockerfile -t satellite-seg:cuda124 .
```

On Windows PowerShell, use backticks instead of Bash backslashes:

```powershell
docker build -f dockerfile -t satellite-seg:cuda124 .
```

Run the container with NVIDIA Container Toolkit. Mount the dataset, run outputs,
and model weights instead of baking them into the image:

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/dataset:/app/dataset \
  -v /path/to/runs:/app/runs \
  -v /path/to/outputs:/app/outputs \
  -v /path/to/weights:/app/weights \
  satellite-seg:cuda124 python train.py --architecture yolo \
    --model-name-or-path /app/weights/yolo26n-seg.pt \
    --output-dir runs/yolo26_road \
    --epochs 100 --batch-size 16 --image-size 1024 \
    --num-workers 8 --target-ann-codes 30 --device 0,1,2,3
```

PowerShell equivalent:

```powershell
docker run --rm --gpus all --ipc=host `
  -v "$((Get-Location).Path)\dataset:/app/dataset" `
  -v "$((Get-Location).Path)\runs:/app/runs" `
  -v "$((Get-Location).Path)\outputs:/app/outputs" `
  -v "$((Get-Location).Path)\weights:/app/weights" `
  satellite-seg:cuda124 python train.py --architecture yolo `
    --model-name-or-path /app/weights/yolo26n-seg.pt `
    --output-dir runs/yolo26_road `
    --epochs 100 --batch-size 16 --image-size 1024 `
    --num-workers 8 --target-ann-codes 30 --device 0,1,2,3
```

The same command as a single line, which avoids line-continuation parsing
issues:

```powershell
docker run --rm --gpus all --ipc=host -v "$((Get-Location).Path)\dataset:/app/dataset" -v "$((Get-Location).Path)\runs:/app/runs" -v "$((Get-Location).Path)\outputs:/app/outputs" -v "$((Get-Location).Path)\weights:/app/weights" satellite-seg:cuda124 python train.py --architecture yolo --model-name-or-path /app/weights/yolo26n-seg.pt --output-dir runs/yolo26_road --epochs 100 --batch-size 16 --image-size 1024 --num-workers 8 --target-ann-codes 30 --device 0,1,2,3
```

If you are using Windows Command Prompt (`cmd.exe`), use `set` and
`--mount`. This avoids the Windows drive-colon parsing issues that can cause
`docker: invalid reference format`.

```cmd
set "PROJECT_DIR=%cd%"
docker run --rm --gpus all --ipc=host --mount type=bind,source="%PROJECT_DIR%\dataset",target=/app/dataset --mount type=bind,source="%PROJECT_DIR%\runs",target=/app/runs --mount type=bind,source="%PROJECT_DIR%\outputs",target=/app/outputs --mount type=bind,source="%PROJECT_DIR%\weights",target=/app/weights satellite-seg:cuda124 python train.py --architecture yolo --model-name-or-path /app/weights/yolo26n-seg.pt --output-dir runs/yolo26_road --epochs 100 --batch-size 16 --image-size 1024 --num-workers 8 --target-ann-codes 30 --device 0,1,2,3
```

For CMD multi-line commands, use `^` as the line-continuation character:

```cmd
set "PROJECT_DIR=%cd%"
docker run --rm --gpus all --ipc=host ^
  --mount type=bind,source="%PROJECT_DIR%\dataset",target=/app/dataset ^
  --mount type=bind,source="%PROJECT_DIR%\runs",target=/app/runs ^
  --mount type=bind,source="%PROJECT_DIR%\outputs",target=/app/outputs ^
  --mount type=bind,source="%PROJECT_DIR%\weights",target=/app/weights ^
  satellite-seg:cuda124 python train.py --architecture yolo ^
    --model-name-or-path /app/weights/yolo26n-seg.pt ^
    --output-dir runs/yolo26_road ^
    --epochs 100 --batch-size 16 --image-size 1024 ^
    --num-workers 8 --target-ann-codes 30 --device 0,1,2,3
```

For torch-based models such as Mask2Former, pass one GPU explicitly:

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/dataset:/app/dataset \
  -v /path/to/runs:/app/runs \
  -v /path/to/outputs:/app/outputs \
  satellite-seg:cuda124 python train.py --architecture mask2former \
    --output-dir runs/mask2former_panoptic \
    --epochs 50 --batch-size 2 --image-size 1024 \
    --num-workers 8 --device cuda:0
```

## Train with Hugging Face SegFormer

```powershell
python train.py --architecture segformer --epochs 20 --batch-size 4 --image-size 512
```

Use `--target-ann-codes` to change which `ANN_CD` values are treated as road:

```powershell
python train.py --architecture segformer --target-ann-codes 30
```

The default Hugging Face model is `nvidia/segformer-b0-finetuned-ade-512-512`.
Its segmentation head is resized to two labels: `background` and `road`.

## Train with local UNet

```powershell
python train.py --architecture unet --epochs 50 --batch-size 8 --image-size 512
```

## Train with YOLO segmentation

```powershell
python train.py --architecture yolo --model-name-or-path yolo11n-seg.pt --epochs 50 --batch-size 8 --image-size 640
```

For YOLO, the training script converts JSON road polygons into Ultralytics
segmentation labels under `runs/road_extraction/yolo_dataset`.
The best checkpoint is copied to `runs/road_extraction/best_yolo.pt`.

## Train with Mask2Former panoptic segmentation

```powershell
python train.py --architecture mask2former --output-dir runs/panoptic_segmentation --epochs 20 --batch-size 2 --image-size 512
```

The panoptic path preserves all observed `ANN_CD` classes:
`10,20,30,40,50,55,60,71,75,80,95,100`. `ANN_CD=10` is treated as
building instances, and the remaining classes are treated as stuff classes.

To export COCO-style panoptic annotations without training:

```powershell
python train.py --architecture mask2former --export-panoptic-only --limit 10 --panoptic-data-dir outputs/panoptic_dataset
```

## Evaluate

```powershell
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

```powershell
python infer.py --checkpoint runs/road_extraction/best_model.pt --input infer_data --output-dir outputs/infer
python infer.py --architecture yolo --checkpoint runs/road_extraction/best_yolo.pt --input infer_data --output-dir outputs/yolo_infer --image-size 640
python infer.py --architecture mask2former --checkpoint runs/panoptic_segmentation/best_mask2former.pt --input infer_data --output-dir outputs/panoptic_infer --image-size 512
```

Outputs are pixel-space PNG masks and visual overlays. Use a GIS/raster export
step later if georeferenced GeoTIFF output is required.
