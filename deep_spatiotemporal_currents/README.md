# Deep Spatio-Temporal Currents Forecast (ConvLSTM–U-Net)

This repo turns your `currents.ipynb` CMEMS NetCDF reading/plotting into a full training + inference pipeline
for a ConvLSTM-in-the-bottleneck U-Net that predicts the next surface current field (u,v) from the last L frames.

## What it does
- Loads CMEMS-style NetCDF with `utotal` / `vtotal` (or other vars you specify).
- Builds sliding windows: (t-L ... t-1) -> t.
- Train/val split is **time-based** (no leakage).
- Normalizes using **train-only** statistics (min-max by default).
- Trains a **U-Net encoder/decoder** with a **ConvLSTM** bottleneck.
- Saves checkpoints, metrics, and a config snapshot.

## Quick start

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Put your NetCDF file in the project folder (or pass an absolute path)
Example expected vars:
- `utotal` and `vtotal`
- dims: `time`, `depth`, `latitude`, `longitude` (or `lat`, `lon`)

### 3) Train
**Native grid (fastest setup):**
```bash
python scripts/train.py --data /path/to/your_file.nc --out runs/run1
```

**Crop to a bbox:**
```bash
python scripts/train.py --data /path/to/your_file.nc --out runs/run_bbox \
  --lat-min 42.0 --lat-max 44.5 --lon-min -7.0 --lon-max -3.0
```

**Regrid to fixed HxW (recommended for DL):**
```bash
python scripts/train.py --data /path/to/your_file.nc --out runs/run_64 \
  --regrid-h 64 --regrid-w 64
```

### 4) Inference (predict one step)
```bash
python scripts/infer.py --ckpt runs/run1/checkpoints/best.pt --data /path/to/your_file.nc \
  --t-index 100 --out runs/run1/pred_t100.npz
```

The output `.npz` includes:
- `pred_uv` (2,H,W), `gt_uv` (2,H,W)
- `lat`, `lon` grids and `time` stamp

## Notes
- Default is `utotal/vtotal` at `depth=0`. Change with CLI flags if needed.
- If your file uses `lat/lon` instead of `latitude/longitude`, the loader auto-detects.
- NaNs are filled with 0.0 by default; you can also export a mask channel later if you want.
