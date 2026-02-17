from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

def _find_dim(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    return None

def _find_var(ds: xr.Dataset, name: str) -> str:
    if name in ds.data_vars:
        return name
    raise KeyError(f"Variable '{name}' not found in dataset vars: {list(ds.data_vars)}")

@dataclass
class NormStats:
    method: str
    u_min: float
    u_max: float
    v_min: float
    v_max: float

def compute_minmax(train_uv: np.ndarray) -> NormStats:
    # train_uv: (T,2,H,W)
    u = train_uv[:,0]
    v = train_uv[:,1]
    u_min, u_max = float(np.nanmin(u)), float(np.nanmax(u))
    v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))
    # avoid zero range
    if abs(u_max - u_min) < 1e-12:
        u_max = u_min + 1.0
    if abs(v_max - v_min) < 1e-12:
        v_max = v_min + 1.0
    return NormStats(method="minmax", u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

def apply_minmax(uv: np.ndarray, stats: NormStats) -> np.ndarray:
    out = uv.copy()
    out[:,0] = (out[:,0] - stats.u_min) / (stats.u_max - stats.u_min)
    out[:,1] = (out[:,1] - stats.v_min) / (stats.v_max - stats.v_min)
    return out

def invert_minmax(uv01: np.ndarray, stats: NormStats) -> np.ndarray:
    out = uv01.copy()
    out[0] = out[0] * (stats.u_max - stats.u_min) + stats.u_min
    out[1] = out[1] * (stats.v_max - stats.v_min) + stats.v_min
    return out

def load_cmems_uv(
    path: str,
    u_var: str = "utotal",
    v_var: str = "vtotal",
    depth_index: int = 0,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    regrid_hw: Optional[Tuple[int,int]] = None,
    fillna: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      uv: (T,2,H,W) float32
      time: (T,) numpy datetime64
      lat: (H,) or (H,W) depending on xarray interp -> we return 1D lat coords
      lon: (W,) 1D lon coords
    """
    ds = xr.open_dataset(path, decode_times=True)

    time_dim = _find_dim(ds, ["time", "Times"])
    lat_dim  = _find_dim(ds, ["latitude", "lat", "y"])
    lon_dim  = _find_dim(ds, ["longitude", "lon", "x"])
    depth_dim = _find_dim(ds, ["depth", "depthu", "depthv", "z"])

    if time_dim is None or lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not detect dims. Found dims={list(ds.dims)} coords={list(ds.coords)}")

    u_name = _find_var(ds, u_var)
    v_name = _find_var(ds, v_var)

    u = ds[u_name]
    v = ds[v_name]

    # Select depth if present
    if depth_dim is not None and depth_dim in u.dims:
        u = u.isel({depth_dim: depth_index})
    if depth_dim is not None and depth_dim in v.dims:
        v = v.isel({depth_dim: depth_index})

    # Crop bbox if requested
    if lat_min is not None or lat_max is not None:
        lo = lat_min if lat_min is not None else float(ds[lat_dim].min())
        hi = lat_max if lat_max is not None else float(ds[lat_dim].max())
        u = u.sel({lat_dim: slice(lo, hi)})
        v = v.sel({lat_dim: slice(lo, hi)})
    if lon_min is not None or lon_max is not None:
        lo = lon_min if lon_min is not None else float(ds[lon_dim].min())
        hi = lon_max if lon_max is not None else float(ds[lon_dim].max())
        u = u.sel({lon_dim: slice(lo, hi)})
        v = v.sel({lon_dim: slice(lo, hi)})

    # Regrid to fixed HxW if requested
    if regrid_hw is not None:
        H, W = regrid_hw
        lat_new = np.linspace(float(u[lat_dim].min()), float(u[lat_dim].max()), H)
        lon_new = np.linspace(float(u[lon_dim].min()), float(u[lon_dim].max()), W)
        u = u.interp({lat_dim: lat_new, lon_dim: lon_new})
        v = v.interp({lat_dim: lat_new, lon_dim: lon_new})

    # Load into memory and stack channels
    # Ensure ordering: (time, lat, lon)
    u = u.transpose(time_dim, lat_dim, lon_dim)
    v = v.transpose(time_dim, lat_dim, lon_dim)

    u_np = u.to_numpy()
    v_np = v.to_numpy()

    # Fill NaNs
    if fillna is not None:
        u_np = np.nan_to_num(u_np, nan=fillna)
        v_np = np.nan_to_num(v_np, nan=fillna)

    uv = np.stack([u_np, v_np], axis=1).astype(np.float32)  # (T,2,H,W)

    time = ds[time_dim].to_numpy()
    lat = u[lat_dim].to_numpy()
    lon = u[lon_dim].to_numpy()

    ds.close()
    return uv, time, lat, lon

class SlidingWindowUVDataset(Dataset):
    """
    Each item:
      X: (L,2,H,W)
      Y: (2,H,W)
    """
    def __init__(self, uv: np.ndarray, seq_len: int = 3, pred_horizon: int = 1):
        assert uv.ndim == 4 and uv.shape[1] == 2, "uv must be (T,2,H,W)"
        self.uv = uv
        self.L = int(seq_len)
        self.H = int(pred_horizon)
        if self.H != 1:
            raise NotImplementedError("This starter supports 1-step prediction; extend easily if needed.")

    def __len__(self) -> int:
        T = self.uv.shape[0]
        return max(0, T - self.L)

    def __getitem__(self, idx: int):
        t = idx + self.L
        x = self.uv[t-self.L:t]   # (L,2,H,W)
        y = self.uv[t]            # (2,H,W)
        return torch.from_numpy(x), torch.from_numpy(y)
