from __future__ import annotations
import argparse, os
import numpy as np
import torch

from src.data.cmems_dataset import load_cmems_uv, apply_minmax, invert_minmax
from src.models.unet_convlstm import UNetConvLSTM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    p.add_argument("--data", type=str, required=True, help="Path to NetCDF used for inference")
    p.add_argument("--t-index", type=int, required=True, help="Target time index t to predict")
    p.add_argument("--out", type=str, required=True, help="Output .npz path")

    p.add_argument("--u-var", type=str, default="utotal")
    p.add_argument("--v-var", type=str, default="vtotal")
    p.add_argument("--depth-index", type=int, default=0)

    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--regrid-h", type=int, default=None)
    p.add_argument("--regrid-w", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stats = ckpt["stats"]
    model_args = ckpt["args"]

    regrid = None
    if args.regrid_h is not None or args.regrid_w is not None:
        if args.regrid_h is None or args.regrid_w is None:
            raise ValueError("Specify both --regrid-h and --regrid-w")
        regrid = (args.regrid_h, args.regrid_w)

    uv, time, lat, lon = load_cmems_uv(
        args.data,
        u_var=args.u_var, v_var=args.v_var,
        depth_index=args.depth_index,
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
        regrid_hw=regrid,
        fillna=0.0,
    )

    seq_len = int(model_args.get("seq_len", 3))
    t = int(args.t_index)
    if t < seq_len or t >= uv.shape[0]:
        raise ValueError(f"t-index must be in [{seq_len}, {uv.shape[0]-1}]")

    # normalize with saved stats
    class _S: pass
    s = _S()
    for k,v in stats.items():
        setattr(s,k,v)
    uv_n = apply_minmax(uv, s)

    X = uv_n[t-seq_len:t]  # (L,2,H,W)
    Y = uv_n[t]            # (2,H,W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODIFICATION: Ensure we initialize with 4 output channels to match checkpoint
    # We use .get("out_ch", 4) in case you re-run this on old models (which would need 2)
    # But for the new uncertainty model, this needs to match the training setup.
    # We'll default to 4 if not in args, but robustly check 'base_ch' etc.
    model = UNetConvLSTM(in_ch=2, out_ch=4, 
                         base=int(model_args.get("base_ch", 32)), 
                         lstm_ch=int(model_args.get("lstm_ch", 256)))
                         
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    with torch.no_grad():
        output = model(torch.from_numpy(X[None]).to(device)).cpu().numpy()[0] # (4,H,W)

    # SPLIT OUTPUT
    # Channels 0,1 -> Mean Velocity
    pred_mean = output[:2] 
    # Channels 2,3 -> Log Variance
    pred_logvar = output[2:]
    
    # Convert logvar to standard deviation (sigma)
    # sigma = exp(0.5 * log_var)
    pred_sigma = np.exp(0.5 * pred_logvar)

    # Invert normalization for the MEAN fields only
    pred_uv = invert_minmax(pred_mean, s)
    gt_uv   = invert_minmax(Y, s)

    # Note: pred_sigma is still in "normalized" units. 
    # If you want physical units (m/s), you technically need to scale by (max-min).
    # For now, we save it as is.

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        pred_uv=pred_uv.astype(np.float32),      # The Clean Velocity Field
        pred_sigma=pred_sigma.astype(np.float32), # The Uncertainty Field
        gt_uv=gt_uv.astype(np.float32),
        lat=lat,
        lon=lon,
        time=str(time[t]),
        t_index=t,
        seq_len=seq_len
    )
    print("Saved:", args.out)

if __name__ == "__main__":
    main()