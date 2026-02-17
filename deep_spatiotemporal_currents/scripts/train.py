from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cmems_dataset import (
    load_cmems_uv, SlidingWindowUVDataset,
    compute_minmax, apply_minmax
)
from src.models.unet_convlstm import UNetConvLSTM
from src.utils import seed_all, save_json, rmse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to CMEMS-style NetCDF")
    p.add_argument("--out", type=str, required=True, help="Output run directory")

    p.add_argument("--u-var", type=str, default="utotal")
    p.add_argument("--v-var", type=str, default="vtotal")
    p.add_argument("--depth-index", type=int, default=0)

    p.add_argument("--seq-len", type=int, default=3, help="Number of past frames")
    p.add_argument("--train-frac", type=float, default=0.7, help="Time split fraction for training")

    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--regrid-h", type=int, default=None)
    p.add_argument("--regrid-w", type=int, default=None)

    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--lstm-ch", type=int, default=256)

    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n = 0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = torch.mean((pred - Y) ** 2)
        total_loss += float(loss.item()) * X.shape[0]
        total_rmse += float(rmse(pred, Y).item()) * X.shape[0]
        n += X.shape[0]
    return total_loss / max(n,1), total_rmse / max(n,1)

def main():
    args = parse_args()
    seed_all(args.seed)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)

    regrid = None
    if args.regrid_h is not None or args.regrid_w is not None:
        if args.regrid_h is None or args.regrid_w is None:
            raise ValueError("Specify both --regrid-h and --regrid-w")
        regrid = (args.regrid_h, args.regrid_w)

    uv, t, lat, lon = load_cmems_uv(
        args.data,
        u_var=args.u_var,
        v_var=args.v_var,
        depth_index=args.depth_index,
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
        regrid_hw=regrid,
        fillna=0.0,
    )

    T = uv.shape[0]
    split_t = int(T * args.train_frac)
    # Note: because dataset uses windows, we split on raw time then build datasets per split.
    uv_train = uv[:split_t]
    uv_val   = uv[split_t:]

    stats = compute_minmax(uv_train)
    uv_train_n = apply_minmax(uv_train, stats)
    uv_val_n   = apply_minmax(uv_val, stats)

    # Build datasets
    train_ds = SlidingWindowUVDataset(uv_train_n, seq_len=args.seq_len, pred_horizon=1)
    val_ds   = SlidingWindowUVDataset(uv_val_n,   seq_len=args.seq_len, pred_horizon=1)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetConvLSTM(in_ch=2, out_ch=2, base=args.base_ch, lstm_ch=args.lstm_ch).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_meta = {
        "args": vars(args),
        "device": str(device),
        "data_shape": {"T": int(uv.shape[0]), "C": 2, "H": int(uv.shape[2]), "W": int(uv.shape[3])},
        "norm": stats.__dict__,
    }
    save_json(os.path.join(args.out, "run_meta.json"), run_meta)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for X, Y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = torch.mean((pred - Y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * X.shape[0]
            n += X.shape[0]

        train_loss /= max(n,1)
        val_loss, val_rmse = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "val_rmse": val_rmse,
            "sec": time.time() - t0,
        }
        history.append(row)
        save_json(os.path.join(args.out, "history.json"), history)

        # checkpoint
        ckpt = {
            "model": model.state_dict(),
            "stats": stats.__dict__,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out, "checkpoints", "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out, "checkpoints", "best.pt"))

        print(f"[{epoch:03d}] train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  val_rmse={val_rmse:.6f}")

    print("Done. Best val MSE:", best_val)

if __name__ == "__main__":
    main()
