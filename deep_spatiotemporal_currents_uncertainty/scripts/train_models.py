from __future__ import annotations
import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Project Imports ---
# Ensure you have created src/models/thesis_variants.py as described previously
from src.models.thesis_variants import ThesisVariant
from src.data.cmems_dataset import (
    load_cmems_uv, SlidingWindowUVDataset,
    compute_minmax, apply_minmax, invert_minmax
)
from src.utils import seed_all, save_json

# --- 1. Loss Function (Aleatoric Uncertainty) ---
def uncertainty_loss(pred_mean, pred_logvar, target):
    """
    Computes Gaussian Negative Log Likelihood (NLL).
    Loss = 0.5 * ( exp(-s) * (y - y_hat)^2 + s )
    where s = log_var
    """
    # Squared error
    sq_diff = (pred_mean - target)**2
    # NLL
    loss = 0.5 * torch.exp(-pred_logvar) * sq_diff + 0.5 * pred_logvar
    return loss.mean()

# --- 2. Evaluation Loop ---
def evaluate(model, loader, device, stats=None, mode="unet_convlstm"):
    """
    Returns:
        avg_loss (float): Validation NLL (Normalized)
        avg_rmse (float): Validation RMSE (m/s, physical units)
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    n = 0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            
            # --- Reshape for Standard U-Net if needed ---
            if mode == "standard_unet":
                # Flatten Time and Channels: (B, T, C, H, W) -> (B, T*C, H, W)
                b, t, c, h, w = X.shape
                X = X.view(b, t * c, h, w)
            
            # Forward
            output = model(X)
            
            # Split Output
            pred_mean = output[:, :2]   # (u, v)
            pred_logvar = output[:, 2:] # (log_var_u, log_var_v)
            
            # 1. Calc NLL Loss (on normalized data)
            loss = uncertainty_loss(pred_mean, pred_logvar, Y)
            total_loss += loss.item() * X.shape[0]
            
            # 2. Calc RMSE (convert back to m/s for physical interpretation)
            if stats is not None:
                # Invert MinMax
                pred_uv_phys = invert_minmax(pred_mean.cpu().numpy(), stats)
                true_uv_phys = invert_minmax(Y.cpu().numpy(), stats)
                
                # RMSE calculation
                mse = np.mean((pred_uv_phys - true_uv_phys)**2)
                rmse_val = np.sqrt(mse)
            else:
                # Fallback to normalized RMSE if stats missing
                rmse_val = torch.sqrt(torch.mean((pred_mean - Y)**2)).item()

            total_rmse += rmse_val * X.shape[0]
            n += X.shape[0]

    avg_loss = total_loss / max(n, 1)
    avg_rmse = total_rmse / max(n, 1)
    return avg_loss, avg_rmse

# --- 3. Main Training Script ---
def parse_args():
    p = argparse.ArgumentParser(description="Train models for Thesis Comparison")
    
    # Data & IO
    p.add_argument("--data", type=str, required=True, help="Path to CMEMS NetCDF")
    p.add_argument("--out", type=str, required=True, help="Output folder")
    p.add_argument("--model-mode", type=str, default="unet_convlstm", 
                   choices=["unet_convlstm", "cnn_convlstm", "standard_unet"],
                   help="Thesis comparison mode")

    # Variables
    p.add_argument("--u-var", type=str, default="utotal")
    p.add_argument("--v-var", type=str, default="vtotal")
    p.add_argument("--depth-index", type=int, default=0)

    # Grid / Region
    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument("--regrid-h", type=int, default=None)
    p.add_argument("--regrid-w", type=int, default=None)

    # Training Hyperparams
    p.add_argument("--seq-len", type=int, default=3, help="Input history length")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=32, help="Base channels for U-Net")
    p.add_argument("--lstm-ch", type=int, default=256, help="Hidden channels for Bottleneck")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)

    return p.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)
    
    # Update output directory to include model mode (prevents overwriting)
    args.out = os.path.join(args.out, args.model_mode)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    
    print(f"--- Running Thesis Experiment: {args.model_mode} ---")
    print(f"Output Dir: {args.out}")

    # --- Data Preparation ---
    print("Loading data...")
    ds_full = load_cmems_uv(
        args.data, 
        u_var=args.u_var, v_var=args.v_var, 
        depth_idx=args.depth_index,
        lat_range=(args.lat_min, args.lat_max) if args.lat_min else None,
        lon_range=(args.lon_min, args.lon_max) if args.lon_min else None,
        regrid_shape=(args.regrid_h, args.regrid_w) if args.regrid_h else None
    ) # Returns (Time, 2, H, W)

    # Split Train/Val
    t_split = int(len(ds_full) * args.train_frac)
    train_data = ds_full[:t_split]
    val_data = ds_full[t_split:]

    # Compute Stats (MinMax) on TRAIN only
    print("Computing statistics...")
    stats = compute_minmax(train_data)
    
    # Create Datasets
    train_ds = SlidingWindowUVDataset(train_data, seq_len=args.seq_len, stats=stats)
    val_ds   = SlidingWindowUVDataset(val_data, seq_len=args.seq_len, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True)

    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logic for input channels
    # Standard U-Net stacks frames: Inputs = (u1,v1, u2,v2, u3,v3) -> 2 * seq_len
    # ConvLSTM models take sequential input, so in_ch is just 2 (u,v) per frame.
    current_in_ch = 2 * args.seq_len if args.model_mode == "standard_unet" else 2

    model = ThesisVariant(
        in_ch=current_in_ch,
        out_ch=4, # u_mean, v_mean, u_logvar, v_logvar
        base_ch=args.base_ch,
        lstm_ch=args.lstm_ch,
        seq_len=args.seq_len,
        mode=args.model_mode
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    history = []
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        n = 0

        loop = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs}")
        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)
            # X shape: (B, T, 2, H, W)
            
            # --- Reshape for Standard U-Net ---
            if args.model_mode == "standard_unet":
                # Flatten T and C: (B, T*2, H, W)
                b, t, c, h, w = X.shape
                X = X.view(b, t * c, h, w)
            
            # Forward
            output = model(X)
            
            # Split: first 2 ch = Mean, last 2 ch = LogVar
            pred_mean = output[:, :2]
            pred_logvar = output[:, 2:]
            
            # Loss
            loss = uncertainty_loss(pred_mean, pred_logvar, Y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * X.shape[0]
            n += X.shape[0]
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / max(n, 1)

        # Validation
        val_loss, val_rmse = evaluate(model, val_loader, device, stats, mode=args.model_mode)
        
        # Logging
        dt = time.time() - t0
        print(f"Epoch {epoch}: Train NLL={avg_train_loss:.4f}, Val NLL={val_loss:.4f}, Val RMSE={val_rmse:.4f} m/s, Time={dt:.1f}s")
        
        row = {
            "epoch": epoch,
            "train_nll": avg_train_loss,
            "val_nll": val_loss,
            "val_rmse": val_rmse,
            "time": dt
        }
        history.append(row)
        save_json(os.path.join(args.out, "history.json"), history)

        # Checkpointing
        ckpt = {
            "model": model.state_dict(),
            "stats": stats.__dict__, # Save min/max for inference
            "args": vars(args)
        }
        
        # Save Last
        torch.save(ckpt, os.path.join(args.out, "checkpoints", "last.pt"))
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.out, "checkpoints", "best.pt"))
            print(f"  [+] New Best Model Saved! (Val NLL: {val_loss:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    main()