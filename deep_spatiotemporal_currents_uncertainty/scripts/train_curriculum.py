
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Import your existing modules
from src.data.cmems_dataset import (
    load_cmems_uv, SlidingWindowUVDataset
)
from src.models.unet_convlstm_unc import UNetConvLSTMUncertainty, gaussian_nll
from src.utils import seed_all, save_json

# --- 1. Stats & Normalization (Matching your snippet) ---
class ZScoreStats:
    def __init__(self, u_mean, u_std, v_mean, v_std):
        self.u_mean = u_mean
        self.u_std = u_std
        self.v_mean = v_mean
        self.v_std = v_std
    
    def to_dict(self):
        return {
            "u_mean": float(self.u_mean), "u_std": float(self.u_std),
            "v_mean": float(self.v_mean), "v_std": float(self.v_std)
        }

def compute_zscore(uv):
    # uv: (T,2,H,W)
    u = uv[:,0]
    v = uv[:,1]
    return ZScoreStats(
        np.mean(u), np.std(u) + 1e-8,
        np.mean(v), np.std(v) + 1e-8
    )

def apply_zscore(uv, stats):
    uv_n = uv.copy()
    uv_n[:,0] = (uv_n[:,0] - stats.u_mean) / stats.u_std
    uv_n[:,1] = (uv_n[:,1] - stats.v_mean) / stats.v_std
    return uv_n

# --- 2. Custom Dataset for Curriculum Learning ---
class SequenceUVDataset(SlidingWindowUVDataset):
    """
    Wraps the parent logic but returns a SEQUENCE of future targets (t+1...t+H).
    Assumes data is ALREADY normalized.
    """
    def __init__(self, data, seq_len=12, horizon=6):
        # We manually initialize to avoid relying on parent 'min_val' logic
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

    def __getitem__(self, i):
        start_x = i
        end_x = i + self.seq_len
        end_y = end_x + self.horizon

        X = self.data[start_x : end_x]       # History: (Seq, 2, H, W)
        Y_seq = self.data[end_x : end_y]     # Future:  (Horizon, 2, H, W)

        return torch.from_numpy(X).float(), torch.from_numpy(Y_seq).float()

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

# --- 3. Helper: Teacher Forcing Schedule ---
def get_tf_ratio(epoch, k_decay):
    k = float(k_decay)
    return k / (k + np.exp(epoch / k))

# --- 4. Main Training Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    
    # Curriculum Args
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--k-decay", type=float, default=15.0)
    
    # Standard Args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.7)
    
    # Data Args
    parser.add_argument("--regrid-h", type=int, default=None)
    parser.add_argument("--regrid-w", type=int, default=None)
    parser.add_argument("--u-var", type=str, default="utotal")
    parser.add_argument("--v-var", type=str, default="vtotal")

    args, _ = parser.parse_known_args()
    
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # 1. Load Data (Corrected Unpacking)
    print(f"Loading {args.data}...")
    regrid_hw = (args.regrid_h, args.regrid_w) if args.regrid_h else None
    
    # FIX: Unpack 4 values, ignoring time/lat/lon
    uv, _, _, _ = load_cmems_uv(
        args.data, 
        u_var=args.u_var, 
        v_var=args.v_var,
        regrid_hw=regrid_hw,
        depth_index=0
    )
    # uv is already (T, 2, H, W)

    # 2. Preprocess (Z-Score)
    split_idx = int(len(uv) * args.train_frac)
    train_data_raw = uv[:split_idx]
    val_data_raw = uv[split_idx:]

    print("Computing Z-Score stats on training set...")
    stats = compute_zscore(train_data_raw)
    
    # Save stats for inference later
    save_json(os.path.join(args.out, "stats.json"), stats.to_dict())

    train_data = apply_zscore(train_data_raw, stats)
    val_data   = apply_zscore(val_data_raw, stats)

    # 3. Create Datasets
    ds_train = SequenceUVDataset(train_data, seq_len=args.seq_len, horizon=args.horizon)
    # Validation dataset can be standard or sequence, we use sequence to track recursive loss
    ds_val   = SequenceUVDataset(val_data, seq_len=args.seq_len, horizon=args.horizon)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = UNetConvLSTMUncertainty(base_ch=32, lstm_ch=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    print(f"Starting Curriculum Training (Horizon={args.horizon}, Decay={args.k_decay})")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        tf_ratio = get_tf_ratio(epoch, args.k_decay)
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch} | TF={tf_ratio:.2f}")
        for X, Y_seq in pbar:
            X, Y_seq = X.to(device), Y_seq.to(device)
            
            # Initial Input
            curr_input = X # (B, Seq, 2, H, W)
            loss_seq = 0
            
            # Autoregressive Loop
            for t in range(args.horizon):
                target_frame = Y_seq[:, t]  # (B, 2, H, W)
                
                # Predict
                mu, logvar = model(curr_input)
                
                # Loss (replace the old single line)
                loss_mse = nn.functional.mse_loss(mu, target_frame)
                loss_nll = gaussian_nll(mu, logvar, target_frame)
                loss_seq += loss_nll + loss_mse
                
                # Scheduled Sampling
                if np.random.random() < tf_ratio:
                    next_in = target_frame
                else:
                    next_in = mu
                
                # Update Window
                curr_input = torch.cat([curr_input[:, 1:], next_in.unsqueeze(1)], dim=1)

            loss_final = loss_seq / args.horizon
            
            opt.zero_grad()
            loss_final.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            train_loss += loss_final.item()
            n_batches += 1
            pbar.set_postfix({"loss": train_loss/n_batches})
        
        avg_train_loss = train_loss / n_batches
        history.append({"epoch": epoch, "loss": avg_train_loss, "tf_ratio": tf_ratio})
        save_json(os.path.join(args.out, "history.json"), history)
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))

if __name__ == "__main__":
    main()