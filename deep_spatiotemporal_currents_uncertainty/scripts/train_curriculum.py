"""
train_curriculum.py

Curriculum autoregressive training for UNetConvLSTM with uncertainty.

Inspired by GraphCast (Lam et al., 2023) and FuXi (Chen et al., 2023):
  - Phase 1:  pretrain on 1-step loss  (model learns basic dynamics)
  - Phase 2+: gradually increase AR steps (model learns to handle its own errors)

The curriculum schedule is:
    ar_steps = [1, 2, 4, 8, 12]
Each stage trains for --epochs-per-stage epochs.

Usage:
    python train_curriculum.py \
        --data /path/to/cmems.nc \
        --out runs/curriculum_unc \
        --seq-len 12 \
        --max-horizon 12 \
        --epochs-per-stage 100 \
        --batch 4 \
        --lr 1e-3 \
        --base-ch 32 \
        --lstm-ch 256

The script saves:
    <out>/checkpoints/best.pt         best model by val RMSE
    <out>/checkpoints/last.pt         latest model
    <out>/checkpoints/stage_N.pt      model at end of each curriculum stage
    <out>/history.jsonl               per-epoch log
    <out>/run_meta.json               full config
"""
from __future__ import annotations
import argparse, os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Your existing imports (adjust paths if needed) ----
from src.data.cmems_dataset import load_cmems_uv
from src.data.cmems_dataset_ar import SlidingWindowMultiStep
from src.models.unet_convlstm_unc import UNetConvLSTMUncertainty, gaussian_nll


# =====================================================================
# Helpers
# =====================================================================
class ZScoreStats:
    def __init__(self, u_mean, u_std, v_mean, v_std):
        self.u_mean = u_mean
        self.u_std = u_std
        self.v_mean = v_mean
        self.v_std = v_std


def compute_zscore(uv: np.ndarray) -> ZScoreStats:
    u = uv[:, 0]
    v = uv[:, 1]
    return ZScoreStats(
        u_mean=float(np.mean(u)),
        u_std=float(np.std(u) + 1e-8),
        v_mean=float(np.mean(v)),
        v_std=float(np.std(v) + 1e-8),
    )


def apply_zscore(uv: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    uv_n = uv.copy()
    uv_n[:, 0] = (uv_n[:, 0] - stats.u_mean) / stats.u_std
    uv_n[:, 1] = (uv_n[:, 1] - stats.v_mean) / stats.v_std
    return uv_n


def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Core: autoregressive forward + loss (differentiable, for training)
# =====================================================================
def ar_forward_loss(
    model: UNetConvLSTMUncertainty,
    X: torch.Tensor,            # (B, seq_len, 2, H, W)
    Y_seq: torch.Tensor,        # (B, max_horizon, 2, H, W)
    ar_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run `ar_steps` of autoregressive rollout **with gradients**,
    accumulating the Gaussian NLL loss at each step.

    Returns (loss, rmse_of_means).
    """
    B = X.shape[0]
    current_input = X                       # (B, seq_len, 2, H, W)
    total_nll = torch.tensor(0.0, device=X.device)
    total_mse = torch.tensor(0.0, device=X.device)

    for t in range(ar_steps):
        mu, logvar = model(current_input)   # each (B, 2, H, W)

        target = Y_seq[:, t]                # (B, 2, H, W)
        step_nll = gaussian_nll(mu, logvar, target)
        total_nll = total_nll + step_nll
        total_mse = total_mse + torch.mean((mu - target) ** 2)

        # Shift window: drop oldest frame, append predicted mean
        next_frame = mu.unsqueeze(1)        # (B, 1, 2, H, W)
        current_input = torch.cat([current_input[:, 1:], next_frame], dim=1)

    avg_nll = total_nll / ar_steps
    avg_rmse = torch.sqrt(total_mse / ar_steps)
    return avg_nll, avg_rmse


# =====================================================================
# Validation (no grad, always runs full max_horizon rollout)
# =====================================================================
@torch.no_grad()
def validate(
    model: UNetConvLSTMUncertainty,
    loader: DataLoader,
    ar_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_rmse = 0.0
    n = 0

    for X, Y_seq in loader:
        X = X.to(device)
        Y_seq = Y_seq.to(device)

        nll, r = ar_forward_loss(model, X, Y_seq, ar_steps=ar_steps)
        bs = X.shape[0]
        total_nll += nll.item() * bs
        total_rmse += r.item() * bs
        n += bs

    return total_nll / max(n, 1), total_rmse / max(n, 1)


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Curriculum AR training")
    p.add_argument("--data", type=str, required=True, help="Path to CMEMS NetCDF")
    p.add_argument("--out", type=str, required=True, help="Output run directory")

    # Data loading
    p.add_argument("--u-var", type=str, default="utotal")
    p.add_argument("--v-var", type=str, default="vtotal")
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--regrid-h", type=int, default=64)
    p.add_argument("--regrid-w", type=int, default=64)
    p.add_argument("--train-frac", type=float, default=0.7)

    # Sequence / horizon
    p.add_argument("--seq-len", type=int, default=12, help="Input sequence length")
    p.add_argument("--max-horizon", type=int, default=12, help="Max prediction steps")

    # Curriculum schedule (comma-separated)
    p.add_argument(
        "--curriculum", type=str, default="1,2,4,8,12",
        help="Comma-separated list of AR steps per stage (default: 1,2,4,8,12)"
    )
    p.add_argument("--epochs-per-stage", type=int, default=100)

    # Optimiser
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay-per-stage", type=float, default=0.5,
                   help="Multiply LR by this factor at each new stage")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (important for multi-step backprop)")

    # Model
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--lstm-ch", type=int, default=256)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    seed_all(args.seed)

    curriculum = [int(x) for x in args.curriculum.split(",")]
    print(f"Curriculum schedule: {curriculum}")
    print(f"Epochs per stage:   {args.epochs_per_stage}")

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and normalise data
    # ------------------------------------------------------------------
    regrid = (args.regrid_h, args.regrid_w) if args.regrid_h and args.regrid_w else None

    uv, time_arr, lat, lon = load_cmems_uv(
        args.data,
        u_var=args.u_var,
        v_var=args.v_var,
        depth_index=args.depth_index,
        regrid_hw=regrid,
    )
    print(f"Loaded data: {uv.shape}")  # (T, 2, H, W)

    T = uv.shape[0]
    split_t = int(T * args.train_frac)
    uv_train, uv_val = uv[:split_t], uv[split_t:]

    stats = compute_zscore(uv_train)
    uv_train_n = apply_zscore(uv_train, stats)
    uv_val_n = apply_zscore(uv_val, stats)

    # ------------------------------------------------------------------
    # 2. Datasets (multi-step targets)
    # ------------------------------------------------------------------
    train_ds = SlidingWindowMultiStep(uv_train_n, seq_len=args.seq_len, max_horizon=args.max_horizon)
    val_ds = SlidingWindowMultiStep(uv_val_n, seq_len=args.seq_len, max_horizon=args.max_horizon)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")

    # ------------------------------------------------------------------
    # 3. Model + optimiser
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNetConvLSTMUncertainty(base_ch=args.base_ch, lstm_ch=args.lstm_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Save run metadata
    # ------------------------------------------------------------------
    run_meta = {
        "args": vars(args),
        "curriculum": curriculum,
        "device": str(device),
        "data_shape": list(uv.shape),
        "stats": stats.__dict__,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    save_json(os.path.join(args.out, "run_meta.json"), run_meta)

    history_path = os.path.join(args.out, "history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    best_val_rmse = float("inf")
    global_epoch = 0

    # ------------------------------------------------------------------
    # 5. Curriculum training loop
    # ------------------------------------------------------------------
    for stage_idx, ar_steps in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: ar_steps = {ar_steps}")
        print(f"{'='*60}")

        # Decay learning rate for stages after the first
        if stage_idx > 0:
            for pg in opt.param_groups:
                pg["lr"] *= args.lr_decay_per_stage
            print(f"  LR decayed to {opt.param_groups[0]['lr']:.2e}")

        for local_epoch in range(args.epochs_per_stage):
            t0 = time.time()
            model.train()
            train_nll_sum = 0.0
            n_train = 0

            for X, Y_seq in tqdm(
                train_loader,
                desc=f"S{stage_idx} E{local_epoch:03d} (ar={ar_steps})",
                leave=False,
            ):
                X = X.to(device)
                Y_seq = Y_seq.to(device)

                nll, _ = ar_forward_loss(model, X, Y_seq, ar_steps=ar_steps)

                opt.zero_grad(set_to_none=True)
                nll.backward()

                # Gradient clipping — critical for multi-step backprop stability
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                opt.step()

                bs = X.shape[0]
                train_nll_sum += nll.item() * bs
                n_train += bs

            train_nll = train_nll_sum / max(n_train, 1)
            elapsed = time.time() - t0

            # ---- Validate ----
            do_val = (local_epoch % args.val_every == 0) or (local_epoch == args.epochs_per_stage - 1)
            if do_val:
                val_nll, val_rmse = validate(model, val_loader, ar_steps=ar_steps, device=device)
                print(
                    f"  [{global_epoch:04d}] stage={stage_idx} ar={ar_steps} "
                    f"train_nll={train_nll:.6f}  val_nll={val_nll:.6f}  "
                    f"val_rmse={val_rmse:.6f}  ({elapsed:.1f}s)"
                )
            else:
                val_nll, val_rmse = float("nan"), float("nan")
                print(
                    f"  [{global_epoch:04d}] stage={stage_idx} ar={ar_steps} "
                    f"train_nll={train_nll:.6f}  ({elapsed:.1f}s)"
                )

            # ---- Log ----
            row = {
                "global_epoch": global_epoch,
                "stage": stage_idx,
                "ar_steps": ar_steps,
                "local_epoch": local_epoch,
                "train_nll": float(train_nll),
                "val_nll": float(val_nll) if do_val else None,
                "val_rmse": float(val_rmse) if do_val else None,
                "lr": float(opt.param_groups[0]["lr"]),
                "sec": round(elapsed, 1),
            }
            with open(history_path, "a") as f:
                f.write(json.dumps(row) + "\n")

            # ---- Checkpoints ----
            ckpt_payload = {
                "global_epoch": global_epoch,
                "stage": stage_idx,
                "ar_steps": ar_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_rmse": best_val_rmse if best_val_rmse < float("inf") else None,
                "stats": stats.__dict__,
                "seq_len": args.seq_len,
                "max_horizon": args.max_horizon,
                "curriculum": curriculum,
                "args": vars(args),
            }

            torch.save(ckpt_payload, os.path.join(ckpt_dir, "last.pt"))

            if do_val and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
                print(f"    -> New best val_rmse={val_rmse:.6f}")

            global_epoch += 1

        # ---- Save end-of-stage checkpoint ----
        torch.save(ckpt_payload, os.path.join(ckpt_dir, f"stage_{stage_idx}_ar{ar_steps}.pt"))
        print(f"  Saved stage checkpoint: stage_{stage_idx}_ar{ar_steps}.pt")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training complete.  Best val_rmse = {best_val_rmse:.6f}")
    print(f"  Best:    {os.path.join(ckpt_dir, 'best.pt')}")
    print(f"  Last:    {os.path.join(ckpt_dir, 'last.pt')}")
    print(f"  History: {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
