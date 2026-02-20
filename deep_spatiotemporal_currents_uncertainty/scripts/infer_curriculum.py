"""
infer_curriculum.py

Autoregressive inference for a curriculum-trained UNetConvLSTMUncertainty model.
Produces an .npz with predictions + uncertainty for N steps.

Usage:
    python infer_curriculum.py \
        --ckpt runs/curriculum_unc/checkpoints/best.pt \
        --data /path/to/cmems.nc \
        --t-index 100 \
        --steps 12 \
        --out results/forecast_t100.npz
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

from src.data.cmems_dataset import load_cmems_uv
from src.models.unet_convlstm_unc import UNetConvLSTMUncertainty


# =====================================================================
# Helpers (same normalisation as training)
# =====================================================================
class ZScoreStats:
    def __init__(self, u_mean, u_std, v_mean, v_std):
        self.u_mean = u_mean
        self.u_std = u_std
        self.v_mean = v_mean
        self.v_std = v_std


def apply_zscore(uv: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    uv_n = uv.copy()
    uv_n[:, 0] = (uv_n[:, 0] - stats.u_mean) / stats.u_std
    uv_n[:, 1] = (uv_n[:, 1] - stats.v_mean) / stats.v_std
    return uv_n


def inverse_zscore(img_norm: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    img_phys = np.zeros_like(img_norm)
    img_phys[0] = img_norm[0] * stats.u_std + stats.u_mean
    img_phys[1] = img_norm[1] * stats.v_std + stats.v_mean
    return img_phys


def inverse_zscore_sigma(sigma_norm: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    sigma_phys = np.zeros_like(sigma_norm)
    sigma_phys[0] = sigma_norm[0] * stats.u_std
    sigma_phys[1] = sigma_norm[1] * stats.v_std
    return sigma_phys


# =====================================================================
# Autoregressive rollout
# =====================================================================
@torch.no_grad()
def predict_autoregressive(
    model: UNetConvLSTMUncertainty,
    initial_seq: torch.Tensor,   # (1, seq_len, 2, H, W)
    steps: int = 12,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        preds:   (steps, 2, H, W) — predicted means  (normalised)
        sigmas:  (steps, 2, H, W) — predicted std-dev (normalised)
    """
    model.eval()
    current_seq = initial_seq.clone().to(device)
    preds, sigmas = [], []

    for _ in range(steps):
        mu, logvar = model(current_seq)

        preds.append(mu.cpu().numpy()[0])                           # (2, H, W)
        sigmas.append(torch.exp(0.5 * logvar).cpu().numpy()[0])     # (2, H, W)

        next_frame = mu.unsqueeze(1)  # (1, 1, 2, H, W)
        current_seq = torch.cat([current_seq[:, 1:], next_frame], dim=1)

    return np.array(preds), np.array(sigmas)


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Curriculum AR inference")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--t-index", type=int, required=True,
                   help="Index in the FULL dataset (not val split) to start prediction from")
    p.add_argument("--steps", type=int, default=12, help="Number of AR steps")
    p.add_argument("--out", type=str, required=True, help="Output .npz path")

    p.add_argument("--u-var", type=str, default="utotal")
    p.add_argument("--v-var", type=str, default="vtotal")
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--regrid-h", type=int, default=64)
    p.add_argument("--regrid-w", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt["args"]
    stats = ZScoreStats(**ckpt["stats"])
    seq_len = ckpt["seq_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetConvLSTMUncertainty(
        base_ch=int(saved_args.get("base_ch", 32)),
        lstm_ch=int(saved_args.get("lstm_ch", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.ckpt}  (stage={ckpt.get('stage')}, ar={ckpt.get('ar_steps')})")

    # ---- Load data ----
    regrid = (args.regrid_h, args.regrid_w) if args.regrid_h and args.regrid_w else None
    uv, time_arr, lat, lon = load_cmems_uv(
        args.data, u_var=args.u_var, v_var=args.v_var,
        depth_index=args.depth_index, regrid_hw=regrid,
    )

    t = args.t_index
    if t < seq_len or t + args.steps >= uv.shape[0]:
        raise ValueError(
            f"t-index {t} out of range. Need [{seq_len}, {uv.shape[0] - args.steps - 1}]"
        )

    # Normalise and build input window
    uv_n = apply_zscore(uv, stats)
    X = uv_n[t - seq_len : t]  # (seq_len, 2, H, W)
    input_seq = torch.from_numpy(X[None]).float().to(device)

    # ---- Predict ----
    preds_norm, sigmas_norm = predict_autoregressive(model, input_seq, steps=args.steps, device=str(device))

    # ---- De-normalise ----
    preds_phys = np.array([inverse_zscore(p, stats) for p in preds_norm])
    sigmas_phys = np.array([inverse_zscore_sigma(s, stats) for s in sigmas_norm])

    # Ground truth for each predicted step
    gt_phys = uv[t : t + args.steps]  # (steps, 2, H, W)

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        preds_uv=preds_phys.astype(np.float32),      # (steps, 2, H, W) physical
        preds_sigma=sigmas_phys.astype(np.float32),   # (steps, 2, H, W) physical
        gt_uv=gt_phys.astype(np.float32),             # (steps, 2, H, W) physical
        lat=lat,
        lon=lon,
        t_index=t,
        seq_len=seq_len,
        steps=args.steps,
        time_start=str(time_arr[t]),
    )
    print(f"Saved {args.steps}-step forecast to {args.out}")


if __name__ == "__main__":
    main()
