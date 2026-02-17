from __future__ import annotations
import torch
import torch.nn as nn
from .unet_convlstm import UNetConvLSTM

class UNetConvLSTMUncertainty(nn.Module):
    """
    Wraps UNetConvLSTM but outputs per-pixel mean + log-variance.
    Output:
      mu:     (B,2,H,W)
      logvar: (B,2,H,W)  (clamped for stability)
    """
    def __init__(self, base_ch: int = 32, lstm_ch: int = 256):
        super().__init__()
        # Create the same backbone but with a 4-channel head
        self.backbone = UNetConvLSTM(in_ch=2, out_ch=2, base=base_ch, lstm_ch=lstm_ch)
        in_ch = self.backbone.head.in_channels
        self.backbone.head = nn.Conv2d(in_ch, 4, kernel_size=1)

    def forward(self, x_seq):
        out = self.backbone(x_seq)          # (B,4,H,W)
        mu = out[:, :2]
        logvar = out[:, 2:]
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar

def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL averaged over batch/pixels/channels."""
    inv_var = torch.exp(-logvar)
    return 0.5 * torch.mean(inv_var * (y - mu) ** 2 + logvar)
