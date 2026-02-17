from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convlstm import ConvLSTM

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class UNetConvLSTM(nn.Module):
    """
    Input:  (B,T,2,H,W)
    Output: (B,2,H,W)
    Design:
      - Encode each frame with U-Net encoder
      - ConvLSTM on bottleneck features across time
      - Decode using skip connections from the *last* frame encoder (simple & effective starter)
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 2, base: int = 32, lstm_ch: int = 256):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)          # H
        self.enc2 = conv_block(base, base*2)         # H/2
        self.enc3 = conv_block(base*2, base*4)       # H/4
        self.enc4 = conv_block(base*4, base*8)       # H/8

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base*8, lstm_ch)
        self.temporal = ConvLSTM(in_ch=lstm_ch, hid_ch=lstm_ch, k=3)

        self.up3 = nn.ConvTranspose2d(lstm_ch, base*8, 2, stride=2)
        self.dec3 = conv_block(base*8 + base*8, base*8)

        self.up2 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec2 = conv_block(base*4 + base*4, base*4)

        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec1 = conv_block(base*2 + base*2, base*2)

        self.up0 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec0 = conv_block(base + base, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def _encode(self, x):
        # x: (B,2,H,W)
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)
        p3 = self.pool(s3)
        s4 = self.enc4(p3)
        p4 = self.pool(s4)
        b = self.bottleneck(p4)
        return b, (s1, s2, s3, s4)

    def forward(self, x_seq):
        # x_seq: (B,T,2,H,W)
        B, T, C, H, W = x_seq.shape

        bottlenecks = []
        last_skips = None
        for t in range(T):
            b, skips = self._encode(x_seq[:,t])
            bottlenecks.append(b)
            last_skips = skips

        b_seq = torch.stack(bottlenecks, dim=1)  # (B,T,lstm_ch,H/16,W/16)
        h_last = self.temporal(b_seq)            # (B,lstm_ch,H/16,W/16)

        s1, s2, s3, s4 = last_skips

        x = self.up3(h_last)                 # -> H/8
        x = torch.cat([x, s4], dim=1)
        x = self.dec3(x)

        x = self.up2(x)                      # -> H/4
        x = torch.cat([x, s3], dim=1)
        x = self.dec2(x)

        x = self.up1(x)                      # -> H/2
        x = torch.cat([x, s2], dim=1)
        x = self.dec1(x)

        x = self.up0(x)                      # -> H
        x = torch.cat([x, s1], dim=1)
        x = self.dec0(x)

        out = self.head(x)
        return out
