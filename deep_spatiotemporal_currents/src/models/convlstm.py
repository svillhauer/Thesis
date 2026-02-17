from __future__ import annotations
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        padding = k // 2
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=k, padding=padding)

    def forward(self, x, h, c):
        # x: (B,in,H,W), h/c: (B,hid,H,W)
        if h is None:
            B, _, H, W = x.shape
            h = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)

        cat = torch.cat([x, h], dim=1)
        gates = self.conv(cat)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, k=k)

    def forward(self, x_seq):
        # x_seq: (B,T,C,H,W)
        B, T, C, H, W = x_seq.shape
        h, c = None, None
        for t in range(T):
            h, c = self.cell(x_seq[:,t], h, c)
        return h  # last hidden state (B,hid,H,W)
