import torch
import torch.nn as nn

class ThesisVariant(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32, lstm_ch=256, seq_len=3, mode='unet_convlstm'):
        """
        Args:
            mode (str): Options are:
                'unet_convlstm' (Proposed Method: Skips + LSTM)
                'cnn_convlstm'  (Ablation 1: No Skips + LSTM)
                'standard_unet' (Ablation 2: Skips + No LSTM)
        """
        super().__init__()
        self.mode = mode
        self.seq_len = seq_len
        
        # --- Encoder (Downsampling) ---
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        
        # --- Bottleneck ---
        # If using LSTM, input dim is preserved. If Standard U-Net, we collapse time into channels.
        if 'lstm' in mode:
            # ConvLSTM takes a sequence
            self.bottleneck = ConvLSTMCell(input_dim=base_ch * 8, hidden_dim=lstm_ch, kernel_size=(3,3), bias=True)
            bot_out_ch = lstm_ch
        else:
            # Standard U-Net: Bottleneck is just a Conv block. 
            # We treat the "time" dimension as extra channels in the encoder if needed, 
            # but usually for U-Net we stack inputs at the start.
            # Here, we assume the Encoder processed frames independently or stacked. 
            # Simplified: We use a standard Conv layer instead of LSTM.
            self.bottleneck = DoubleConv(base_ch * 8, lstm_ch)
            bot_out_ch = lstm_ch

        # --- Decoder (Upsampling) ---
        self.up1 = Up(bot_out_ch, base_ch * 4, use_skips=('unet' in mode))
        self.up2 = Up(base_ch * 4, base_ch * 2, use_skips=('unet' in mode))
        self.up3 = Up(base_ch * 2, base_ch, use_skips=('unet' in mode))
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Time, Channels, H, W)
        b, t, c, h, w = x.shape
        
        if 'lstm' in self.mode:
            # --- Method 1 & 2: Time-Distributed Encoder ---
            # Reshape to (B*T, C, H, W) to pass through encoder
            x_flat = x.view(b * t, c, h, w)
            
            x1 = self.inc(x_flat)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3) # Shape: (B*T, 256, H/8, W/8)
            
            # Reshape back to sequence for LSTM: (B, T, Features, H, W)
            lstm_in = x4.view(b, t, -1, x4.shape[-2], x4.shape[-1])
            
            # ConvLSTM Step (Process sequence, keep last state)
            # Assuming simple recurrence on the sequence
            h_state, c_state = None, None
            for t_step in range(t):
                h_state, c_state = self.bottleneck(lstm_in[:, t_step], (h_state, c_state))
            
            bottleneck_out = h_state # The hidden state from the last timestep
            
            # Prepare skips from the LAST timestep only (common practice) 
            # or you can zero them out if mode == 'cnn_convlstm' inside the Up block
            skip1 = x1.view(b, t, -1, h, w)[:, -1]
            skip2 = x2.view(b, t, -1, h//2, w//2)[:, -1]
            skip3 = x3.view(b, t, -1, h//4, w//4)[:, -1]
            
        else:
            # --- Method 3: Standard U-Net (Frame Stacking) ---
            # Collapse Time into Channels: (B, T*C, H, W)
            # Note: You might need to adjust 'in_ch' in init to be in_ch * seq_len
            # But for fairness, we often process the *last* frame or stack them.
            # Let's assume we modify the first layer logic or stack inputs.
            # For this thesis code, let's keep it simple: Pass only the last frame 
            # OR refactor inputs. 
            
            # ALTERNATIVE: Just run the encoder on the last frame (t) 
            # but that loses history. Better: Stack frames.
            x_stacked = x.view(b, -1, h, w) 
            # Note: This requires changing the input conv channel count dynamically
            # or passing a 1x1 conv first.
            
            # *Assumption*: User changes 'in_ch' argument to 'in_ch * seq_len' when calling this mode.
            x1 = self.inc(x_stacked)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            
            bottleneck_out = self.bottleneck(x4)
            
            skip1, skip2, skip3 = x1, x2, x3

        # --- Decoder ---
        x = self.up1(bottleneck_out, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        logits = self.outc(x)
        return logits

# --- Helper Blocks (Put these in the same file) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + (out_channels if use_skips else 0), out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.use_skips:
            # Concatenate skip connection
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
        
# Note: You will need to copy your existing ConvLSTMCell implementation here 
# or import it from src.models.unet_convlstm
from src.models.unet_convlstm import ConvLSTMCell