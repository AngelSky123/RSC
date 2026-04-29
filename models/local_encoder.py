"""
Module 4: 局部时频编码器 - Res 3D Conv Block (Temporal-Spectral-Channel)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Res3DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=3, spatial_kernel=3):
        super().__init__()
        t_pad = temporal_kernel // 2
        s_pad = spatial_kernel // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                               padding=(t_pad, s_pad, s_pad))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                               padding=(t_pad, s_pad, s_pad))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class LocalSpatioTemporalEncoder(nn.Module):
    """Input: (B, T, C_f, 114, 10) -> Output: Z_local (B, T, C_z, 114, 10)"""

    def __init__(self, in_channels=128, hidden_dim=128, out_dim=128,
                 num_blocks=3, temporal_kernel=3):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i in range(num_blocks):
            ch_out = hidden_dim if i < num_blocks - 1 else out_dim
            layers.append(Res3DConvBlock(ch_in, ch_out, temporal_kernel=temporal_kernel))
            ch_in = ch_out
        self.encoder = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C_f, T, 114, 10)
        z = self.encoder(x)             # (B, C_z, T, 114, 10)
        z = z.permute(0, 2, 1, 3, 4)   # (B, T, C_z, 114, 10)
        return z


class LocalFeaturePooling(nn.Module):
    """Z_local (B, T, C_z, 114, 10) -> (B, T, C_g)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, z_local):
        B, T, C, H, W = z_local.shape
        z = z_local.reshape(B * T, C, H, W)
        z = self.pool(z).squeeze(-1).squeeze(-1)  # (B*T, C_z)
        z = self.proj(z)
        return z.reshape(B, T, -1)