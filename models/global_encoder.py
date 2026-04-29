"""
Module 6: 全局时序建模器 v2
新增: MixStyle on temporal features
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixstyle import MixStyleTemporal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim, patch_size=1):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        return self.norm(x)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.conv(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class GlobalTemporalModeler(nn.Module):
    """Input: (B, T, C_in) -> Output: Z_global (B, T, C_g)
    
    With MixStyle after patch embedding to mix temporal domain styles.
    """

    def __init__(self, in_dim, global_dim=128, num_transformer_layers=3,
                 num_heads=4, tcn_channels=None, tcn_kernel_size=3,
                 dropout=0.1, max_seq_len=500):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [global_dim, global_dim]
        self.patch_embed = PatchEmbedding(in_dim, global_dim)
        self.pos_encoding = PositionalEncoding(global_dim, max_seq_len, dropout)

        # MixStyle after embedding, before transformer
        self.mixstyle = MixStyleTemporal(p=0.5, alpha=0.3)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(global_dim, num_heads, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])
        self.tcn_blocks = nn.ModuleList()
        for i, ch in enumerate(tcn_channels):
            dilation = 2 ** i
            self.tcn_blocks.append(TemporalConvBlock(ch, tcn_kernel_size, dilation, dropout))
        self.final_norm = nn.LayerNorm(global_dim)
        self.global_dim = global_dim

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_encoding(x)

        # Apply MixStyle to mix temporal statistics across samples
        x = self.mixstyle(x)

        for block in self.transformer_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        for tcn in self.tcn_blocks:
            x = tcn(x)
        x = x.permute(0, 2, 1)
        z_global = self.final_norm(x)
        return z_global