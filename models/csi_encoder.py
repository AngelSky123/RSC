"""
Module 3: 双分支 CSI 编码器 v2
新增:
  - InstanceNorm 替代/辅助 BatchNorm (去除环境统计偏移)
  - MixStyle 插入在残差块之间 (混合域风格)
  - 环境归一化层 (per-sample whitening)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixstyle import MixStyle2D


class EnvironmentNormalization(nn.Module):
    """环境归一化: Per-sample instance normalization on CSI features.
    
    不同环境的CSI幅度分布差异很大, 这一层对每个样本独立归一化,
    去除环境相关的全局统计量, 只保留相对变化模式.
    """

    def __init__(self, num_channels, momentum=0.1):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=True)

    def forward(self, x):
        """x: (B, C, H, W)"""
        return self.instance_norm(x)


class ResBlock2D(nn.Module):
    """2D残差块 with dual normalization: BatchNorm + InstanceNorm.
    
    BatchNorm captures shared statistics across samples.
    InstanceNorm removes per-sample environment-specific statistics.
    Learnable gate combines both.
    """

    def __init__(self, in_channels, out_channels, stride=1, use_instnorm=True):
        super().__init__()
        self.use_instnorm = use_instnorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_instnorm:
            self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
            # Learnable gate: 0=all BN, 1=all IN
            self.gate1 = nn.Parameter(torch.tensor(0.3))
            self.gate2 = nn.Parameter(torch.tensor(0.3))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def _norm(self, x, bn, instn, gate):
        """Blend BN and IN using learnable gate."""
        if self.use_instnorm:
            g = torch.sigmoid(gate)
            return (1 - g) * bn(x) + g * instn(x)
        return bn(x)

    def forward(self, x):
        out = self.conv1(x)
        out = F.gelu(self._norm(out, self.bn1,
                                self.in1 if self.use_instnorm else None,
                                self.gate1 if self.use_instnorm else None))
        out = self.conv2(out)
        out = self._norm(out, self.bn2,
                         self.in2 if self.use_instnorm else None,
                         self.gate2 if self.use_instnorm else None)
        out = out + self.shortcut(x)
        return F.gelu(out)


class AmplitudeBranch(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, out_dim=64):
        super().__init__()
        self.env_norm = EnvironmentNormalization(in_channels)
        self.block1 = ResBlock2D(in_channels, hidden_dim, use_instnorm=True)
        self.mixstyle1 = MixStyle2D(p=0.5, alpha=0.3)
        self.block2 = ResBlock2D(hidden_dim, out_dim, use_instnorm=True)

    def forward(self, x):
        x = self.env_norm(x)
        x = self.block1(x)
        x = self.mixstyle1(x)  # Mix styles after first block
        x = self.block2(x)
        return x


class PhaseAwareBranch(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=32, out_dim=64):
        super().__init__()
        self.env_norm = EnvironmentNormalization(in_channels)
        self.block1 = ResBlock2D(in_channels, hidden_dim, use_instnorm=True)
        self.mixstyle1 = MixStyle2D(p=0.5, alpha=0.3)
        self.block2 = ResBlock2D(hidden_dim, out_dim, use_instnorm=True)

    def forward(self, x):
        x = self.env_norm(x)
        x = self.block1(x)
        x = self.mixstyle1(x)
        x = self.block2(x)
        return x


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_amp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate_phase = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.cross_attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, feat_amp, feat_phase):
        concat = torch.cat([feat_amp, feat_phase], dim=1)
        g_amp = self.gate_amp(concat)
        g_phase = self.gate_phase(concat)
        gated = g_amp * feat_amp + g_phase * feat_phase
        cross = self.cross_attn(concat)
        return gated + cross


class DualBranchCSIEncoder(nn.Module):
    """Input: (B, T, 9, 114, 10) -> Output: (B, T, C_f, 114, 10)
    
    With EnvironmentNorm + MixStyle for domain generalization.
    """

    def __init__(self, amp_channels=3, phase_channels=6,
                 hidden_dim=32, out_dim=64, chunk_size=16):
        super().__init__()
        self.amp_channels = amp_channels
        self.phase_channels = phase_channels
        self.chunk_size = chunk_size
        self.amp_branch = AmplitudeBranch(amp_channels, hidden_dim, out_dim)
        self.phase_branch = PhaseAwareBranch(phase_channels, hidden_dim, out_dim)
        self.fusion = GatedFusion(out_dim)

    def _process_chunk(self, amp_chunk, phase_chunk):
        feat_amp = self.amp_branch(amp_chunk)
        feat_phase = self.phase_branch(phase_chunk)
        return self.fusion(feat_amp, feat_phase)

    def forward(self, x):
        B, T, C, H, W = x.shape
        amp_input = x[:, :, :self.amp_channels]
        phase_input = x[:, :, self.amp_channels:]

        outputs = []
        for t_start in range(0, T, self.chunk_size):
            t_end = min(t_start + self.chunk_size, T)
            chunk_len = t_end - t_start

            amp_chunk = amp_input[:, t_start:t_end].reshape(
                B * chunk_len, self.amp_channels, H, W)
            phase_chunk = phase_input[:, t_start:t_end].reshape(
                B * chunk_len, self.phase_channels, H, W)

            fused_chunk = self._process_chunk(amp_chunk, phase_chunk)
            C_f = fused_chunk.shape[1]
            fused_chunk = fused_chunk.reshape(B, chunk_len, C_f, H, W)
            outputs.append(fused_chunk)

        return torch.cat(outputs, dim=1)