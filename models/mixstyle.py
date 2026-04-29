"""
MixStyle: 特征层面的域风格混合

Paper: "Domain Generalization with MixStyle" (Zhou et al., ICLR 2021)

核心思想: 在CNN中间层混合不同样本的特征统计量(均值和方差),
生成"虚拟域"的特征, 迫使模型学习对统计量变化鲁棒的表示.

对WiFi CSI特别有效, 因为不同环境的CSI主要差异就是统计特性
(幅度分布、方差模式), MixStyle直接在特征层面打乱这些统计量.
"""
import torch
import torch.nn as nn
import random


class MixStyle(nn.Module):
    """MixStyle layer: mix feature statistics across samples.
    
    在训练时:
      1. 计算每个样本的特征均值和标准差 (instance statistics)
      2. 随机选择batch中另一个样本
      3. 用随机权重混合两者的统计量
      4. 用混合后的统计量重新归一化当前样本的特征
    
    在测试时: 直接通过, 不做任何操作.
    """

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
            p: 启用MixStyle的概率
            alpha: Beta分布参数, 控制混合强度 (越小越激进)
            eps: 数值稳定
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (B, C, ...) — any tensor with batch and channel dims
        Returns:
            mixed x: same shape
        """
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        if B < 2:
            return x

        # Compute instance statistics
        # Flatten all dims except B and C
        feat_shape = x.shape
        x_flat = x.reshape(B, feat_shape[1], -1)  # (B, C, N)

        mu = x_flat.mean(dim=-1, keepdim=True)     # (B, C, 1)
        sig = (x_flat.var(dim=-1, keepdim=True) + self.eps).sqrt()  # (B, C, 1)

        # Normalize
        x_normed = (x_flat - mu) / sig  # (B, C, N)

        # Random mixing weight from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1)
        ).to(x.device)

        # Shuffle indices for mixing partner
        perm = torch.randperm(B, device=x.device)

        # Mix statistics
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]

        # Apply mixed statistics
        x_mixed = x_normed * sig_mix + mu_mix  # (B, C, N)

        return x_mixed.reshape(feat_shape)


class MixStyle2D(nn.Module):
    """MixStyle for 2D feature maps (B, C, H, W).
    
    Statistics computed per-instance per-channel over (H, W).
    """

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        """x: (B, C, H, W)"""
        if not self.training or random.random() > self.p:
            return x

        B, C, H, W = x.shape
        if B < 2:
            return x

        mu = x.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()

        x_normed = (x - mu) / sig

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1, 1)
        ).to(x.device)

        perm = torch.randperm(B, device=x.device)

        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]

        return x_normed * sig_mix + mu_mix


class MixStyleTemporal(nn.Module):
    """MixStyle for temporal features (B, T, C).
    
    Statistics computed per-instance per-channel over T.
    Particularly useful for WiFi CSI temporal features where
    environment affects the temporal statistics.
    """

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        """x: (B, T, C)"""
        if not self.training or random.random() > self.p:
            return x

        B, T, C = x.shape
        if B < 2:
            return x

        mu = x.mean(dim=1, keepdim=True)   # (B, 1, C)
        sig = (x.var(dim=1, keepdim=True) + self.eps).sqrt()

        x_normed = (x - mu) / sig

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1)
        ).to(x.device)

        perm = torch.randperm(B, device=x.device)

        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]

        return x_normed * sig_mix + mu_mix