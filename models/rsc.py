"""
Modules 5 & 7: Representation Self-Challenging (RSC)
RSC-I (局部自挑战): spectro-temporal mask + channel mask on Z_local
RSC-II (全局自挑战): time-wise mask + channel-wise mask on Z_global

训练阶段启用RSC; 测试阶段关闭masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSCModule(nn.Module):
    """通用RSC模块: 基于梯度的Top-k特征Masking."""

    def __init__(self, drop_pct=0.333, batch_pct=0.333):
        super().__init__()
        self.drop_pct = drop_pct
        self.batch_pct = batch_pct

    def compute_mask_from_gradient(self, gradient, drop_pct):
        with torch.no_grad():
            flat_grad = gradient.reshape(gradient.shape[0], -1)
            k = max(1, int((1.0 - drop_pct) * flat_grad.shape[1]))
            threshold, _ = flat_grad.kthvalue(k, dim=1, keepdim=True)
            for _ in range(len(gradient.shape) - 2):
                threshold = threshold.unsqueeze(-1)
            mask = (gradient < threshold).float()
        return mask

    def forward(self, z, gradient=None):
        if not self.training or gradient is None:
            return z
        B = z.shape[0]
        device = z.device
        num_apply = max(1, int(B * self.batch_pct))
        apply_mask = torch.zeros(B, dtype=torch.bool, device=device)
        indices = torch.randperm(B, device=device)[:num_apply]
        apply_mask[indices] = True
        mask = self.compute_mask_from_gradient(gradient, self.drop_pct)
        expand_mask = apply_mask.view(B, *([1] * (len(z.shape) - 1)))
        effective_mask = torch.where(expand_mask, mask, torch.ones_like(mask))
        z_masked = z * effective_mask
        return z_masked


class RSCLocalChallenger(nn.Module):
    """RSC-I: 局部自挑战 on Z_local (B, T, C_z, 114, 10)
    1. Spectro-Temporal Mask: mask on (114, 10) spatial dims
    2. Channel Mask: mask on C_z channel dim
    """

    def __init__(self, spatial_drop_pct=0.333, channel_drop_pct=0.333, batch_pct=0.333):
        super().__init__()
        self.spatial_rsc = RSCModule(spatial_drop_pct, batch_pct)
        self.channel_rsc = RSCModule(channel_drop_pct, batch_pct)

    def compute_spectro_temporal_mask(self, z_local, gradient):
        spatial_grad = gradient.abs().mean(dim=2)  # (B, T, 114, 10)
        BT = spatial_grad.shape[0] * spatial_grad.shape[1]
        spatial_grad_flat = spatial_grad.reshape(BT, -1)
        k = max(1, int((1.0 - self.spatial_rsc.drop_pct) * spatial_grad_flat.shape[1]))
        threshold, _ = spatial_grad_flat.kthvalue(k, dim=1, keepdim=True)
        threshold = threshold.reshape(spatial_grad.shape[0], spatial_grad.shape[1], 1, 1)
        mask = (spatial_grad < threshold).float().unsqueeze(2)
        return mask

    def compute_channel_mask(self, z_local, gradient):
        channel_grad = gradient.abs().mean(dim=(-2, -1))  # (B, T, C_z)
        BT = channel_grad.shape[0] * channel_grad.shape[1]
        channel_grad_flat = channel_grad.reshape(BT, -1)
        k = max(1, int((1.0 - self.channel_rsc.drop_pct) * channel_grad_flat.shape[1]))
        threshold, _ = channel_grad_flat.kthvalue(k, dim=1, keepdim=True)
        threshold = threshold.reshape(channel_grad.shape[0], channel_grad.shape[1], 1)
        mask = (channel_grad < threshold).float().unsqueeze(-1).unsqueeze(-1)
        return mask

    def forward(self, z_local, gradient=None):
        if not self.training or gradient is None:
            return z_local
        B = z_local.shape[0]
        device = z_local.device
        num_apply = max(1, int(B * self.spatial_rsc.batch_pct))
        apply_mask = torch.zeros(B, dtype=torch.bool, device=device)
        indices = torch.randperm(B, device=device)[:num_apply]
        apply_mask[indices] = True
        spatial_mask = self.compute_spectro_temporal_mask(z_local, gradient)
        channel_mask = self.compute_channel_mask(z_local, gradient)
        combined_mask = spatial_mask * channel_mask
        batch_selector = apply_mask.view(B, 1, 1, 1, 1).float()
        effective_mask = batch_selector * combined_mask + (1 - batch_selector)
        return z_local * effective_mask


class RSCGlobalChallenger(nn.Module):
    """RSC-II: 全局自挑战 on Z_global (B, T, C_g)
    1. Time-wise Mask: mask on T time dim
    2. Channel-wise Mask: mask on C_g channel dim
    """

    def __init__(self, time_drop_pct=0.333, channel_drop_pct=0.333, batch_pct=0.333):
        super().__init__()
        self.time_drop_pct = time_drop_pct
        self.channel_drop_pct = channel_drop_pct
        self.batch_pct = batch_pct

    def compute_time_mask(self, gradient):
        time_grad = gradient.abs().mean(dim=-1)
        k = max(1, int((1.0 - self.time_drop_pct) * time_grad.shape[1]))
        threshold, _ = time_grad.kthvalue(k, dim=1, keepdim=True)
        mask = (time_grad < threshold).float().unsqueeze(-1)
        return mask

    def compute_channel_mask(self, gradient):
        channel_grad = gradient.abs().mean(dim=1)
        k = max(1, int((1.0 - self.channel_drop_pct) * channel_grad.shape[1]))
        threshold, _ = channel_grad.kthvalue(k, dim=1, keepdim=True)
        mask = (channel_grad < threshold).float().unsqueeze(1)
        return mask

    def forward(self, z_global, gradient=None):
        if not self.training or gradient is None:
            return z_global
        B = z_global.shape[0]
        device = z_global.device
        num_apply = max(1, int(B * self.batch_pct))
        apply_mask = torch.zeros(B, dtype=torch.bool, device=device)
        indices = torch.randperm(B, device=device)[:num_apply]
        apply_mask[indices] = True
        time_mask = self.compute_time_mask(gradient)
        channel_mask = self.compute_channel_mask(gradient)
        combined_mask = time_mask * channel_mask
        batch_selector = apply_mask.view(B, 1, 1).float()
        effective_mask = batch_selector * combined_mask + (1 - batch_selector)
        return z_global * effective_mask