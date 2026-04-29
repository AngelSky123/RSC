"""
CSI-specific data augmentation for cross-environment robustness.

WiFi CSI跨环境的核心差异:
  - 多径信道不同 → 幅度分布偏移
  - 设备位置/朝向变化 → 相位偏移
  - 环境反射物不同 → 频率选择性衰落模式不同
  - 人距离设备远近不同 → 整体信号强度变化

这些增强模拟上述变化, 迫使模型学习对环境不变的人体动态特征.
"""
import torch
import torch.nn as nn
import numpy as np


class CSIAugmentor(nn.Module):
    """CSI数据增强器: 在训练时对CSI张量施加多种环境模拟扰动.
    
    Input: (T, 9, 114, 10) — [3 amp + 6 phase]
    Output: (T, 9, 114, 10) — augmented
    """

    def __init__(self,
                 amp_scale_range=(0.7, 1.3),
                 phase_noise_std=0.15,
                 subcarrier_drop_pct=0.1,
                 antenna_drop_prob=0.1,
                 time_warp_prob=0.3,
                 freq_mask_prob=0.3,
                 freq_mask_width=10,
                 channel_shuffle_prob=0.2,
                 p=0.8):
        """
        Args:
            amp_scale_range: 幅度随机缩放范围 (模拟信号强度变化)
            phase_noise_std: 相位高斯噪声标准差 (模拟相位偏移)
            subcarrier_drop_pct: 随机丢弃子载波比例 (模拟频率选择性衰落)
            antenna_drop_prob: 随机丢弃整条天线概率 (模拟天线故障/遮挡)
            time_warp_prob: 时序微扰概率
            freq_mask_prob: 频域mask概率 (SpecAugment风格)
            freq_mask_width: 频域mask最大宽度
            channel_shuffle_prob: 天线通道随机置换概率
            p: 整体启用增强的概率
        """
        super().__init__()
        self.amp_scale_range = amp_scale_range
        self.phase_noise_std = phase_noise_std
        self.subcarrier_drop_pct = subcarrier_drop_pct
        self.antenna_drop_prob = antenna_drop_prob
        self.time_warp_prob = time_warp_prob
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_width = freq_mask_width
        self.channel_shuffle_prob = channel_shuffle_prob
        self.p = p

    def forward(self, csi):
        """
        Args:
            csi: (T, 9, 114, 10) — [amp(3) + sin_phase(3) + cos_phase(3)]
        Returns:
            augmented csi: same shape
        """
        if np.random.rand() > self.p:
            return csi

        csi = csi.clone()
        T, C, H, W = csi.shape  # T, 9, 114, 10

        amp = csi[:, :3]       # (T, 3, 114, 10)
        phase = csi[:, 3:]     # (T, 6, 114, 10)

        # === 1. Amplitude random scaling (模拟不同环境信号强度) ===
        if np.random.rand() < 0.7:
            # Per-antenna scaling
            scale = torch.empty(1, 3, 1, 1).uniform_(
                self.amp_scale_range[0], self.amp_scale_range[1]
            )
            amp = amp * scale
            amp = amp.clamp(0, 1)

        # === 2. Phase noise injection (模拟环境相位偏移) ===
        if np.random.rand() < 0.7:
            noise = torch.randn_like(phase) * self.phase_noise_std
            phase = phase + noise

        # === 3. Subcarrier dropout (模拟频率选择性衰落) ===
        if np.random.rand() < 0.5:
            num_drop = max(1, int(self.subcarrier_drop_pct * H))
            drop_indices = np.random.choice(H, num_drop, replace=False)
            amp[:, :, drop_indices, :] = 0
            phase[:, :, drop_indices, :] = 0

        # === 4. Frequency band masking (SpecAugment风格, 模拟窄带干扰) ===
        if np.random.rand() < self.freq_mask_prob:
            mask_width = np.random.randint(1, self.freq_mask_width + 1)
            start = np.random.randint(0, max(1, H - mask_width))
            amp[:, :, start:start + mask_width, :] = 0
            phase[:, :, start:start + mask_width, :] = 0

        # === 5. Antenna channel dropout (模拟天线遮挡) ===
        if np.random.rand() < self.antenna_drop_prob:
            drop_ant = np.random.randint(0, 3)
            amp[:, drop_ant] = 0
            phase[:, drop_ant * 2] = 0
            phase[:, drop_ant * 2 + 1] = 0

        # === 6. Antenna channel shuffle (模拟天线排列差异) ===
        if np.random.rand() < self.channel_shuffle_prob:
            perm = torch.randperm(3)
            amp = amp[:, perm]
            # Shuffle corresponding sin/cos pairs
            phase_pairs = phase.reshape(T, 3, 2, H, W)
            phase_pairs = phase_pairs[:, perm]
            phase = phase_pairs.reshape(T, 6, H, W)

        # === 7. Temporal jitter (微小时序偏移, 模拟采样不同步) ===
        if np.random.rand() < self.time_warp_prob and T > 4:
            # Random shift 1-2 frames
            shift = np.random.randint(1, 3)
            if np.random.rand() < 0.5:
                amp = torch.cat([amp[shift:], amp[-shift:].clone()], dim=0)
                phase = torch.cat([phase[shift:], phase[-shift:].clone()], dim=0)
            else:
                amp = torch.cat([amp[:shift].clone(), amp[:-shift]], dim=0)
                phase = torch.cat([phase[:shift].clone(), phase[:-shift]], dim=0)

        # === 8. Gaussian noise on amplitude (模拟测量噪声) ===
        if np.random.rand() < 0.5:
            amp_noise = torch.randn_like(amp) * 0.02
            amp = (amp + amp_noise).clamp(0, 1)

        csi[:, :3] = amp
        csi[:, 3:] = phase
        return csi


class EnvironmentSimulator:
    """更强的环境模拟: 模拟完全不同的多径信道.
    
    通过对子载波维度施加随机频率选择性衰落来模拟不同房间的信道响应.
    """

    @staticmethod
    def random_multipath_fading(csi, num_paths=3):
        """Simulate multipath fading by applying random frequency-selective gain.
        
        Args:
            csi: (T, 9, 114, 10)
            num_paths: number of simulated multipath components
        Returns:
            faded csi: same shape
        """
        T, C, H, W = csi.shape
        
        # Generate random frequency-selective fading profile
        fading = torch.ones(1, 1, H, 1)
        for _ in range(num_paths):
            freq = np.random.uniform(0.5, 5.0)
            phase_offset = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.1, 0.5)
            
            subcarrier_idx = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1)
            component = 1.0 + amplitude * torch.sin(
                2 * np.pi * freq * subcarrier_idx / H + phase_offset
            )
            fading = fading * component

        # Normalize fading to keep overall power similar
        fading = fading / fading.mean()

        # Apply only to amplitude channels
        csi = csi.clone()
        csi[:, :3] = csi[:, :3] * fading
        csi[:, :3] = csi[:, :3].clamp(0, 1)
        
        return csi