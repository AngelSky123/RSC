"""
CSI-RSC-PoseDG v5 — with action classifier to prevent mean pose collapse.

Key additions:
  - ActionClassifier head on z_global
  - Gradient monitoring (input sensitivity check)
"""
import torch
import torch.nn as nn

from .csi_encoder import DualBranchCSIEncoder
from .local_encoder import LocalSpatioTemporalEncoder, LocalFeaturePooling
from .global_encoder import GlobalTemporalModeler
from .pose_decoder import PoseDecoder
from .pose_decoder import ActionClassifier


class CSIRSCPoseDG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._debug_printed = False
        self.rsc_drop_pct = args.rsc2_time_drop_pct
        self.rsc_batch_pct = args.rsc2_batch_pct

        self.csi_encoder = DualBranchCSIEncoder(
            amp_channels=args.amp_channels,
            phase_channels=args.phase_channels,
            hidden_dim=args.encoder_hidden_dim,
            out_dim=args.encoder_out_dim,
        )
        self.local_encoder = LocalSpatioTemporalEncoder(
            in_channels=args.encoder_out_dim,
            hidden_dim=args.local_hidden_dim,
            out_dim=args.local_out_dim,
            num_blocks=args.num_res3d_blocks,
        )
        self.feature_pooling = LocalFeaturePooling(
            in_channels=args.local_out_dim,
            out_channels=args.global_dim,
        )
        self.global_modeler = GlobalTemporalModeler(
            in_dim=args.global_dim,
            global_dim=args.global_dim,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads,
            tcn_channels=args.tcn_channels,
            tcn_kernel_size=args.tcn_kernel_size,
            dropout=args.transformer_dropout,
            max_seq_len=args.seq_len + 50,
        )
        self.pose_decoder = PoseDecoder(
            in_dim=args.global_dim,
            hidden_dim=args.coarse_hidden_dim,
            gcn_hidden=args.gcn_hidden_dim,
            num_gcn_layers=args.num_gcn_layers,
            num_joints=args.num_joints,
        )
        # Action classifier to prevent feature collapse
        self.action_classifier = ActionClassifier(
            in_dim=args.global_dim,
            num_actions=args.num_actions,
        )

    def forward_backbone(self, csi):
        feat = self.csi_encoder(csi)
        z_local = self.local_encoder(feat)
        z_pooled = self.feature_pooling(z_local)
        z_global = self.global_modeler(z_pooled)
        return z_local, z_global

    def forward_decoder(self, z_global):
        return self.pose_decoder(z_global)

    def forward(self, csi):
        z_local, z_global = self.forward_backbone(csi)
        p_coarse, p_final = self.forward_decoder(z_global)
        action_logits = self.action_classifier(z_global)
        return {
            'p_coarse': p_coarse,
            'p_final': p_final,
            'z_local': z_local,
            'z_global': z_global,
            'action_logits': action_logits,
        }

    def _apply_rsc_mask(self, z, gradient):
        B, T, C = z.shape
        num_apply = max(1, int(B * self.rsc_batch_pct))
        perm = torch.randperm(B, device=z.device)
        apply_indices = perm[:num_apply]
        z_masked = z.clone()
        for idx in apply_indices:
            g = gradient[idx].abs()
            g_flat = g.reshape(-1)
            num_to_drop = max(1, int(self.rsc_drop_pct * g_flat.numel()))
            num_to_keep = max(1, g_flat.numel() - num_to_drop)
            threshold, _ = g_flat.kthvalue(num_to_keep)
            mask = (g < threshold).float()
            z_masked[idx] = z[idx] * mask
        return z_masked

    def forward_rsc(self, csi, pose_3d, loss_fn):
        # Step 1: Backbone
        z_local, z_global_raw = self.forward_backbone(csi)
        z_global = z_global_raw.detach().clone().requires_grad_(True)

        # Step 2: Clean decode
        p_coarse_clean, p_final_clean = self.forward_decoder(z_global)

        # Action classification (from raw z_global with gradients to backbone)
        action_logits = self.action_classifier(z_global_raw)

        # Step 3: Gradient for RSC
        loss_for_grad = loss_fn(p_final_clean, pose_3d)
        grad_global = torch.autograd.grad(
            loss_for_grad, z_global,
            create_graph=False, retain_graph=True,
        )[0]

        # Step 4: RSC mask
        with torch.no_grad():
            z_global_masked = self._apply_rsc_mask(
                z_global.detach(), grad_global.detach())
        z_global_masked = z_global_masked.requires_grad_(True)

        # Debug print once
        if not self._debug_printed:
            with torch.no_grad():
                diff = (z_global.detach() - z_global_masked.detach()).abs()
                pct = 100.0 * (diff > 1e-8).float().sum().item() / diff.numel()
            print(f"[RSC DEBUG] z_global: {z_global.shape}, "
                  f"masked {pct:.1f}%, "
                  f"grad_norm={grad_global.abs().mean():.6f}")
            self._debug_printed = True

        # Step 5: Masked decode
        p_coarse_masked, p_final_masked = self.forward_decoder(z_global_masked)

        return {
            'p_coarse_clean': p_coarse_clean,
            'p_final_clean': p_final_clean,
            'z_local': z_local,
            'z_global': z_global,
            'z_global_raw': z_global_raw,  # with backbone gradients
            'p_coarse_masked': p_coarse_masked,
            'p_final_masked': p_final_masked,
            'z_global_masked': z_global_masked,
            'action_logits': action_logits,
        }