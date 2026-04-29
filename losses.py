"""
Training objectives v2 — with anti-collapse losses

New losses to prevent mean pose collapse:
  L_div:    Diversity loss — penalizes low variance in batch predictions
  L_action: Action classification loss — forces encoder to distinguish actions  
  L_input:  Input-sensitivity loss — ensures different CSI → different predictions

L_total = L_pose(masked) + α·L_pose(clean) + β·L_cons 
        + γ·L_div + δ·L_action
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_decoder import H36M_BONES


class CoordinateLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        dist = torch.norm(pred - gt, dim=-1)
        return dist.mean()


class BoneConsistencyLoss(nn.Module):
    def __init__(self, bones=None):
        super().__init__()
        self.bones = bones or H36M_BONES

    def compute_bone_lengths(self, joints):
        bone_lengths = []
        for i, j in self.bones:
            length = torch.norm(joints[:, :, i] - joints[:, :, j], dim=-1)
            bone_lengths.append(length)
        return torch.stack(bone_lengths, dim=-1)

    def forward(self, pred, gt):
        return F.l1_loss(self.compute_bone_lengths(pred),
                         self.compute_bone_lengths(gt))


class VelocitySmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred_vel = pred[:, 1:] - pred[:, :-1]
        gt_vel = gt[:, 1:] - gt[:, :-1]
        return torch.norm(pred_vel - gt_vel, dim=-1).mean()


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_clean, pred_masked):
        return torch.norm(pred_clean.detach() - pred_masked, dim=-1).mean()


class DiversityLoss(nn.Module):
    """L_div: 惩罚 batch 内预测姿态方差过低.
    
    如果模型塌陷到均值姿态, batch 内所有预测几乎一样,
    方差趋近于零. 这个损失鼓励 batch 内不同样本的预测有差异.
    
    计算方式: -log(var + eps), 方差越小损失越大.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, pred):
        """
        Args:
            pred: (B, T, 17, 3) predicted poses
        Returns:
            scalar loss (lower variance → higher loss)
        """
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        # Mean pose per sample: (B, 17, 3) — average over time
        mean_pose = pred.mean(dim=1)  # (B, 17, 3)

        # Variance across batch for each joint coordinate
        var = mean_pose.var(dim=0).mean()  # scalar

        # Negative log variance: low var → high loss
        loss = -torch.log(var + self.eps)

        return loss


class TemporalDiversityLoss(nn.Module):
    """惩罚时序上预测变化过小 (模型输出每帧都一样).
    
    正常人体运动中, 相邻帧的姿态应该有变化.
    如果预测的时序变化量远小于GT, 说明模型没有捕捉动态.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred, gt: (B, T, 17, 3)
        Returns:
            scalar loss
        """
        # Temporal motion magnitude
        pred_motion = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1).mean(dim=(1, 2))  # (B,)
        gt_motion = torch.norm(gt[:, 1:] - gt[:, :-1], dim=-1).mean(dim=(1, 2))  # (B,)

        # Ratio: pred_motion should be similar to gt_motion
        # If pred is static (collapsed), pred_motion ≈ 0, loss is high
        ratio = pred_motion / (gt_motion + 1e-6)

        # Penalize when ratio < 1 (pred less dynamic than GT)
        loss = F.relu(1.0 - ratio).mean()

        return loss


class InputSensitivityLoss(nn.Module):
    """确保不同的CSI输入产生不同的预测.
    
    在 batch 内, 如果两个样本的GT姿态差异大, 
    但预测姿态差异小, 就施加惩罚.
    
    本质是一个soft contrastive loss on predictions.
    """

    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def forward(self, pred, gt):
        """
        Args:
            pred, gt: (B, T, 17, 3)
        Returns:
            scalar loss
        """
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        # Average pose per sample: (B, 17*3)
        pred_flat = pred.mean(dim=1).reshape(B, -1)
        gt_flat = gt.mean(dim=1).reshape(B, -1)

        # Pairwise distances in GT space and pred space
        loss = torch.tensor(0.0, device=pred.device)
        count = 0

        for i in range(B):
            for j in range(i + 1, B):
                gt_dist = torch.norm(gt_flat[i] - gt_flat[j])
                pred_dist = torch.norm(pred_flat[i] - pred_flat[j])

                # If GT are different (gt_dist > margin), pred should also differ
                if gt_dist > self.margin:
                    # Penalize when pred_dist is much smaller than gt_dist
                    ratio = pred_dist / (gt_dist + 1e-6)
                    loss = loss + F.relu(0.5 - ratio)  # want ratio > 0.5
                    count += 1

        if count > 0:
            loss = loss / count
        return loss




class PoseLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.5):
        super().__init__()
        self.coord_loss = CoordinateLoss()
        self.bone_loss = BoneConsistencyLoss()
        self.vel_loss = VelocitySmoothLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred, gt):
        l_coord = self.coord_loss(pred, gt)
        l_bone = self.bone_loss(pred, gt)
        l_vel = self.vel_loss(pred, gt)
        total = l_coord + self.lambda1 * l_bone + self.lambda2 * l_vel
        return total, {
            'l_coord': l_coord.item(),
            'l_bone': l_bone.item(),
            'l_vel': l_vel.item(),
        }


class TotalLoss(nn.Module):
    """L_total with anti-collapse losses.
    
    L = L_pose(masked) + α·L_pose(clean) + β·L_cons
      + γ·L_div + γ·L_temp_div + γ·L_input_sens
    """

    def __init__(self, lambda1=1.0, lambda2=0.5, alpha=0.5, beta=2.0,
                 gamma=0.1, delta=0.1):
        super().__init__()
        self.pose_loss = PoseLoss(lambda1, lambda2)
        self.cons_loss = ConsistencyLoss()
        self.div_loss = DiversityLoss()
        self.temp_div_loss = TemporalDiversityLoss()
        self.input_sens_loss = InputSensitivityLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # weight for anti-collapse losses
        self.delta = delta  # weight for action classification

    def forward(self, outputs, gt, training=True, action_loss=None):
        loss_dict = {}

        if training and 'p_final_masked' in outputs:
            l_pose_masked, md = self.pose_loss(outputs['p_final_masked'], gt)
            l_pose_clean, cd = self.pose_loss(outputs['p_final_clean'], gt)
            l_cons = self.cons_loss(outputs['p_final_clean'],
                                    outputs['p_final_masked'])

            # Anti-collapse losses (use clean path predictions)
            pred_clean = outputs['p_final_clean']
            l_div = self.div_loss(pred_clean)
            l_temp_div = self.temp_div_loss(pred_clean, gt)
            l_input_sens = self.input_sens_loss(pred_clean, gt)

            total = (l_pose_masked
                     + self.alpha * l_pose_clean
                     + self.beta * l_cons
                     + self.gamma * (l_div + l_temp_div + l_input_sens))

            if action_loss is not None:
                total = total + self.delta * action_loss

            loss_dict.update({
                'l_total': total.item(),
                'l_pose_masked': l_pose_masked.item(),
                'l_pose_clean': l_pose_clean.item(),
                'l_cons': l_cons.item(),
                'l_div': l_div.item(),
                'l_temp_div': l_temp_div.item(),
                'l_input_sens': l_input_sens.item(),
                'l_action': action_loss.item() if action_loss is not None else 0,
                'l_coord_masked': md['l_coord'],
                'l_coord_clean': cd['l_coord'],
                'l_bone_masked': md['l_bone'],
                'l_vel_masked': md['l_vel'],
            })
        else:
            pred = outputs.get('p_final', outputs.get('p_final_clean'))
            l_pose, details = self.pose_loss(pred, gt)
            total = l_pose
            loss_dict.update({
                'l_total': total.item(),
                'l_pose': l_pose.item(),
                **details,
            })

        return total, loss_dict