"""
Module 8: 3D 姿态解码头
- Coarse Pose Head (MLP): Z_global -> P_coarse (T x 17 x 3)
- Skeleton Refiner (Graph Convolution): P_coarse -> P_final (T x 17 x 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# H36M 17 joints skeleton
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),        # Right leg
    (0, 4), (4, 5), (5, 6),        # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine -> Head
    (8, 11), (11, 12), (12, 13),   # Left arm
    (8, 14), (14, 15), (15, 16),   # Right arm
]

NUM_JOINTS = 17
NUM_BONES = len(H36M_BONES)


def build_adjacency_matrix(num_joints=17, bones=None, self_loop=True):
    if bones is None:
        bones = H36M_BONES
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in bones:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        A = A + np.eye(num_joints, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


class CoarsePoseHead(nn.Module):
    """Input: Z_global (B, T, C_g) -> Output: P_coarse (B, T, 17, 3)"""

    def __init__(self, in_dim=256, hidden_dim=512, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, z_global):
        B, T, _ = z_global.shape
        out = self.mlp(z_global)
        return out.reshape(B, T, self.num_joints, 3)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj_matrix).float())
        self.W = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        support = self.W(x)
        out = torch.matmul(self.adj, support)
        BT, J, C = out.shape
        out = self.bn(out.reshape(BT * J, C)).reshape(BT, J, C)
        return out


class SkeletonRefiner(nn.Module):
    """Input: P_coarse (B, T, 17, 3) -> Output: P_final (B, T, 17, 3)"""

    def __init__(self, in_features=3, hidden_dim=128, num_layers=3, num_joints=17):
        super().__init__()
        adj = build_adjacency_matrix(num_joints)
        self.input_proj = GraphConvLayer(in_features, hidden_dim, adj)
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim, adj))
        self.output_proj = nn.Linear(hidden_dim, 3)
        self.num_joints = num_joints

    def forward(self, p_coarse):
        B, T, J, _ = p_coarse.shape
        x = p_coarse.reshape(B * T, J, 3)
        x = F.gelu(self.input_proj(x))
        for gcn in self.gcn_layers:
            residual = x
            x = F.gelu(gcn(x)) + residual
        delta = self.output_proj(x)
        delta = delta.reshape(B, T, J, 3)
        p_final = p_coarse + delta
        return p_final


class PoseDecoder(nn.Module):
    """Input: Z_global (B, T, C_g) -> Output: P_coarse, P_final (B, T, 17, 3)"""

    def __init__(self, in_dim=256, hidden_dim=512, gcn_hidden=128,
                 num_gcn_layers=3, num_joints=17):
        super().__init__()
        self.coarse_head = CoarsePoseHead(in_dim, hidden_dim, num_joints)
        self.refiner = SkeletonRefiner(
            in_features=3, hidden_dim=gcn_hidden,
            num_layers=num_gcn_layers, num_joints=num_joints
        )

    def forward(self, z_global):
        p_coarse = self.coarse_head(z_global)
        p_final = self.refiner(p_coarse)
        return p_coarse, p_final


class ActionClassifier(nn.Module):
    """动作分类辅助头: 从全局特征预测动作类别."""

    def __init__(self, in_dim=128, num_actions=27):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, num_actions),
        )

    def forward(self, z_global):
        z_pooled = z_global.mean(dim=1)
        return self.classifier(z_pooled)