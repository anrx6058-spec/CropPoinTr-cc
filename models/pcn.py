#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCN - Point Completion Network (CVPR 2019)
最经典的点云补全baseline

结构非常简单：
- Encoder: PointNet (全局max pooling → 特征向量)
- Decoder: 全连接层生成点云

论文: https://arxiv.org/abs/1808.00671
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCN(nn.Module):
    def __init__(self, num_points=2048, feat_dim=1024, decoder_dim=1024):
        """
        Args:
            num_points: 输出补全点云的点数
            feat_dim: 编码器输出特征维度
            decoder_dim: 解码器中间维度
        """
        super(PCN, self).__init__()
        self.num_points = num_points
        self.feat_dim = feat_dim

        # === Encoder (PointNet风格) ===
        # 输入: B x N x 3
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(feat_dim)

        # === Decoder ===
        # 3个全连接层，逐步上采样到目标点数
        self.fc1 = nn.Linear(feat_dim, decoder_dim)
        self.fc2 = nn.Linear(decoder_dim, decoder_dim // 2)
        self.fc3 = nn.Linear(decoder_dim // 2, num_points * 3)

        self.bn_fc1 = nn.BatchNorm1d(decoder_dim)
        self.bn_fc2 = nn.BatchNorm1d(decoder_dim // 2)

    def encoder(self, x):
        """
        输入: x (B, N, 3)
        输出: global_feat (B, feat_dim)
        """
        # B x N x 3 -> B x 3 x N
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))   # B x 128 x N
        x = F.relu(self.bn2(self.conv2(x)))   # B x 256 x N
        x = F.relu(self.bn3(self.conv3(x)))   # B x feat_dim x N

        # 全局max pooling
        global_feat = torch.max(x, 2, keepdim=False)[0]  # B x feat_dim
        return global_feat

    def decoder(self, global_feat):
        """
        输入: global_feat (B, feat_dim)
        输出: points (B, num_points, 3)
        """
        x = F.relu(self.bn_fc1(self.fc1(global_feat)))  # B x decoder_dim
        x = F.relu(self.bn_fc2(self.fc2(x)))            # B x decoder_dim/2
        x = self.fc3(x)                                 # B x (num_points * 3)

        # reshape 成点云格式
        points = x.view(-1, self.num_points, 3)
        return points

    def forward(self, x):
        """
        输入: x (B, N, 3)  - 不完整的输入点云
        输出: points (B, num_points, 3)  - 补全后的点云
        """
        global_feat = self.encoder(x)
        points = self.decoder(global_feat)
        return points


if __name__ == '__main__':
    # 测试
    batch_size = 4
    input_points = 2048
    output_points = 1024

    model = PCN(num_points=output_points)
    x = torch.randn(batch_size, input_points, 3)

    out = model(x)
    print(f"输入:  {x.shape}")
    print(f"输出:  {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 验证输出是有效的点云坐标
    print(f"输出范围: [{out.min():.3f}, {out.max():.3f}]")
