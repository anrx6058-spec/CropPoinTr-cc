#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RP-PCN: 从 CP-PCN 适配的模型
修复了原始代码中的 CUDA 硬编码问题，支持 CPU 运行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """K近邻搜索"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """构建图特征 (DGCNN)"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)

    # 修复：使用输入tensor的设备，而不是硬编码cuda
    device = x.device
    idx = idx.to(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.k = 20
        self.conv1 = torch.nn.Conv2d(2 * 3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64 * 2, 64, 1)
        self.conv3 = torch.nn.Conv2d(64 * 2, 128, 1)
        self.conv4 = torch.nn.Conv2d(128 * 2, 256, 1)
        self.conv5 = torch.nn.Conv2d(256 * 2, 512, 1)
        self.conv6 = torch.nn.Conv2d(512 * 2, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = get_graph_feature(x, k=self.k)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x_128 = torch.unsqueeze(x3, 3)
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x = get_graph_feature(x3, k=self.k)
        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x_256 = torch.unsqueeze(x4, 3)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x = get_graph_feature(x4, k=self.k)
        x = F.relu(self.bn5(self.conv5(x)))
        x5 = x.max(dim=-1, keepdim=False)[0]
        x_512 = torch.unsqueeze(x5, 3)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x = get_graph_feature(x5, k=self.k)
        x = F.relu(self.bn6(self.conv6(x)))
        x6 = x.max(dim=-1, keepdim=False)[0]
        x_1024 = torch.unsqueeze(x6, 3)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class Latentfeature(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature, 1)
        return latentfeature


class RP_PCN(nn.Module):
    """RP-PCN 生成器（简化版，去掉判别器，只做补全）"""

    def __init__(self, num_scales=3, each_scales_size=1,
                 point_scales_list=[2048, 512, 256], crop_point_num=1024):
        super(RP_PCN, self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)

        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc1_1 = nn.Linear(1024, 256 * 512)
        self.fc2_1 = nn.Linear(512, 128 * 256)
        self.fc3_1 = nn.Linear(256, 128 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 256), 1)
        self.conv2_1 = torch.nn.Conv1d(256, 6, 1)

    def forward(self, x):
        """
        输入: x 是列表 [input1, input2, input3]
              input1: (B, 2048, 3)
              input2: (B, 512, 3)
              input3: (B, 256, 3)
        输出: fine (B, crop_point_num, 3)  补全后的精细点云
        """
        x = self.latentfeature(x)

        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x_3 = F.relu(self.fc3(x_2))

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 128, 3)

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 256, 128)
        pc2_xyz = self.conv2_1(pc2_feat)

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 256)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 128, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 256, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 256, int(self.crop_point_num / 256), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc3_xyz  # 只返回最终的精细点云


if __name__ == '__main__':
    input1 = torch.randn(4, 2048, 3)
    input2 = torch.randn(4, 512, 3)
    input3 = torch.randn(4, 256, 3)
    input_ = [input1, input2, input3]

    model = RP_PCN(3, 1, [2048, 512, 256], 1024)
    out = model(input_)
    print(f"输入:  [2048, 512, 256] 三个尺度的点云")
    print(f"输出:  {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
