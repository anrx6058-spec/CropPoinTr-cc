"""
评估指标：Chamfer Distance, F-Score
"""

import torch
import torch.nn as nn


def chamfer_distance_numpy(array1, array2):
    """
    计算两个点云之间的 Chamfer Distance
    array1, array2: (B, N, 3)
    返回: scalar
    """
    def array2samples_distance(array1, array2):
        num_point1, num_features1 = array1.shape
        num_point2, num_features2 = array2.shape
        expanded_array1 = array1.repeat(num_point2, 1)
        expanded_array2 = torch.reshape(
            torch.unsqueeze(array2, 1).repeat(1, num_point1, 1),
            (-1, num_features2))
        distances = (expanded_array1 - expanded_array2) * (expanded_array1 - expanded_array2)
        distances = torch.sum(distances, dim=1)
        distances = torch.reshape(distances, (num_point2, num_point1))
        distances = torch.min(distances, dim=1)[0]
        distances = torch.mean(distances)
        return distances

    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (0.5 * av_dist1 + 0.5 * av_dist2) / batch_size
    return dist * 100


class ChamferLoss(nn.Module):
    """Chamfer Distance Loss"""
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, pred, target):
        return chamfer_distance_numpy(pred, target)


def compute_fscore(pred, target, threshold=0.01):
    """
    计算 F-Score
    pred, target: (B, N, 3)
    threshold: 判断为"正确"的距离阈值
    """
    def nn_distance(array1, array2):
        """计算array1中每个点到array2的最近距离"""
        diff = array1[:, None, :] - array2[None, :, :]  # (N1, N2, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # (N1, N2)
        return torch.min(dist, dim=1)[0]  # (N1,)

    batch_size = pred.shape[0]
    f_scores = []

    for i in range(batch_size):
        pred_i = pred[i]
        target_i = target[i]

        # pred -> target 的最近距离
        d1 = nn_distance(pred_i, target_i)
        # target -> pred 的最近距离
        d2 = nn_distance(target_i, pred_i)

        # 精度: pred中有多少点在target附近
        precision = (d1 < threshold ** 2).float().mean()
        # 召回: target中有多少点在pred附近
        recall = (d2 < threshold ** 2).float().mean()

        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = torch.tensor(0.0)
        f_scores.append(f_score)

    return torch.stack(f_scores).mean()
