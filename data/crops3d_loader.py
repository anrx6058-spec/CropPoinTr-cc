"""
数据加载器：适配 Crops3D 数据集
支持 .ply 文件读取、FPS采样、归一化、遮挡模拟
"""

import os
import struct
import numpy as np
import torch
import torch.utils.data as data


def read_ply_binary(filepath):
    """
    读取 binary_little_endian 格式的 .ply 文件
    自动解析 header 中的属性类型
    返回: Nx3 numpy array (xyz坐标)
    """
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                type_name = parts[-2]
                prop_name = parts[-1]
                properties.append((type_name, prop_name))

        if num_vertices == 0:
            raise ValueError(f"无法解析顶点数量: {filepath}")

        type_info = {
            'char': (1, 'b'), 'uchar': (1, 'B'),
            'short': (2, 'h'), 'ushort': (2, 'H'),
            'int': (4, 'i'), 'uint': (4, 'I'),
            'float': (4, 'f'), 'double': (8, 'd'),
        }

        vertex_bytes = 0
        fmt_chars = []
        for t, _ in properties:
            t_lower = t.lower()
            if t_lower in type_info:
                vertex_bytes += type_info[t_lower][0]
                fmt_chars.append(type_info[t_lower][1])
            else:
                vertex_bytes += 4
                fmt_chars.append('f')

        struct_fmt = '<' + ''.join(fmt_chars)

        vertices = []
        for _ in range(num_vertices):
            data_bytes = f.read(vertex_bytes)
            if len(data_bytes) < vertex_bytes:
                break
            values = struct.unpack(struct_fmt, data_bytes)
            vertices.append([values[0], values[1], values[2]])

    return np.array(vertices, dtype=np.float32)


def pc_normalize(pc):
    """点云归一化到单位球内"""
    pc = np.asarray(pc, dtype=np.float64)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc.astype(np.float32)


def farthest_point_sample_numpy(points, npoint):
    """numpy实现的farthest point sampling"""
    points = np.asarray(points, dtype=np.float64)
    N, C = points.shape
    if N <= npoint:
        idx = np.arange(N)
        repeat_times = (npoint // N) + 1
        idx = np.tile(idx, repeat_times)[:npoint]
        return idx

    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return centroids


class Crops3DDataset(data.Dataset):
    """
    Crops3D 数据集加载器
    用于点云补全任务：输入(不完整) -> 目标(完整)
    """
    def __init__(self, root_dir, crop_type='Rapeseed', split='train',
                 npoints_input=2048, npoints_target=1024, normalize=True,
                 seed=42):
        super(Crops3DDataset, self).__init__()
        self.root_dir = root_dir
        self.crop_type = crop_type
        self.split = split
        self.npoints_input = npoints_input
        self.npoints_target = npoints_target
        self.normalize = normalize

        # 构建文件列表
        crop_dir = os.path.join(root_dir, crop_type)
        if not os.path.exists(crop_dir):
            raise FileNotFoundError(f"作物目录不存在: {crop_dir}")

        all_files = sorted([f for f in os.listdir(crop_dir) if f.endswith('.ply')])

        # 划分 train/test
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))
        split_idx = int(len(all_files) * 0.8)

        if split == 'train':
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.file_list = [os.path.join(crop_dir, all_files[i]) for i in selected]
        print(f"[{split}] {crop_type}: 加载 {len(self.file_list)} 个 .ply 文件")

    def __getitem__(self, index):
        filepath = self.file_list[index]
        points = read_ply_binary(filepath)  # Nx3

        # FPS采样到足够多的点
        if len(points) > self.npoints_input * 2:
            sampled_idx = farthest_point_sample_numpy(points, self.npoints_input * 2)
            points = points[sampled_idx]

        # 归一化
        if self.normalize:
            points = pc_normalize(points)

        # 模拟遮挡：
        # 输入点云（模拟表面，点数较少）
        if len(points) >= self.npoints_input:
            input_idx = farthest_point_sample_numpy(points, self.npoints_input)
        else:
            input_idx = np.arange(len(points))
            while len(input_idx) < self.npoints_input:
                input_idx = np.concatenate([input_idx, np.arange(len(points))])
            input_idx = input_idx[:self.npoints_input]
        input_pc = points[input_idx]

        # 目标点云（完整，点数较多）
        if len(points) >= self.npoints_target:
            target_idx = farthest_point_sample_numpy(points, self.npoints_target)
        else:
            target_idx = np.arange(len(points))
            while len(target_idx) < self.npoints_target:
                target_idx = np.concatenate([target_idx, np.arange(len(points))])
            target_idx = target_idx[:self.npoints_target]
        target_pc = points[target_idx]

        input_pc = torch.from_numpy(input_pc).float()
        target_pc = torch.from_numpy(target_pc).float()

        return input_pc, target_pc

    def __len__(self):
        return len(self.file_list)


class Crops3DDatasetMultiScale(data.Dataset):
    """
    多尺度输入版本（用于 RP-PCN / JCS-Net）
    返回 [input1(2048), input2(512), input3(256)] 和 target
    """
    def __init__(self, root_dir, crop_type='Rapeseed', split='train',
                 scales=[2048, 512, 256], npoints_target=1024,
                 normalize=True, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.crop_type = crop_type
        self.split = split
        self.scales = scales
        self.npoints_target = npoints_target
        self.normalize = normalize

        crop_dir = os.path.join(root_dir, crop_type)
        if not os.path.exists(crop_dir):
            raise FileNotFoundError(f"作物目录不存在: {crop_dir}")

        all_files = sorted([f for f in os.listdir(crop_dir) if f.endswith('.ply')])

        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))
        split_idx = int(len(all_files) * 0.8)

        if split == 'train':
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.file_list = [os.path.join(crop_dir, all_files[i]) for i in selected]
        print(f"[{split}] {crop_type}: 加载 {len(self.file_list)} 个 .ply 文件")

    def __getitem__(self, index):
        filepath = self.file_list[index]
        points = read_ply_binary(filepath)

        max_scale = max(self.scales + [self.npoints_target])
        if len(points) > max_scale * 2:
            sampled_idx = farthest_point_sample_numpy(points, max_scale * 2)
            points = points[sampled_idx]

        if self.normalize:
            points = pc_normalize(points)

        # 多尺度输入
        inputs = []
        for scale in self.scales:
            if len(points) >= scale:
                idx = farthest_point_sample_numpy(points, scale)
            else:
                idx = np.arange(len(points))
                while len(idx) < scale:
                    idx = np.concatenate([idx, np.arange(len(points))])
                idx = idx[:scale]
            inputs.append(torch.from_numpy(points[idx]).float())

        # 目标
        if len(points) >= self.npoints_target:
            target_idx = farthest_point_sample_numpy(points, self.npoints_target)
        else:
            target_idx = np.arange(len(points))
            while len(target_idx) < self.npoints_target:
                target_idx = np.concatenate([target_idx, np.arange(len(points))])
            target_idx = target_idx[:self.npoints_target]
        target = torch.from_numpy(points[target_idx]).float()

        return inputs, target

    def __len__(self):
        return len(self.file_list)
