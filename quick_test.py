#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本
不训练完整epoch，只验证：
1. 数据能正常加载
2. 模型能正常前向传播
3. 损失能正常计算和反向传播

用法:
    python quick_test.py --model pcn
    python quick_test.py --model rp_pcn
    python quick_test.py --model jcs_net
"""

import os
import sys
import argparse
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model, list_models
from data.crops3d_loader import Crops3DDataset, Crops3DDatasetMultiScale
from utils.metrics import ChamferLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pcn', choices=list_models())
    parser.add_argument('--dataset_root', type=str,
                        default=r'C:\Users\Chenc\Desktop\跑代码\Crops3D')
    parser.add_argument('--crop_type', type=str, default='Rapeseed')
    parser.add_argument('--npoints_target', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_batches', type=int, default=3,
                        help='测试多少个batch')
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main():
    args = get_args()

    print("=" * 60)
    print("快速验证")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset_root}")
    print(f"Batch: {args.batch_size}, 测试Batch数: {args.num_batches}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 1. 测试数据加载
    print("[1/4] 测试数据加载...")
    start = time.time()
    if args.model in ['rp_pcn', 'jcs_net']:
        dataset = Crops3DDatasetMultiScale(
            root_dir=args.dataset_root,
            crop_type=args.crop_type,
            split='train',
            scales=[2048, 512, 256],
            npoints_target=args.npoints_target
        )
    else:
        dataset = Crops3DDataset(
            root_dir=args.dataset_root,
            crop_type=args.crop_type,
            split='train',
            npoints_input=2048,
            npoints_target=args.npoints_target
        )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"  [OK] 数据集大小: {len(dataset)}")
    print(f"  [OK] 加载耗时: {time.time()-start:.2f}s\n")

    # 2. 测试模型创建
    print("[2/4] 测试模型创建...")
    if args.model == 'pcn':
        model = get_model('pcn', num_points=args.npoints_target)
    elif args.model == 'rp_pcn':
        model = get_model('rp_pcn', crop_point_num=args.npoints_target)
    else:
        model = get_model('jcs_net', crop_point_num=args.npoints_target)

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [OK] 模型创建成功")
    print(f"  [OK] 参数量: {params:.2f} M\n")

    # 3. 测试前向传播
    print("[3/4] 测试前向传播...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = ChamferLoss()

    batch_times = []
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        t0 = time.time()
        optimizer.zero_grad()

        if args.model in ['rp_pcn', 'jcs_net']:
            inputs, target = batch_data
            inputs = [x.to(device) for x in inputs]
            target = target.to(device)
            output = model(inputs)
        else:
            input_pc, target = batch_data
            input_pc = input_pc.to(device)
            target = target.to(device)
            output = model(input_pc)

        if args.model == 'jcs_net':
            output = output[0]

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        t1 = time.time()
        batch_times.append(t1 - t0)

        print(f"  Batch {batch_idx+1}: Loss={loss.item():.4f}, Time={t1-t0:.2f}s")

    avg_time = sum(batch_times) / len(batch_times)
    print(f"  [OK] 平均每个batch: {avg_time:.2f}s")
    print(f"  [OK] 预估1个epoch: {avg_time * len(dataloader) / 60:.1f} 分钟\n")

    # 4. 测试评估
    print("[4/4] 测试评估指标...")
    from utils.metrics import compute_fscore
    with torch.no_grad():
        fscore = compute_fscore(output, target)
    print(f"  [OK] F-Score: {fscore.item():.4f}\n")

    print("=" * 60)
    print("[PASS] 全部验证通过！")
    print("=" * 60)
    print(f"\n模型: {args.model}")
    print(f"参数量: {params:.2f} M")
    print(f"每个batch: {avg_time:.2f}s")
    print(f"预估每epoch: {avg_time * len(dataloader) / 60:.1f} 分钟")
    print(f"\n可以开始正式训练了！")
    print(f"命令: python train.py --model {args.model} --epochs 200")


if __name__ == '__main__':
    main()
