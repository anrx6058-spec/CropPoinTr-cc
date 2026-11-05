#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本
加载训练好的模型，在测试集上评估

用法:
    python evaluate.py --model pcn --checkpoint checkpoints/pcn/pcn_epoch100.pth
    python evaluate.py --model rp_pcn --checkpoint checkpoints/rp_pcn/rp_pcn_epoch50.pth
"""

import os
import sys
import argparse
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model, list_models
from data.crops3d_loader import Crops3DDataset, Crops3DDatasetMultiScale
from utils.metrics import ChamferLoss, compute_fscore


def get_args():
    parser = argparse.ArgumentParser(description='点云补全评估')
    parser.add_argument('--model', type=str, required=True,
                        choices=list_models(),
                        help=f'选择模型: {list_models()}')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--dataset_root', type=str,
                        default=r'C:\Users\Chenc\Desktop\跑代码\Crops3D',
                        help='Crops3D数据集根目录')
    parser.add_argument('--crop_type', type=str, default='Rapeseed',
                        help='作物类型')
    parser.add_argument('--npoints_input', type=int, default=2048)
    parser.add_argument('--npoints_target', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/eval_results.json',
                        help='结果保存路径')
    return parser.parse_args()


def setup_device(device_str):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"使用设备: {device}")
    return device


def get_test_dataloader(args):
    if args.model in ['rp_pcn', 'jcs_net']:
        dataset = Crops3DDatasetMultiScale(
            root_dir=args.dataset_root,
            crop_type=args.crop_type,
            split='test',
            scales=[2048, 512, 256],
            npoints_target=args.npoints_target,
            normalize=True,
            seed=args.seed
        )
    else:
        dataset = Crops3DDataset(
            root_dir=args.dataset_root,
            crop_type=args.crop_type,
            split='test',
            npoints_input=args.npoints_input,
            npoints_target=args.npoints_target,
            normalize=True,
            seed=args.seed
        )

    return DataLoader(dataset, batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers)


def evaluate(model, dataloader, device, args):
    """评估模型"""
    model.eval()
    criterion = ChamferLoss()

    total_cd = 0.0
    total_fscore = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in dataloader:
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

            # Chamfer Distance
            cd = criterion(output, target)
            total_cd += cd.item()

            # F-Score
            fscore = compute_fscore(output, target)
            total_fscore += fscore.item()

            num_batches += 1

    avg_cd = total_cd / num_batches
    avg_fscore = total_fscore / num_batches

    return {
        'chamfer_distance': round(avg_cd, 4),
        'f_score': round(avg_fscore, 4),
        'num_samples': len(dataloader.dataset),
        'num_batches': num_batches
    }


def main():
    args = get_args()
    print("=" * 60)
    print("点云补全评估")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"权重: {args.checkpoint}")
    print("=" * 60)

    device = setup_device(args.device)

    # 加载模型
    print("\n[加载模型]")
    if args.model == 'pcn':
        model = get_model('pcn', num_points=args.npoints_target)
    elif args.model == 'rp_pcn':
        model = get_model('rp_pcn', crop_point_num=args.npoints_target)
    elif args.model == 'jcs_net':
        model = get_model('jcs_net', crop_point_num=args.npoints_target)

    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'

    model = model.to(device)
    print(f"已加载 epoch {epoch} 的权重")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 加载数据
    print("\n[加载测试数据]")
    test_loader = get_test_dataloader(args)

    # 评估
    print("\n[开始评估]")
    results = evaluate(model, test_loader, device, args)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"Chamfer Distance: {results['chamfer_distance']}")
    print(f"F-Score:          {results['f_score']}")
    print(f"测试样本数:       {results['num_samples']}")
    print("=" * 60)

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results['model'] = args.model
    results['checkpoint'] = args.checkpoint
    results['epoch'] = epoch

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
