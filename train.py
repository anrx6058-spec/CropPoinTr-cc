#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练脚本
支持 PCN, RP-PCN, JCS-Net

用法:
    python train.py --model pcn --epochs 10 --batch_size 4
    python train.py --model rp_pcn --epochs 10 --batch_size 2
    python train.py --model jcs_net --epochs 10 --batch_size 2
"""

import os
import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model, list_models
from data.crops3d_loader import Crops3DDataset, Crops3DDatasetMultiScale
from utils.metrics import ChamferLoss


def get_args():
    parser = argparse.ArgumentParser(description='点云补全训练')

    # 模型
    parser.add_argument('--model', type=str, default='pcn',
                        choices=list_models(),
                        help=f'选择模型: {list_models()}')

    # 数据
    parser.add_argument('--dataset_root', type=str,
                        default=r'C:\Users\Chenc\Desktop\跑代码\Crops3D',
                        help='Crops3D数据集根目录')
    parser.add_argument('--crop_type', type=str, default='Rapeseed',
                        help='作物类型')
    parser.add_argument('--npoints_input', type=int, default=2048,
                        help='输入点云点数')
    parser.add_argument('--npoints_target', type=int, default=1024,
                        help='目标点云点数')

    # 训练
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--lr_decay', type=float, default=0.7,
                        help='学习率衰减率')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='学习率衰减步长')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')

    # 其他
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: auto/cpu/cuda')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='每多少轮保存一次')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='每多少batch打印一次日志')

    args = parser.parse_args()
    return args


def setup_device(device_str):
    """设置计算设备"""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def setup_seed(seed):
    """设置随机种子保证可复现"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子: {seed}")


def get_dataloader(args, split='train'):
    """根据模型类型获取对应的数据加载器"""
    if args.model in ['rp_pcn', 'jcs_net']:
        # 多尺度输入
        dataset = Crops3DDatasetMultiScale(
            root_dir=args.dataset_root,
            crop_type=args.crop_type,
            split=split,
            scales=[2048, 512, 256],
            npoints_target=args.npoints_target,
            normalize=True,
            seed=args.seed
        )
    else:
        # 单尺度输入 (PCN)
        dataset = Crops3DDataset(
            root_dir=args.dataset_root,
            crop_type=args.crop_type,
            split=split,
            npoints_input=args.npoints_input,
            npoints_target=args.npoints_target,
            normalize=True,
            seed=args.seed
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        drop_last=(split == 'train')
    )
    return dataloader


def train_epoch(model, dataloader, optimizer, criterion, device, args, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        if args.model in ['rp_pcn', 'jcs_net']:
            # 多尺度输入: inputs是列表, target是tensor
            inputs, target = batch_data
            inputs = [x.to(device) for x in inputs]
            target = target.to(device)
            output = model(inputs)
        else:
            # 单尺度输入
            input_pc, target = batch_data
            input_pc = input_pc.to(device)
            target = target.to(device)
            output = model(input_pc)

        # JCS-Net 输出是元组 (fine, seg_logits)
        if args.model == 'jcs_net':
            output = output[0]

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % args.log_interval == 0:
            print(f"  Epoch[{epoch}] Batch[{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, args):
    """验证"""
    model.eval()
    total_loss = 0.0
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

            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, args, save_dir):
    """保存模型"""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }
    path = os.path.join(save_dir, f'{args.model}_epoch{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"模型已保存: {path}")


def main():
    args = get_args()
    print("=" * 60)
    print("点云补全训练")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.crop_type}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    # 设置
    setup_seed(args.seed)
    device = setup_device(args.device)

    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 数据加载器
    print("\n[加载数据]")
    train_loader = get_dataloader(args, split='train')
    test_loader = get_dataloader(args, split='test')

    # 模型
    print(f"\n[创建模型: {args.model}]")
    if args.model == 'pcn':
        model = get_model('pcn', num_points=args.npoints_target)
    elif args.model == 'rp_pcn':
        model = get_model('rp_pcn', crop_point_num=args.npoints_target)
    elif args.model == 'jcs_net':
        model = get_model('jcs_net', crop_point_num=args.npoints_target)

    model = model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    criterion = ChamferLoss()

    # 训练循环
    print("\n[开始训练]")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args, epoch)
        val_loss = validate(model, test_loader, criterion, device, args)
        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"Epoch[{epoch}/{args.epochs}] "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}  Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, args, save_dir)
            print(f"  -> 最佳模型 (Val: {val_loss:.4f})")

        # 定期保存
        if epoch % args.save_epoch == 0:
            save_checkpoint(model, optimizer, epoch, args, save_dir)

    print("\n[训练完成]")
    print(f"最佳验证Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")


if __name__ == '__main__':
    main()
