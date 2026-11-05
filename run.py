#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行脚本：训练所有模型并评估
支持在多个数据集上自动运行

用法:
    python run.py --mode all --epochs 200
    python run.py --mode all --quick_test    # 快速测试(2个epoch)
    python run.py --model pcn --epochs 5     # 只跑单个模型
"""

import os
import sys
import argparse
import json
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import list_models


def get_args():
    parser = argparse.ArgumentParser(description='一键运行对比实验')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'eval', 'all', 'test'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default=None,
                        help='只运行指定模型（默认运行所有）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小（PCN用这个，多尺度模型会自动减半）')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式（只跑2个epoch）')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=[r'C:\Users\Chenc\Desktop\跑代码\Crops3D',
                                 r'C:\Users\Chenc\Desktop\跑代码\Crops3D_10k'],
                        help='数据集路径列表，可指定多个')
    parser.add_argument('--python', type=str, default='python',
                        help='Python解释器路径')
    return parser.parse_args()


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[运行] {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[错误] {description} 运行失败! 返回码: {result.returncode}")
    else:
        print(f"[完成] {description} 耗时: {elapsed:.1f}s")

    return result.returncode == 0


def train_model(model_name, dataset_root, epochs, batch_size, python, quick_test):
    """训练单个模型"""
    bs = batch_size
    if model_name in ['rp_pcn', 'jcs_net']:
        bs = max(1, bs // 2)

    epochs = 2 if quick_test else epochs

    cmd = [
        python, 'train.py',
        '--model', model_name,
        '--dataset_root', dataset_root,
        '--epochs', str(epochs),
        '--batch_size', str(bs),
        '--save_epoch', str(max(1, epochs // 5)),
        '--log_interval', '5'
    ]

    dataset_name = os.path.basename(dataset_root)
    return run_command(cmd, f"训练 {model_name} [{dataset_name}]")


def eval_model(model_name, dataset_root, python):
    """评估单个模型"""
    checkpoint_dir = os.path.join('checkpoints', model_name)

    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pth'):
                checkpoints.append(f)

    if not checkpoints:
        print(f"[跳过] {model_name} 没有找到checkpoint")
        return False

    checkpoints.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]) if 'epoch' in x else 0)
    latest = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest)

    dataset_name = os.path.basename(dataset_root)
    cmd = [
        python, 'evaluate.py',
        '--model', model_name,
        '--checkpoint', checkpoint_path,
        '--dataset_root', dataset_root,
        '--output', f'results/{model_name}_{dataset_name}_eval.json'
    ]

    return run_command(cmd, f"评估 {model_name} [{dataset_name}]")


def summarize_results(datasets):
    """汇总所有评估结果（按数据集分组）"""
    print(f"\n{'='*60}")
    print("[结果汇总]")
    print(f"{'='*60}")

    results = {}
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("没有找到结果目录")
        return

    for dataset_root in datasets:
        dataset_name = os.path.basename(dataset_root)
        dataset_results = {}
        for f in os.listdir(results_dir):
            if f.endswith(f'_{dataset_name}_eval.json'):
                model_name = f.replace(f'_{dataset_name}_eval.json', '')
                with open(os.path.join(results_dir, f), 'r') as fp:
                    data = json.load(fp)
                    dataset_results[model_name] = data
        results[dataset_name] = dataset_results

    for dataset_name, dataset_results in results.items():
        if not dataset_results:
            continue
        print(f"\n--- 数据集: {dataset_name} ---")
        print(f"{'模型':<15} {'CD↓':<12} {'F-Score↑':<12} {'样本数':<8}")
        print("-" * 50)
        for model_name in sorted(dataset_results.keys()):
            r = dataset_results[model_name]
            cd = r.get('chamfer_distance', 'N/A')
            fs = r.get('f_score', 'N/A')
            n = r.get('num_samples', 'N/A')
            print(f"{model_name:<15} {str(cd):<12} {str(fs):<12} {str(n):<8}")

    # 保存汇总
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n汇总已保存: {summary_path}")


def main():
    args = get_args()

    print("=" * 60)
    print("油菜点云补全 对比实验")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"模型: {args.model if args.model else '全部'}")
    print(f"数据集: {args.datasets}")
    print(f"Epochs: {args.epochs}")
    print(f"快速测试: {args.quick_test}")
    print("=" * 60)

    # 确定要运行的模型
    if args.model:
        models = [args.model]
    else:
        models = list_models()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 训练
    if args.mode in ['train', 'all', 'test']:
        for dataset_root in args.datasets:
            for model_name in models:
                success = train_model(model_name, dataset_root, args.epochs, args.batch_size, args.python, args.quick_test)
                if not success:
                    print(f"[警告] {model_name} 在 {dataset_root} 训练失败，继续下一个...")

    # 评估
    if args.mode in ['eval', 'all', 'test']:
        for dataset_root in args.datasets:
            for model_name in models:
                success = eval_model(model_name, dataset_root, args.python)
                if not success:
                    print(f"[警告] {model_name} 在 {dataset_root} 评估失败，继续下一个...")

    # 汇总
    if args.mode in ['eval', 'all', 'test']:
        summarize_results(args.datasets)

    print(f"\n{'='*60}")
    print("[全部完成]")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
