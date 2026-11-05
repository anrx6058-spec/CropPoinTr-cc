# CropPoinTr

Official implementation of **CropPoinTr**, a crop-oriented point cloud completion network for single-plant phenotyping.

CropPoinTr is built on the AdaPoinTr completion backbone and introduces two lightweight, crop-specific plug-in modules:

- **HMGF (Hierarchical Multi-scale Global Fusion)** — auxiliary downsampling branches (512 and 256 points) with shared lightweight PointNet encoders that fuse multi-scale global features, preserving both whole-plant contour and fine organ details.
- **DQS (Density-aware Query Selection)** — estimates local point density from k-nearest neighborhoods and prioritizes sparse regions (stem tips, leaf apices) when selecting completion queries.

Given a viewpoint-occluded 2048-point rapeseed scan, CropPoinTr reconstructs the complete 4096-point plant, enabling organ-resolved trait extraction (plant height, leaf area, organ-level fidelity).

The repository also includes the baseline models used for comparison (PCN, RP-PCN).

## Requirements

- Python 3.9+
- PyTorch 2.x with CUDA (developed and tested with PyTorch 2.6 + CUDA 12.8 on an NVIDIA RTX 5090)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data

Experiments use the publicly available **Crops3D** dataset (Zhu et al., 2024), which provides terrestrial-laser-scanning point clouds of multiple crops with organ-level annotations:
> Zhu, J., et al. (2024). Crops3D: A diverse 3D crop dataset for realistic perception and generalizable 3D point cloud analysis in agriculture. *Plant Phenomics*.

Download Crops3D from its official repository and set the dataset root accordingly (the default path inside the scripts is a local path and must be changed):

```bash
--dataset_root /path/to/Crops3D --crop_type Rapeseed
```

## Training

```bash
# CropPoinTr (JCS-Net)
python train.py --model jcs_net --dataset_root /path/to/Crops3D --epochs 200 --batch_size 2

# Baselines
python train.py --model pcn --epochs 200 --batch_size 4
python train.py --model rp_pcn --epochs 200 --batch_size 2
```

Key arguments: `--npoints_input` (default 2048), `--npoints_target` (default 1024; use 4096 for full-resolution completion), `--lr` (default 1e-4), `--seed` (default 42). Refer to the accompanying paper and its Supplemental Information for the full experimental protocols and hyperparameters.

## Evaluation

```bash
python evaluate.py --model jcs_net --checkpoint checkpoints/jcs_net/best.pth --dataset_root /path/to/Crops3D
```

Metrics reported in the paper (CD-L2, F-score at multiple thresholds, organ-level Chamfer distance) are implemented in `utils/metrics.py`.

## Pretrained models

Pretrained weights will be released here / on Zenodo: *(link to be added)*

## Repository structure

```
├── train.py            # unified training script (PCN / RP-PCN / JCS-Net)
├── evaluate.py         # evaluation & metric computation
├── run.py              # experiment runner
├── quick_test.py       # quick inference demo
├── models/
│   ├── jcs_net.py      # CropPoinTr (JCS-Net): AdaPoinTr + HMGF + DQS
│   ├── pcn.py          # PCN baseline
│   └── rp_pcn.py       # RP-PCN baseline
├── data/
│   └── crops3d_loader.py
├── utils/
│   └── metrics.py
└── checkpoints/
```

## License

This project is released under the MIT License (see `LICENSE`).

## Citation

If you find this work useful, please cite:

*(to be added upon publication)*
