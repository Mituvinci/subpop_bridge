# BRIDGE: Structured Role-Specialists for Robust Subpopulation Shift

This repository provides the official implementation for the NeurIPS 2026 submission:
*BRIDGE: Structured Role-Specialists for Robust Subpopulation Shift*.

## Overview

BRIDGE (Balanced Role-specialists with IsoMax Distance-based Group Experts) is a structured multi-head framework for subpopulation robustness. It induces head diversity through explicit specialization mechanisms adapted to subgroup observability:

- **With group annotations:** minority-specialized, majority-specialized, and balanced heads defined by group frequencies.
- **Without group annotations (AG):** sequential covariance-regularized prototype decorrelation.

BRIDGE uses only 3 heads (vs. 15 in DPE) and achieves competitive or superior worst-group accuracy on 8 SubpopBench datasets.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, CUDA 12.1 on NVIDIA A30/A100 GPUs.

## Data Preparation

BRIDGE uses the SubpopBench data format. Download datasets following [SubpopBench](https://github.com/YyzHarry/SubpopBench):

| Dataset | Domain | Classes | Groups | Download |
|---------|--------|---------|--------|----------|
| Waterbirds | Image | 2 | 4 | [link](https://github.com/YyzHarry/SubpopBench) |
| CelebA | Image | 2 | 4 | [link](https://github.com/YyzHarry/SubpopBench) |
| MetaShift | Image | 2 | 4 | [link](https://github.com/YyzHarry/SubpopBench) |
| CheXpert | Medical | 2 | 4 | [link](https://github.com/YyzHarry/SubpopBench) |
| CivilComments | Text | 2 | 4 | [link](https://github.com/YyzHarry/SubpopBench) |
| MultiNLI | Text | 3 | 6 | [link](https://github.com/YyzHarry/SubpopBench) |
| ImageNetBG | Image (AG) | 9 | 18 | [link](https://github.com/YyzHarry/SubpopBench) |
| Living17 | Image (AG) | 17 | 34 | [link](https://github.com/YyzHarry/SubpopBench) |

After downloading, prepare manifests:

```bash
python data/prepare_subpopbench.py
```

## Training

BRIDGE training follows a multi-stage pipeline. Each dataset has a self-contained script in `scripts/` that runs the full pipeline.

### Full pipeline (recommended)

Each script chains: (1) ERM backbone training, (2) optional specialist merging, and (3) BRIDGE head training with multiple rho values.

```bash
# Example: Waterbirds
SEED=0 sbatch scripts/run_waterbirds.sh

# Example: CelebA
SEED=0 sbatch scripts/run_celeba.sh

# Example: CivilComments (text, uses DeBERTa)
SEED=0 sbatch scripts/run_civilcomments.sh

# AG datasets (no group annotations)
sbatch scripts/run_imagenetbg.sh
sbatch scripts/run_living17.sh
```

### Individual stages

**Stage 1: ERM backbone**
```bash
python train_erm.py \
    --dataset waterbirds \
    --backbone resnet50 \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --seed 0
```

**Stage 2: BRIDGE heads (with group annotations)**
```bash
python train_bridge.py \
    --dataset waterbirds \
    --backbone resnet50 \
    --merged-ckpt path/to/erm_checkpoint.pt \
    --role-strength 0.7 \
    --epochs 20 \
    --lr 5e-4 \
    --entropic-scale 30 \
    --seed 0
```

**Stage 2: BRIDGE heads (AG, no group annotations)**
```bash
python train_bridge_ag.py \
    --dataset imagenetbg \
    --backbone resnet50 \
    --merged-ckpt path/to/erm_checkpoint.pt \
    --cov-reg 1e4 \
    --lr 1e-3 \
    --entropic-scale 10 \
    --epochs 50 \
    --seed 0
```

## Evaluation

Evaluation is performed automatically at the end of training. Each run saves:
- `eval_test.json`: per-group accuracies, WGA, gap, and average accuracy.
- `model_best_wga.pt`: best checkpoint by validation WGA.

## Results

### With group annotations (Table 1)

| Dataset | ERM | DFR | GroupDRO | DPE | BRIDGE |
|---------|-----|-----|----------|-----|--------|
| Waterbirds | 72.4 | 90.2 | 90.6 | 92.5 | 92.3 |
| CelebA | 41.1 | 88.9 | 88.6 | 91.9 | 92.7 |
| MetaShift | 71.5 | 80.1 | 76.8 | 86.1 | 87.4 |
| CheXpert | 60.7 | 70.5 | 73.2 | 71.1 | 76.1 |
| CivilComments | 57.4 | 69.2 | 69.7 | 70.3 | 73.2 |
| MultiNLI | 67.9 | 72.4 | 77.7 | 75.2 | 76.1 |

### Without group annotations (Table 2)

| Dataset | ERM | DFR | DPE | BRIDGE |
|---------|-----|-----|-----|--------|
| ImageNetBG | 72.0 | -- | 79.1 | 79.1 |
| Living17 | 56.3 | -- | 67.2 | 65.8 |

## Code Structure

```
subpop_bridge/
├── README.md
├── requirements.txt
├── train_erm.py           # Stage 1: ERM backbone training
├── train_bridge.py        # Stage 2: BRIDGE with group annotations
├── train_bridge_ag.py     # Stage 2: BRIDGE AG (covariance diversification)
├── merge_specialists.py   # Optional: Optuna-based specialist merging
├── lib/
│   ├── subpop_common.py   # Dataset loading, SubpopBench format
│   └── backbones.py       # ResNet-50, DeBERTa backbones
├── data/
│   ├── prepare_subpopbench.py   # Manifest generation for SubpopBench
│   ├── prepare_fitzpatrick.py   # Fitzpatrick skin lesion dataset
│   └── prepare_ham10000.py      # HAM10000 dataset
└── scripts/
    ├── run_waterbirds.sh
    ├── run_celeba.sh
    ├── run_metashift.sh
    ├── run_chexpert.sh
    ├── run_civilcomments.sh
    ├── run_multinli.sh
    ├── run_imagenetbg.sh
    ├── run_living17.sh
    └── run_fitzpatrick.sh
```

## License

MIT
