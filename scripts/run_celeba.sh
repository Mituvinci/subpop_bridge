#!/usr/bin/env bash
# Full BRIDGE chain for CelebA: ERM* -> BRIDGE (attr + noattr)
#
# Usage:
#   SEED=0 sbatch --time=08:00:00 --partition=gpu_2day \
#     --mem=48G --gres=gpu:1 --cpus-per-task=8 \
#     --job-name=chain_celeba_s0 \
#     --output=logs/chain_celeba_s0_%j.out \
#     --error=logs/chain_celeba_s0_%j.err \
#     scripts/run_celeba.sh

set -eo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /shared/software/conda/etc/profile.d/conda.sh
conda activate pytorch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

DATASET=celeba
BACKBONE=resnet50
SEED=${SEED:-0}

echo "============================================================"
echo "  BRIDGE chain: $DATASET (seed $SEED)"
echo "  Job: ${SLURM_JOB_ID:-local}  Started: $(date)"
echo "============================================================"

# -- Step 1: ERM* backbone training --
echo ""
echo "=== STEP 1: ERM* backbone training ==="
echo "Started: $(date)"

python -u train_erm.py \
    --dataset "$DATASET" --method erm --backbone "$BACKBONE" \
    --batch-size 128 --epochs 120 --optimizer sgd \
    --lr 1e-3 --weight-decay 0 --num-workers 8 \
    --seed "$SEED" --out-dir results/erm_rn50

echo "=== STEP 1 COMPLETE: $(date) ==="

ERM_CKPT=$(ls -t results/erm_rn50/${DATASET}/erm/*_s${SEED}/model_best_wga.pt 2>/dev/null | head -1)
if [ -z "$ERM_CKPT" ] || [ ! -f "$ERM_CKPT" ]; then
    echo "ERROR: ERM* checkpoint not found after training" >&2; exit 1
fi
echo "[chain] ERM* checkpoint: $ERM_CKPT"

# -- Step 2: BRIDGE WITH attribute annotations --
echo ""
echo "=== STEP 2: BRIDGE WITH attribute annotations ==="
echo "Started: $(date)"

for RS in 0.5 0.7 0.9; do
    echo "[chain] BRIDGE rho=$RS val-balance=attribute seed=$SEED"
    python -u train_bridge.py \
        --dataset "$DATASET" --backbone "$BACKBONE" \
        --merged-ckpt "$ERM_CKPT" \
        --val-balance attribute --inference max \
        --role-strength "$RS" --seed "$SEED" \
        --epochs 20 --batch-size 256 \
        --lr 5e-4 --entropic-scale 30 --wd-weight 10
done

echo "=== STEP 2 COMPLETE: $(date) ==="

# -- Step 3: BRIDGE WITHOUT attribute annotations --
echo ""
echo "=== STEP 3: BRIDGE WITHOUT attribute annotations ==="
echo "Started: $(date)"

for RS in 0.5 0.7 0.9; do
    echo "[chain] BRIDGE rho=$RS --no-group-roles seed=$SEED"
    python -u train_bridge.py \
        --dataset "$DATASET" --backbone "$BACKBONE" \
        --merged-ckpt "$ERM_CKPT" \
        --val-balance attribute --inference max \
        --role-strength "$RS" --seed "$SEED" \
        --epochs 20 --batch-size 256 \
        --lr 5e-4 --entropic-scale 30 --wd-weight 10 \
        --no-group-roles
done

echo "=== STEP 3 COMPLETE: $(date) ==="

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE: $DATASET seed $SEED"
echo "  ERM*: $ERM_CKPT"
echo "  Finished: $(date)"
echo "============================================================"
