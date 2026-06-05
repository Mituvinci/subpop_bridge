#!/usr/bin/env bash
# Full BRIDGE chain for ImageNetBG: ERM* -> train_bridge_ag (covariance, AG)
#
# Usage:
#   SEED=0 sbatch --time=08:00:00 --partition=gpu_2day \
#     --mem=48G --gres=gpu:1 --cpus-per-task=8 \
#     --job-name=chain_imagenetbg_s0 \
#     --output=logs/chain_imagenetbg_s0_%j.out \
#     --error=logs/chain_imagenetbg_s0_%j.err \
#     scripts/run_imagenetbg.sh

set -eo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /shared/software/conda/etc/profile.d/conda.sh
conda activate pytorch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

DATASET=imagenetbg
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

# -- Step 2: AG-BRIDGE (covariance-regularized, no group annotations) --
echo ""
echo "=== STEP 2: train_bridge_ag (covariance sweep, AG) ==="
echo "Started: $(date)"

for COV in 3e3 5e3 7e3 1e4 2e4 3e4; do
    echo "[chain] train_bridge_ag cov-reg=$COV seed=$SEED"
    python -u train_bridge_ag.py \
        --dataset "$DATASET" --backbone "$BACKBONE" \
        --merged-ckpt "$ERM_CKPT" \
        --cov-reg "$COV" \
        --lr 1e-3 --entropic-scale 10 \
        --epochs 50 --batch-size 256 \
        --seed "$SEED" --num-workers 4
done

echo "=== STEP 2 COMPLETE: $(date) ==="

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE: $DATASET seed $SEED"
echo "  ERM*: $ERM_CKPT"
echo "  Finished: $(date)"
echo "============================================================"
