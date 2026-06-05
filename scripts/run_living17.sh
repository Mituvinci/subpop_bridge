#!/usr/bin/env bash
# BRIDGE-AG chain for Living17 (seed 0)
# AG dataset -- no group annotations, so NO Step 2 (with attr).
# Step 2: s26f WITHOUT attribute annotations (--no-group-roles)
#
# Usage:
#   sbatch --time=04:00:00 --partition=gpu_2day \
#     --mem=48G --gres=gpu:1 --cpus-per-task=4 \
#     --job-name=chain3_living17_s0 \
#     --output=logs/chain3_living17_s0_%j.out \
#     --error=logs/chain3_living17_s0_%j.err \
#     scripts/chain_3stage_living17.sh

set -eo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /shared/software/conda/etc/profile.d/conda.sh
conda activate pytorch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

DATASET=living17
BACKBONE=resnet50
SEED=${SEED:-0}

ERM_CKPT=$(ls -t results/s19_subpop_rn50_star/${DATASET}/erm/*_s${SEED}/model_best_wga.pt results/s19_subpop_rn50_star/${DATASET}/erm/*_s${SEED}/model.pt 2>/dev/null | head -1 || true)

echo "============================================================"
echo "  BRIDGE-AG chain: $DATASET (seed $SEED) [AG dataset]"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  ERM*:   $ERM_CKPT"
echo "  Started: $(date)"
echo "============================================================"

if [ ! -f "$ERM_CKPT" ]; then
    echo "ERROR: ERM* checkpoint not found" >&2; exit 1
fi

# -- Step 1: AG-BRIDGE (covariance-regularized, no group annotations) --
echo ""
echo "=== STEP 1: train_bridge_ag (covariance sweep, AG) ==="
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
echo "  ERM*:   $ERM_CKPT"
echo "  Finished: $(date)"
echo "============================================================"
