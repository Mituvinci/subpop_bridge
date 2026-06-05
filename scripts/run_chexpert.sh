#!/usr/bin/env bash
# Two-stage BRIDGE chain for CheXpert (seed 0)
# Step 1: s26f WITH attribute annotations
# Step 2: s26f WITHOUT attribute annotations (--no-group-roles)
#
# Usage:
#   sbatch --time=08:00:00 --partition=gpu_2day \
#     --mem=48G --gres=gpu:1 --cpus-per-task=4 \
#     --job-name=chain3_chexpert_s0 \
#     --output=logs/chain3_chexpert_s0_%j.out \
#     --error=logs/chain3_chexpert_s0_%j.err \
#     scripts/chain_3stage_chexpert.sh

set -eo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /shared/software/conda/etc/profile.d/conda.sh
conda activate pytorch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

DATASET=chexpert
BACKBONE=resnet50
SEED=${SEED:-0}

ERM_CKPT=$(ls -t results/s19_subpop_rn50_star/${DATASET}/erm/*_s${SEED}/model_best_wga.pt results/s19_subpop_rn50_star/${DATASET}/erm/*_s${SEED}/model.pt 2>/dev/null | head -1 || true)

echo "============================================================"
echo "  Two-stage BRIDGE chain: $DATASET (seed $SEED)"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  ERM*:   $ERM_CKPT"
echo "  Started: $(date)"
echo "============================================================"

if [ ! -f "$ERM_CKPT" ]; then
    echo "ERROR: ERM* checkpoint not found" >&2; exit 1
fi

# ── Step 1: s26f WITH attribute annotations ──────────────────────
echo ""
echo "=== STEP 1: s26f WITH attribute annotations ==="
echo "Started: $(date)"

for RS in 0.5 0.7 0.9; do
    echo "[chain] s26f rho=$RS val-balance=attribute seed=$SEED"
    python -u train_bridge.py \
        --dataset "$DATASET" --backbone "$BACKBONE" \
        --merged-ckpt "$ERM_CKPT" \
        --val-balance attribute --inference max \
        --role-strength "$RS" --seed "$SEED" \
        --epochs 20 --batch-size 256 \
        --lr 5e-4 --entropic-scale 30 --wd-weight 10
done

echo "=== STEP 1 COMPLETE: $(date) ==="

# ── Step 2: s26f WITHOUT attribute annotations ───────────────────
echo ""
echo "=== STEP 2: s26f WITHOUT attribute annotations ==="
echo "Started: $(date)"

for RS in 0.5 0.7 0.9; do
    echo "[chain] s26f rho=$RS --no-group-roles seed=$SEED"
    python -u train_bridge.py \
        --dataset "$DATASET" --backbone "$BACKBONE" \
        --merged-ckpt "$ERM_CKPT" \
        --val-balance attribute --inference max \
        --role-strength "$RS" --seed "$SEED" \
        --epochs 20 --batch-size 256 \
        --lr 5e-4 --entropic-scale 30 --wd-weight 10 \
        --no-group-roles
done

echo "=== STEP 2 COMPLETE: $(date) ==="
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE: $DATASET seed $SEED"
echo "  ERM*:   $ERM_CKPT"
echo "  Finished: $(date)"
echo "============================================================"
