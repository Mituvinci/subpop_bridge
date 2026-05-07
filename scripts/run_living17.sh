#!/usr/bin/env bash
# Three-stage BRIDGE chain for Living17 (seed 0)
# AG dataset -- no group annotations, so NO Step 2 (with attr).
# Step 1: s20 merge (ERM* + 17 specialists, Optuna 50 trials)
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
export HF_HOME=/scratch/ha00014/hf_cache
export HF_HUB_CACHE=/scratch/ha00014/hf_cache/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /scratch/ha00014/Halimas_projects/Mosaic_Helath_AI
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

DATASET=living17
BACKBONE=resnet50
SEED=${SEED:-0}
N_TRIALS=400
N_GROUPS=17

ERM_CKPT=$(ls -t results/s19_subpop_rn50_star/${DATASET}/erm/*_s${SEED}/model_best_wga.pt results/s19_subpop_rn50_star/${DATASET}/erm/*_s${SEED}/model.pt 2>/dev/null | head -1 || true)
S19_ROOT=results/s19_subpop_rn50

echo "============================================================"
echo "  Three-stage BRIDGE chain: $DATASET (seed $SEED) [AG dataset]"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  ERM*:   $ERM_CKPT"
echo "  S19:    $S19_ROOT"
echo "  Trials: $N_TRIALS"
echo "  Started: $(date)"
echo "============================================================"

if [ ! -f "$ERM_CKPT" ]; then
    echo "ERROR: ERM* checkpoint not found" >&2; exit 1
fi
for g in $(seq 0 $((N_GROUPS-1))); do
    SPEC="$S19_ROOT/$DATASET/bridge_specialist/group_${g}"
    if [ ! -d "$SPEC" ]; then
        echo "ERROR: Specialist group_${g} not found: $SPEC" >&2; exit 1
    fi
done
echo "[chain] All $N_GROUPS specialist dirs verified."

# ── Step 1: s20 Optuna merge ─────────────────────────────────────
echo ""
echo "=== STEP 1: s20 merge (Optuna $N_TRIALS trials) ==="
echo "Started: $(date)"

python -u experiments/s20_subpop_bridge_merge_vit_erm.py \
    --dataset "$DATASET" \
    --backbone "$BACKBONE" \
    --s19-root "$S19_ROOT" \
    --erm-path "$ERM_CKPT" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --batch-size 64

echo "=== STEP 1 COMPLETE: $(date) ==="

MERGE_DIR=$(ls -td results/s20_subpop_bridge_erm_${DATASET}/combined/20*_s${SEED} 2>/dev/null | head -1)
MERGED_CKPT="$MERGE_DIR/merged_model.pt"

if [ -z "$MERGE_DIR" ] || [ ! -f "$MERGED_CKPT" ]; then
    echo "ERROR: merged_model.pt not found after s20" >&2; exit 1
fi
echo "[chain] Merged checkpoint: $MERGED_CKPT"

# ── Step 2: s26f WITHOUT attribute annotations (AG dataset) ──────
echo ""
echo "=== STEP 2: s26f WITHOUT attribute annotations (AG) ==="
echo "Started: $(date)"

for RS in 0.5 0.7 0.9; do
    echo "[chain] s26f rho=$RS --no-group-roles seed=$SEED"
    python -u experiments/s26f_bridge_structured_heads.py \
        --dataset "$DATASET" --backbone "$BACKBONE" \
        --merged-ckpt "$MERGED_CKPT" \
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
echo "  Merged: $MERGED_CKPT"
echo "  Finished: $(date)"
echo "============================================================"
