#!/bin/bash
#SBATCH --account=def-fqureshi
#SBATCH --job-name=vljepa-valsweep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/valsweep_%j.out
#SBATCH --error=logs/valsweep_%j.err

# Usage: sbatch scripts/eval_val_sweep_narval.sh <checkpoint_dir>
#
# Evaluates every end-of-epoch checkpoint (step divisible by 2844 = one epoch
# at batch 32 on the 30,338-image train split) on the VAL split, and prints a
# summary table at the end. Pick the checkpoint with the best val AUC, then run
# test ONCE on that checkpoint only.

module load python/3.10
module load cuda/12.2
source ~/vljepa_env/bin/activate
cd /lustre06/project/6001346/tarunm10/VL-JEPA-Implement

CKPT_DIR="$1"
STEPS_PER_EPOCH=2844
if [ -z "$CKPT_DIR" ]; then
    echo "Usage: sbatch scripts/eval_val_sweep_narval.sh <checkpoint_dir>"
    exit 1
fi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SUMMARY=""
for CKPT in $(ls "$CKPT_DIR"/step_*.pt | sort); do
    STEP=$((10#$(basename "$CKPT" .pt | sed 's/step_//')))
    if [ $((STEP % STEPS_PER_EPOCH)) -ne 0 ]; then
        continue
    fi
    EPOCH=$((STEP / STEPS_PER_EPOCH))
    echo "=============== epoch $EPOCH  ($CKPT) ==============="
    OUT=$(python evaluate.py --checkpoint "$CKPT" --phase val --batch_size 32 2>&1)
    echo "$OUT"
    HM=$(echo "$OUT"  | grep "Best HM"     | awk '{print $4}')
    AUC=$(echo "$OUT" | grep "AUC"         | tail -1 | awk '{print $3}')
    S=$(echo "$OUT"   | grep "Best seen"   | awk '{print $4}')
    U=$(echo "$OUT"   | grep "Best unseen" | awk '{print $4}')
    SUMMARY+=$(printf "epoch %2d  step %6d   seen %8s  unseen %8s  HM %8s  AUC %6s" \
               "$EPOCH" "$STEP" "$S" "$U" "$HM" "$AUC")$'\n'
done

echo
echo "================ VAL SUMMARY ================"
printf "%s" "$SUMMARY"
echo "Pick the epoch with the highest val AUC, then run test once on it."