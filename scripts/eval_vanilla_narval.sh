#!/bin/bash
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --job-name=vljepa-vanilla-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch eval_vanilla_narval.sh <checkpoint_path> [extra eval_vanilla.py args]
#
# Runs eval_vanilla.py — a standalone script that duplicates evaluate.py's
# three-branch scoring + gamma-sweep protocol for the vanilla checkpoint
# format (single shared soft prompt, variant="vanilla" PrimitiveHeads).
# evaluate.py itself is untouched; see eval_vanilla.py's module docstring.

module load python/3.10
module load cuda/12.2

source ~/vljepa_env/bin/activate
cd /lustre06/project/6001346/tarunm10/VL-JEPA-Implement-clip-base

CKPT_PATH="$1"
if [ -z "$CKPT_PATH" ]; then
    echo "Usage: sbatch eval_vanilla_narval.sh <checkpoint_path> [extra evaluate.py args]"
    exit 1
fi
shift

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python eval_vanilla.py --checkpoint "$CKPT_PATH" "$@" --batch_size 32
