#!/bin/bash
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --job-name=vljepa-vanilla-eval
#SBATCH --output=logs/%x_%j/out.log
#SBATCH --error=logs/%x_%j/err.log

# Usage: sbatch eval_vanilla_narval.sh <checkpoint_path> [extra evaluate.py args]
#
# Runs the SAME evaluate.py used for v3-CLIP r1 (unmodified — three-branch
# scoring, gamma sweep, everything identical), but with VLJEPA_SOFT_PROMPT_CKPT
# set so its text-bank construction (encode_all_pairs -> get_text_features)
# routes through this checkpoint's trained soft prompt instead of plain
# zero-shot CLIP text encoding. See the hook in models/clip_encoder.py and
# the checkpoint format in train_vanilla.py.

module load python/3.10
module load cuda/12.2

source ~/vljepa_env/bin/activate
cd /lustre06/project/6001346/tarunm10/VL-JEPA-Implement

# Create log dir — SLURM needs it to exist before writing
mkdir -p logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}

CKPT_PATH="$1"
if [ -z "$CKPT_PATH" ]; then
    echo "Usage: sbatch eval_vanilla_narval.sh <checkpoint_path> [extra evaluate.py args]"
    exit 1
fi
shift

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLJEPA_SOFT_PROMPT_CKPT="$CKPT_PATH"

python evaluate.py --checkpoint "$CKPT_PATH" "$@" --batch_size 32
