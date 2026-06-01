#!/bin/bash
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --job-name=vljepa-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

module load python/3.10
module load cuda/12.2

source ~/vljepa_env/bin/activate
cd ~/VL-JEPA-Implement

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CHECKPOINT_DIR=/scratch/tarunm10/vljepa_checkpoints
python evaluate.py --batch_size 32 "$@"