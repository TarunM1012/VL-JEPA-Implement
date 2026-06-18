#!/bin/bash
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=vljepa-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

module load python/3.10
module load cuda/12.2

source ~/vljepa_env/bin/activate
cd /lustre06/project/6001346/tarunm10/VL-JEPA-Implement

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python evaluate.py "$@" --batch_size 32