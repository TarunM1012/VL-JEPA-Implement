#!/bin/bash
#SBATCH --account=def-fqureshi
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:50:00
#SBATCH --job-name=vljepa-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

module load python/3.10
module load cuda/12.2
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

source ~/vljepa_env/bin/activate
cd ~/VL-JEPA-Implement

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CHECKPOINT_DIR=/scratch/tarunm10/vljepa_checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python evaluate.py --batch_size 32 "$@"