#!/bin/bash
#SBATCH --account=def-fqureshi       # Compute Canada allocation account
#SBATCH --job-name=vljepa-train      # Name shown in squeue
#SBATCH --gres=gpu:a100:1            # Request one A100 40GB GPU
#SBATCH --cpus-per-task=4            # CPU cores for data loading workers
#SBATCH --mem=40G                    # Host RAM (mirrors GPU VRAM to avoid bottleneck)
#SBATCH --time=24:00:00              # Wall-clock limit (HH:MM:SS)
#SBATCH --output=logs/train_%j.out   # stdout; %j expands to SLURM job ID
#SBATCH --error=logs/train_%j.err    # stderr; kept separate for easier debugging

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load required system modules (Narval module stack)
module load python/3.10
module load cuda/12.2

# Activate project virtualenv
source ~/vljepa_env/bin/activate

# Move to project root on Lustre scratch (fast parallel filesystem)
cd /lustre06/project/6001346/tarunm10/VL-JEPA-Implement

# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

# Ensure output directories exist before the job writes to them
mkdir -p logs
mkdir -p checkpoints

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python train.py --batch_size 32 --epochs 10

echo "End: $(date)"
