## Project
Reproducing VL-JEPA (arXiv:2512.10942) for research lab under Prof Faisal Qureshi at Ontario Tech.

## Architecture
- X-encoder: visual encoder for context frames (frozen V-JEPA 2 or fallback vit)
- Y-encoder: frozen visual encoder for target frames
- Language encoder: LLaMA 3 transformer layers
- Predictor: predicts Y-encoder output from X-encoder + text
- Loss: prediction loss + anti-collapse regularization

## Stack
- PyTorch, Python 3.10, CUDA 12.1
- Training on Narval A100 40GB (SLURM)
- Smoke tests on RTX 3050 WSL locally

## Key decisions
- Writing original implementation, Soumya repo used as reference only
- Building component by component, testing each before moving on