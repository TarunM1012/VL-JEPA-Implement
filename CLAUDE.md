# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Original implementation of VL-JEPA ([arXiv:2512.10942](https://arxiv.org/abs/2512.10942)) for a research lab under Prof. Faisal Qureshi at Ontario Tech. The Soumya repo is used as reference only — code here is written from scratch.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10 and CUDA 12.1. Training targets Narval A100 40GB (SLURM). Smoke tests run locally on RTX 3050 WSL.

## Commands

```bash
# Run tests
pytest

# Run a single test file
pytest tests/test_visual_encoder.py

# Run training (once implemented)
python scripts/train.py
```

## Architecture

The model has four components that are built and tested independently:

| Component | File | Role |
|---|---|---|
| X-Encoder | `models/visual_encoder.py` | Encodes context frames; frozen V-JEPA 2 with ViT fallback |
| Y-Encoder | `models/visual_encoder.py` | Frozen encoder for target frames |
| Language Encoder | `models/language_encoder.py` | LLaMA 3 transformer layers for text conditioning |
| Predictor | `models/predictor.py` | Predicts Y-encoder output from X-encoder + language features |

**Loss** (`models/loss.py`): prediction loss + anti-collapse regularization.

**Training** (`scripts/train.py`): main training loop.

**Config** (`configs/base.yaml`): YAML-based model and training hyperparameters, loaded via PyYAML.

**Experiment tracking**: Weights & Biases (`wandb`).

## Development approach

Build one component at a time and test it before moving on. The current branch `visual-encoder` is implementing the visual encoder module.
