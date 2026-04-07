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

| Component | File | Status | Role |
|---|---|---|---|
| X-Encoder | `models/visual_encoder.py` | ✅ Done | Encodes context frames; frozen V-JEPA 2 with ViT-L/16 timm fallback |
| Y-Encoder | `models/y_encoder.py` | ✅ Done | Frozen EmbeddingGemma-300M backbone + trainable Linear(hidden→1536) head; L2-normalised output |
| Predictor | `models/predictor.py` | ✅ Done | LLaMA 3.2-1B layers 8–15 (≈490M); bidirectional attention; fuses visual patches + text query; L2-normalised 1536-dim output |
| Language Encoder | `models/language_encoder.py` | ⬜ Todo | LLaMA 3 transformer layers for text conditioning |

**Loss** (`models/loss.py`): prediction loss + anti-collapse regularization. ⬜ Todo

**Training** (`scripts/train.py`): main training loop. ⬜ Todo

**Config** (`configs/base.yaml`): YAML-based model and training hyperparameters, loaded via PyYAML.

**Experiment tracking**: Weights & Biases (`wandb`).

## Key implementation details

- **VisualEncoder** (`visual_encoder.py`): V-JEPA 2 strips CLS token, returns `(B, F, num_patches, 1024)`. Timm fallback encodes frames independently.
- **YEncoder** (`y_encoder.py`): EmbeddingGemma-300M frozen; attention-mask-weighted mean pool; `Linear(hidden→1536, bias=False)`; projection head uses 5% of base LR.
- **Predictor** (`predictor.py`): Extracts `llama.model.layers[8:16]` + `embed_tokens` (frozen) + `norm` + `rotary_emb`. Visual tokens prepended to text tokens. Bidirectional mask: additive 4-D `(B,1,1,S)` with −inf on padding columns, no causal triangle. `attn_implementation="eager"` avoids SDPA `cache_position` requirement. `_pass_position_embeddings` flag probed at init for transformers ≥ 4.45 compatibility.

## Development approach

Build one component at a time and test it before moving on. Current branch: `predictor`.
