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
| Primitive Heads | `models/primitive_heads.py` | ✅ Done | Three from-scratch bidirectional encoder heads: AttributeHead, ObjectHead (each ~14M params), CompositionHead MLP; routed by `data/primitive_sampler.py` |
| Language Encoder | `models/language_encoder.py` | ⬜ Todo | (reserved; not used in current approach) |

**Loss** (`models/loss.py`): bidirectional InfoNCE, computed independently per head per batch type. ✅ Done

**Training** (`train.py`): three-head routing loop via `RoutingDataLoader`. ✅ Done

**Data routing** (`data/primitive_sampler.py`): `PrimitiveBatchSampler` + `RoutingDataLoader`; guarantees distinct primitive keys per batch, interleaves attr/obj/comp streams round-robin. ✅ Done

**Config** (`configs/base.yaml`): YAML-based model and training hyperparameters, loaded via PyYAML.

**Experiment tracking**: Weights & Biases (`wandb`).

## Key implementation details

- **VisualEncoder** (`visual_encoder.py`): V-JEPA 2 strips CLS token, returns `(B, F, num_patches, 1024)`. Timm fallback encodes frames independently.
- **YEncoder** (`y_encoder.py`): EmbeddingGemma-300M frozen; attention-mask-weighted mean pool; `Linear(hidden→1536, bias=False)`; projection head uses 5% of base LR.
- **PrimitiveHeads** (`primitive_heads.py`): Three heads — two `TransformerHead` (attr/obj, ~14M params each, hidden=512, 4 layers, 8 heads, bidirectional, mean-pool, pre-LN) + one `CompositionHead` (MLP, concat fusion, ~9.4M params). Composition head runs attr/obj heads under `no_grad` so comp-batch gradients never touch attr/obj weights.
- **RoutingDataLoader** (`data/primitive_sampler.py`): `PrimitiveBatchSampler` guarantees distinct primitive keys per batch (no false negatives in InfoNCE). Texts returned are already mode-appropriate (attr-word / obj-word / full phrase).

## Development approach

Build one component at a time and test it before moving on. Current branch: `v2r1-primitive-heads`.
Update CONTEXT.md after every significant change or fix.
