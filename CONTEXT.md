## Project
Reproducing VL-JEPA (arXiv:2512.10942) for research lab under Prof Faisal Qureshi at Ontario Tech.

## Current trajectory
Implementing a three-head primitive predictor architecture on branch `v2r1-primitive-heads`.
The single LLaMA-3.2-1B predictor has been replaced with three lightweight from-scratch heads.
Goal: improve harmonic mean (HM) on MIT-States CZSL by disentangling attribute / object / composition predictions.

## Architecture
- X-encoder: frozen V-JEPA 2 (ViT-L/16 timm fallback); output (B, F, num_patches, 1024)
- Y-encoder: frozen EmbeddingGemma-300M backbone + trainable Linear(hidden→1536); L2-normalised (B, 1536)
- Primitive heads (models/primitive_heads.py):
  - AttributeHead: TransformerHead(visual → 1536), trained vs Y("attr word")
  - ObjectHead: TransformerHead(visual → 1536), trained vs Y("obj word")
  - CompositionHead: MLP(attr_embed + obj_embed → 1536), trained vs Y("attr obj phrase")
  - Composition head runs attr/obj heads under no_grad — comp-batch gradients never touch attr/obj weights
- Loss: bidirectional InfoNCE (models/loss.py), computed independently per head per batch type

## TransformerHead spec
- 4 encoder layers, hidden=512, 8 attention heads, FFN=2048 (4× hidden)
- Fully bidirectional (no causal mask) — fresh from-scratch encoder, not a retrofitted causal model
- Mean-pool over all patch tokens → Linear(512→1536) → L2-normalise
- ~13.9 M params per head; ~37 M total across all three heads
- Pre-LN (norm_first=True) for stable random-init training

## CompositionHead spec
- concat([attr_embed, obj_embed]) → MLP(3072→1536→1536) with GELU → L2-normalise
- ~9.4 M params; simple so any HM gain is attributable to routing, not richer fusion

## Routing (data/primitive_sampler.py)
- PrimitiveBatchSampler: guarantees distinct primitive keys per batch (no false negatives)
- RoutingDataLoader: interleaves attr/obj/comp streams round-robin; texts already reduced to mode-appropriate strings
- Each batch trains exactly one head; losses are not summed across heads

## Evaluation (evaluate.py)
- Three-branch scoring: attr_pred·attr_bank + obj_pred·obj_bank + comp_pred·comp_bank (λ-weighted)
- Composition bank = Y("attr obj") full phrase — NOT sum of attr+obj embeddings

## Stack
- PyTorch, Python 3.10, CUDA 12.1
- Training on Narval A100 40GB (SLURM)
- Smoke tests on RTX 3050 WSL locally

## Key decisions
- Writing original implementation; Soumya repo used as reference only
- No pretrained model for predictor heads — all three built from random init
- predictor.py removed; replaced by models/primitive_heads.py

## Update policy
Update this file after every significant change, fix, or architectural decision. Remove obsolete entries.
