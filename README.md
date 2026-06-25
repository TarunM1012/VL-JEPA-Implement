# VL-JEPA for Compositional Zero-Shot Learning

**From-scratch implementation of VL-JEPA (arXiv:2512.10942) repurposed for Compositional Zero-Shot Learning (CZSL)**

Research project under Prof. Faisal Qureshi at Ontario Tech University. Long-term goal: integration with continual learning toward a Compositional Continual Zero-Shot Learning (CCZSL) system, targeting a workshop paper submission.

---

## What is this?

Standard VL-JEPA uses a predictive (non-contrastive) architecture for image-text pretraining. **No published work has applied a JEPA-style predictive architecture to CZSL** — every strong baseline (CGE, CompCos, CAPE, Troika) uses CLIP. This project explores whether a predictive objective can learn disentangled attribute-object representations competitive with contrastive methods.

CZSL asks: given training images of *red apple* and *wooden chair*, can a model recognize an *unseen* composition like *red chair*? The challenge is learning what "red" and "chair" mean independently, not memorizing their co-occurrences.

---

## Architecture

```
Image (MIT-States) ──► V-JEPA 2 ViT-L [frozen] ──► patch tokens (B, F, N, 1024)
                                                          │
                          ┌───────────────────────────────┤
                          │                               │
                    AttributeHead                    ObjectHead
                 (4L BiTransformer)             (4L BiTransformer)
                     ~13.9M params                  ~13.9M params
                          │                               │
                          └──────────► CompositionHead ◄──┘
                                       (3-layer MLP)
                                        ~9.4M params

Text ──► EmbeddingGemma-300M [frozen] ──► projection head [trainable] ──► 1536-d target
```

**Three predictor heads** replace the original LLaMA 3.2-1B predictor:
- **AttributeHead** — predicts attribute embeddings invariant to object (e.g., "red")
- **ObjectHead** — predicts object embeddings invariant to attribute (e.g., "chair")  
- **CompositionHead** — fuses attr+obj embeddings via MLP concat fusion (e.g., "red chair")

**Loss:** Bidirectional symmetric InfoNCE, computed *independently* per head per batch. Total ~37M trainable params (down from ~400M+ with LLaMA).

### Structured Primitive Batching

The key novelty: instead of random batches, a `PrimitiveBatchSampler` constructs:
- **Attribute batches** — same attribute, varied objects (forces attribute-invariant representations)
- **Object batches** — same object, varied attributes (forces object-invariant representations)
- **Composition batches** — standard mixed pairs

Batches cycle in strict 1:1:1 alternation. Distinct primitive keys per batch are guaranteed, eliminating false negatives in the InfoNCE loss.

---

## Results

**Best result (v2r1, calibrated):**

| Split | Seen | Unseen | HM | AUC |
|-------|------|--------|----|-----|
| Test  | 10.17% | 12.07% | **11.04%** | 0.0265 |

Calibration protocol: γ swept over 20 values on val set, best γ=0.11 applied once to test. This is the first result following the standard generalized CZSL protocol — all prior runs used uncalibrated direct test eval.

**Calibration impact:** Uncalibrated HM 5.70% → calibrated **11.04%** with zero retraining. The model learned meaningful representations; the seen/unseen imbalance was an inference artifact correctable by γ.

**Baselines (CLIP-free tier, calibrated HM):** CompCos ~16%, COT ~25%, CPF ~26%.

---

## Run History

Seven runs document the full development trajectory, each attributable to one change:

| Run | Change | HM (uncalibrated) | Key Finding |
|-----|--------|-------------------|-------------|
| Run 1 | Label text as query | 0.36% | Text→text shortcut; model never used image |
| Run 2 | Neutral `"a photo of"` query | 6.80% | **19× improvement** — single biggest change |
| Run 3 | Layer freezing (LLaMA 8–11) | 6.38% | Slightly worse; froze needed capacity |
| Run 4 | Split attr+obj encoding | 7.05% | First principled compositional change |
| Run 5 | Auxiliary cosine loss | 7.89% | Bug: `.item()` detached grad; noise |
| Run 6 | Fixed aux loss + grad clipping | 7.37% | Aux loss never fired at batch_size=32 |
| Run 7 | Dual classification heads | 7.72% | Memorized seen pairs; collapsed unseen |
| **v2r1** | Three primitive heads + structured batching | **11.04%** (calibrated) | First properly-protocolled result |

---

## Dataset

[MIT-States](https://web.mit.edu/phillipi/Public/states_and_transformations/index.html) compositional split (ExplainableML/czsl):
- 34,562 train / 18,375 val / 23,306 test images
- 115 attributes × 245 objects → 1,962 pairs (700 unseen at test)

---

## Project Structure

```
models/
  visual_encoder.py     # V-JEPA 2 wrapper with timm fallback
  y_encoder.py          # EmbeddingGemma-300M + projection head
  primitive_heads.py    # AttributeHead, ObjectHead, CompositionHead
  loss.py               # Bidirectional InfoNCE
data/
  primitive_sampler.py  # PrimitiveBatchSampler + RoutingDataLoader
train.py                # Three-head routing training loop
evaluate.py             # Generalized CZSL eval + γ calibration sweep
configs/base.yaml       # Model and training hyperparameters
```

---

## Setup

```bash
pip install -r requirements.txt
# Requires Python 3.10, CUDA 12.1
```

```bash
# Run tests
pytest

# Train
python train.py

# Evaluate (val sweep → test apply)
python evaluate.py --phase val   # find best γ
python evaluate.py --phase test --gamma 0.11
```

Training targets Narval A100 40GB (SLURM). Smoke tests run on RTX 3050.

---

## Roadmap (v2r2 series)

One architectural change per run for clean ablation:

- **v2r2r1** — raw visual tokens into composition head (Troika/CPF precedent)
- **v2r2r2** — object-conditioning on attribute head via FiLM (CANet/CPF precedent)
- **v2r2r3** — attribute loss upweighting (1.5 : 1.0 : 1.0)
- **v2r2r4** — CSP-style soft prompts on Y-encoder text targets

---

## References

- [VL-JEPA paper](https://arxiv.org/abs/2512.10942)
- [Troika (CVPR 2024)](https://arxiv.org/abs/2303.15230)
- [CPF (2025)](https://arxiv.org/abs/2503.12712)
- [CANet (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Learning_Conditional_Attributes_for_Compositional_Zero-Shot_Learning_CVPR_2023_paper.html)
- [PromptCCZSL (IJCAI 2024)](https://arxiv.org/abs/2405.01024)
