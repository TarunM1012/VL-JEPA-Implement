# VL-JEPA for Compositional Zero-Shot Learning — Project Context

## Project Overview

Adapting the JEPA predictive (non-contrastive) training objective for Compositional Zero-Shot Learning (CZSL) on MIT-States, under Prof. Faisal Qureshi at Ontario Tech University. Long-term goal is integration with labmate Alan's continual learning work toward a Compositional Continual Zero-Shot Learning (CCZSL) system. Workshop paper target: September 2026. Possible top-tier conference submission for the combined CCZSL work.

**Core novelty claim:** No published work applies a JEPA-style predictive objective to CZSL. Every competitive baseline (Troika, CAILA, CLUSPRO, CAMS, EVA) uses a contrastive CLIP backbone. This is genuine novelty independent of benchmark numbers.

**Paper framing (September 2026):** The paper is a systematic empirical study. V-JEPA 2 rows in the ablation table are not failures — they isolate the backbone contribution and motivate the CLIP swap. The JEPA predictive objective (predict primitive embeddings in embedding space via InfoNCE, not standard CLIP contrastive alignment) is the contribution. CLIP is the right vehicle for it.

---

## Architecture Pivot (September 2026)

**Decision:** Swap visual backbone from V-JEPA 2 → CLIP ViT-L/14, and text encoder from EmbeddingGemma-300M → CLIP text encoder. Keep the JEPA predictive objective (per-head InfoNCE), three-head structure, primitive batch sampler, and three-branch inference scoring unchanged.

**Reasoning:**
- V-JEPA 2 was pretrained on masked spatiotemporal patch prediction — it never learned attribute/object semantics. The frozen features treat attribute-object pairs atomically, which is the root cause of the performance ceiling (~11-12% HM across all v2 runs).
- No published CZSL work uses V-JEPA as a backbone. The field has uniformly converged on CLIP ViT-L/14 for competitive results. The performance gap between ResNet and CLIP backbones is larger than any architectural innovation within either tier (survey finding).
- EmbeddingGemma-300M is mismatched with V-JEPA 2 features — both spaces were pretrained independently with no alignment. CLIP image + text encoders share a pretrained alignment space, making the prediction task substantially easier to learn from 34k images.
- The JEPA predictive objective is preserved: predicting CLIP text embeddings from CLIP visual tokens via per-head InfoNCE is a different training signal from standard CLIP contrastive fine-tuning. This is the novelty claim.

**V-JEPA 2 results are kept in the paper** as the ablation baseline that isolates backbone contribution. Without them, there is no experiment isolating what the backbone contributes vs. what the objective contributes.

**Target performance with CLIP backbone:** 39–42% closed-world HM on MIT-States. This puts the work between Troika (39.3%) and CAMS/EVA (41.0%) — competitive with 2024-2025 SOTA. Reaching 40%+ requires intermediate CLIP layer access (CAMS-style), not just final CLS token.

**Branch:** implemented on `v3-CLIP` (off main; plan originally named this `clip-backbone`). Dataset, sampler, evaluate.py, and three-branch scoring are backbone-agnostic and carried over largely unchanged (see status below for the one shared change — frame-axis removal).

**Implementation status (2026-09-09): backbone swap complete, smoke-tested, not yet trained.** All of `models/clip_encoder.py` (new), `models/primitive_heads.py`, `train.py`, `evaluate.py`, and `data/mit_states.py` are rewritten for CLIP; see "Current Architecture (CLIP backbone)" below for the resulting design. Verification performed this session:
- `tests/test_clip_smoke.py` (new): 10-sample forward pass, image → CLIP patch tokens → attr/obj/comp heads → 768-dim unit-norm embeddings, text → CLIP text features → 768-dim, InfoNCE loss computes cleanly. Run against real downloaded ViT-L/14 weights (skips gracefully if `clip` or the checkpoint isn't available).
- `models/clip_encoder.py`'s hand-rolled visual-tower forward pass (patch tokens from the last transformer block, pre-projection) verified **bit-exact** (max abs diff 0.0) against CLIP's own `encode_image`/`encode_text` on random inputs.
- Existing smoke tests (`primitive_heads.py`, `loss.py`, `primitive_sampler.py`) all still pass after the frame-axis removal.
- Added the missing `openai/CLIP` GitHub install to `requirements.txt` — it was absent, so `pip install -r requirements.txt` would not have installed it.

Not yet done: a real training run (explicitly out of scope for this pass — submitted via sbatch separately) and intermediate-layer (CAMS-style) access for the attr/obj heads, which the 40%+ HM target below still depends on.

---

## Previous Architecture (v2, V-JEPA 2 backbone — superseded, kept as ablation baseline)

**Visual Encoder:** Frozen V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`). Patch tokens shape (B, F, num_patches, 1024). Images duplicated to 2 frames for video format compliance.

**Text Encoder (Y-Encoder):** Frozen EmbeddingGemma-300M (`google/embeddinggemma-300m`) + trainable projection head (768→1536) at 0.05x base lr.

**Predictor — Three Primitive Heads:**
- AttributeHead: 4-layer bidirectional transformer, hidden=512, 8 heads, FFN=2048, mean-pool → Linear(512→1536) → L2-norm. ~13.92M params.
- ObjectHead: identical. ~13.92M params.
- CompositionHead (v2r2r1 onward): MLP taking concat(attr_embed 1536, obj_embed 1536, visual_vec 1024) → Linear(4608→1536) → GELU → Linear(1536→1536) → GELU → Linear(1536→1536) → L2-norm. ~9.44M params. Runs attr/obj heads under torch.no_grad().
- Total trainable: ~37.28M params.

**Loss:** Bidirectional symmetric InfoNCE per head, independently. Attr-batches update only attr head; obj-batches update only obj head; comp-batches update only composition head. Learnable temperature τ (init 0.07, clamped).

**Batch Sampler:** PrimitiveBatchSampler cycles attr→obj→comp batches 1:1:1. Attr-batches: fixed attribute, varied objects. Obj-batches: fixed object, varied attributes. Comp-batches: random mixed. Distinct primitive keys per batch guaranteed.

---

## Current Architecture (CLIP backbone, v3-CLIP branch — implemented, untrained)

**Visual Encoder** (`models/clip_encoder.py`, new): Frozen CLIP ViT-L/14 image encoder, loaded via the `openai/CLIP` package (`clip.load("ViT-L/14", download_root=...)`) — **not** HuggingFace's `CLIPModel`. `get_visual_features(images)` returns patch tokens from the final transformer block, CLS token dropped, **before** CLIP's own `ln_post`/projection — shape `(B, num_patches, 1024)` (256 patches at 224px). Manually re-implements the visual-tower forward pass (conv1 → patchify → prepend CLS → add positional embedding → `ln_pre` → transformer) to intercept pre-projection features; verified bit-exact against `clip_model.encode_image()`'s own CLS path.

**Text Encoder** (same module): Frozen CLIP text encoder, `get_text_features(texts)` = `clip_model.encode_text(clip.tokenize(texts))` — CLIP's own projected output, shape `(B, 768)`. **Note:** the actual ViT-L/14 projected embedding dim is **768**, not 512 (512 is the ViT-B/32 and ViT-B/16 dim) — confirmed from the loaded checkpoint's `text_projection.shape`. All downstream dims below use 768.

**Predictor — same three-head structure, updated dims:**
- AttributeHead / ObjectHead (`TransformerHead`): unchanged 4-layer bidirectional transformer, hidden=512, 8 heads, FFN=2048, mean-pool, ~13.53M params each. Input is now `(B, num_patches, 1024)` (no frame axis — see below); output `Linear(512→768)` → L2-norm.
- CompositionHead: MLP taking `concat(attr_embed 768, obj_embed 768, visual_vec 1024)` = 2560 → `Linear(2560→768)` → GELU → `Linear(768→768)` → GELU → `Linear(768→768)` → L2-norm, ~3.15M params. Still runs attr/obj heads under `torch.no_grad()`.
- Total trainable: ~30.2M params (down from ~37.3M in v2, since the shared embedding dim shrank 1536→768).

**Frame axis removed:** the `(B, F, C, H, W)` / `(B, F, P, D)` video-format axis — a V-JEPA 2 holdover, MIT-States images were duplicated across it — is gone entirely. `MITStates.__getitem__` returns a plain `(C, H, W)` image preprocessed with CLIP's own `preprocess` transform (injected via a required `transform` constructor arg, not built internally); `TransformerHead`/`CompositionHead` operate on `(B, P, D)` / `(B, D)` tensors directly; `train.py`/`evaluate.py` no longer flatten/unflatten a frame dim around `get_visual_features`.

**Loss:** Unchanged — bidirectional symmetric InfoNCE per head, learnable temperature τ. Dimension-agnostic (just cosine similarity + cross-entropy), so the 1536→768 shrink required no changes to `models/loss.py` itself.

**Batch Sampler:** Unchanged — `PrimitiveBatchSampler` is agnostic to image shape/dim, only groups by (attr, obj) string keys.

**Key architectural decision for 40%+ (not yet implemented):** Intermediate CLIP layer access on attr/obj heads (CAMS precedent — last M transformer blocks via cross-attention with learnable latent queries). Final-block patch tokens alone are what's implemented now; lower layers carrying more attribute-relevant local texture information is the next lever toward the 40%+ target.

**What stayed identical:** `PrimitiveBatchSampler`, `evaluate.py`'s three-branch scoring (λc=1.0/λa=0.5/λo=0.5) and γ-calibration protocol, the per-head InfoNCE training loop structure in `train.py`.

---

## Experimental Results — Full History

### Pre-overhaul runs (LLaMA predictor, uncalibrated, not directly comparable to v2 results)

| Run | Key Change | Seen | Unseen | HM | Notes |
|-----|-----------|------|--------|----|-------|
| Run 1 | Label query baseline | 1.88% | 0.20% | 0.36% | Text→text shortcut, degenerate |
| Run 2 | Neutral query "a photo of" | 78.62% | 3.55% | 6.80% | 19x improvement, single biggest lever |
| Run 3 | Freeze LLaMA layers 8–11 | 77.80% | 3.33% | 6.38% | Slightly worse than Run 2 |
| Run 4 | Separate attr/obj encoding | 73.03% | 3.70% | 7.05% | First principled compositional change |
| Run 5 | Aux cosine loss (buggy) | 69.73% | 4.18% | 7.89% | Aux loss gradient detached — noise |
| Run 6 | Fixed aux loss + grad clip | 73.80% | 3.88% | 7.37% | Aux loss never fired at batch_size=32 |
| Run 7 | Dual classification heads | 45.29% | 4.22% | 7.72% | Heads memorized seen pairs — negative result |

### v2 runs (three primitive heads + PrimitiveBatchSampler, calibrated, γ=0.11 from val)

| Run | Change | Seen | Unseen | HM | AUC |
|-----|--------|------|--------|----|-----|
| v2r1 | Three heads + primitive batching | 10.17% | 12.07% | 11.04% | 0.0265 |
| v2r2r1 | + visual tokens into composition head | 10.77% | 12.75% | 11.68% | 0.0344 |
| v2r2r3 | + attr loss upweighting 1.5× | 11.54% | 12.54% | 12.02% | 0.0348 |

**Best v2 result:** v2r2r1 HM 11.68% / AUC 0.0344 (test, calibrated).

Note: v2r2r2 (FiLM object-conditioning on attr head) was run but results not yet recorded here. v2r2r3 result included above per memory.

---

## Evaluation Protocol

**Setting:** Closed-world generalized CZSL. Test candidate set = 1,662 pairs (1,262 seen + 400 test-unseen). NOT the full 1,962-pair vocabulary — this was a critical bug fixed in evaluate.py (cand2idx remap).

**Metrics:** Seen accuracy / Unseen accuracy / Harmonic Mean (HM) / AUC. HM is primary. AUC reported as percentage to match literature conventions.

**Calibration:** γ swept over 20 values on val set → best γ applied once to test. Formula: `final_score(c|x) = score(c|x) + γ·𝟙[c∈unseen]`. All v2 results use γ=0.11 found on val.

**Scoring:** Three-branch λ-weighted cosine similarity: `λc·(pred·comp) + λa·(pred·attr) + λo·(pred·obj)`. Default λc=1.0, λa=0.5, λo=0.5.

**SLURM job isolation:** `--export=ALL,CHECKPOINT_DIR=checkpoints/vXrYrZ` per job to avoid checkpoint racing.

---

## Competitive Landscape (Closed-World MIT-States HM, CLIP ViT-L/14)

| Method | HM | AUC | Venue |
|--------|-----|-----|-------|
| Vanilla CLIP | 26.1% | 11.0% | ICML'21 |
| CSP | 36.3% | 19.4% | ICLR'23 |
| DFSP | 37.3% | 20.6% | CVPR'23 |
| Troika | 39.3% | 22.1% | CVPR'24 |
| CAILA | 39.9% | 23.4% | WACV'24 |
| CLUSPRO | 40.7% | 23.8% | ICLR'25 |
| EVA | 41.0% | 24.0% | arXiv Jun'25 |
| CAMS | 41.0% | 24.2% | arXiv Nov'25 |
| H²EM | 41.3% | 24.2% | arXiv Dec'25 |

**Target:** 39–42% HM. Realistic with CLIP backbone + intermediate layer access + primitive batching. Beating 40% requires intermediate layers (CAMS-style). Beating 42% would require LLM descriptions or retrieval augmentation — out of scope for this paper.

**Closed-world focus:** Open-world is out of scope. Our architecture (predictive objective + primitive heads) is not designed for feasibility filtering across 28,175 pairs. Closed-world is where the JEPA objective has the cleanest story.

**V-JEPA 2 baseline tier:** CompCos ~16% HM, CoT ~25% HM (these are non-CLIP methods). Our v2r2r1 at 11.68% is below this tier — confirming V-JEPA 2 backbone is the ceiling, not the objective.

---

## Key Findings (Cross-Run)

1. **Val loss is misleading for CZSL.** v2 architecture broke this pattern — val_loss 1.44 is below the InfoNCE floor of log(32)≈3.47, confirming actual primitive learning for the first time. Pre-overhaul runs never broke this floor.

2. **Calibration is mandatory and high-leverage.** Uncalibrated v2r1: 5.70% HM. Calibrated: 11.04% HM. Same checkpoint, zero retraining. All results must use γ-sweep protocol.

3. **Structured primitive batching + separate heads is the right direction.** Every run that forced primitive separation improved unseen accuracy. This is the core CZSL contribution.

4. **Auxiliary losses and classification heads both failed** (Runs 5/6/7). InfoNCE embedding space is the correct objective for unseen generalization.

5. **V-JEPA 2 features don't disentangle attribute from object.** This is the fundamental ceiling of the v2 architecture. Confirmed by: attr head val loss persistently ~2× higher than obj across all runs; performance plateau at 11-12% HM regardless of head design.

6. **Composition head blindness is real but not the only bottleneck.** v2r2r1 gave it raw visual tokens — +0.64% HM improvement, real but not dramatic. Attr head quality is the deeper issue.

7. **Three-branch inference scoring is a free win.** Same checkpoint, no retraining, consistent +0.4–0.5% HM vs composition-only scoring. Already standard in CAMS, Troika, PromptCCZSL.

---

## Infrastructure

### Compute
- **Narval** (narval.alliancecan.ca): Primary. A100 40GB, allocation `def-fqureshi`. Path: `/lustre06/project/6001346/tarunm10/VL-JEPA-Implement/`. Modules: `python/3.10`, `cuda/12.2`.
- **Fir** (fir.alliancecan.ca): Secondary, unreliable. H100 GPUs but node-specific CUDA init failures. Home 48GB — must set `CHECKPOINT_DIR=/scratch/tarunm10/vljepa_checkpoints`. Use only as overflow.

### Recurring gotchas
- sbatch scripts need `"$@"` to forward argparse args — several lost sessions from this.
- HuggingFace gated models: `hf auth login` on login node, then `TRANSFORMERS_OFFLINE=1` on compute nodes.
- Concurrent jobs need separate `--ckpt_dir` to avoid checkpoint race conditions (temp-file rename collision).
- Optimizer state dicts are tied to parameter group structure — fresh start required after adding new model components.
- Weight decay must NOT be applied to learnable temperature τ (fixed in v2).
- Val batches must NOT reshuffle each epoch (fixed in v2).

### Dataset
- MIT-States CZSL split: 34,562 train / 18,375 val / 23,306 test images. 115 attrs, 245 objects, 1,962 total pairs.
- Line counts verified on Narval: train 1262, val 600, test 800 (pair-level).
- Index mapping consistent across splits — built from union of all three split files, sorted.
- Transferred to Fir via Globus.

---

## Repo Structure

- `models/clip_encoder.py` — **New.** Frozen CLIP ViT-L/14 (openai/CLIP package), exposes `get_visual_features` (pre-projection patch tokens) and `get_text_features`.
- `models/visual_encoder.py`, `models/y_encoder.py` — v2 (V-JEPA 2 / EmbeddingGemma) encoders. No longer imported by `train.py`/`evaluate.py`; kept for the ablation baseline history in this file, not for reuse.
- `data/mit_states.py` — MITStates dataset. Now takes a required `transform` arg (pass `CLIPEncoder.preprocess`); returns `(C, H, W)` images, no frame axis.
- `data/primitive_sampler.py` — PrimitiveBatchSampler + RoutingDataLoader (backbone-agnostic, unchanged by the CLIP swap beyond a `clips`→`images` naming cleanup).
- `models/primitive_heads.py` — Three heads, updated to 768-dim / `(B, P, D)` inputs for CLIP.
- `models/loss.py` — InfoNCELoss. Dimension-agnostic, untouched by the swap.
- `train.py` — Training loop with per-head routing, rewired to `CLIPEncoder`.
- `evaluate.py` — Three-branch scoring + γ-calibration, rewired to `CLIPEncoder`.
- `tests/test_clip_smoke.py` — **New.** 10-sample end-to-end shape/loss smoke test for the CLIP pipeline.
- `CONTEXT.md` — This file. Updated September 2026.
- `CLAUDE.md` — Static repo structure. Do not use for current project state.

**Branch convention:** One change per branch (v2r2r1, v2r2r2, ...). CLIP work: implemented on `v3-CLIP`.

---

## Lab Context

**Prof. Faisal Qureshi** — PI, Ontario Tech University Visual Computing Lab. Published PromptCCZSL (IJCAI 2024 + Dec 2025 update) and the first comprehensive CZSL survey (arXiv:2510.11106, Oct 2025). The survey's four-level taxonomy (no disentanglement → textual → visual → cross-modal) is the framework Faisal uses to situate this work. Current architecture targets cross-modal disentanglement tier.

 
**PromptCCZSL** (arXiv:2512.09172) — Faisal's lab paper. Frozen CLIP backbone, soft-prompt bank, session-aware multi-modal fusion, multi-teacher distillation, cosine anchor loss, orthogonal projection loss, intra-session diversity loss. Three-branch inference scoring (Eq. 4) directly inspired this project's eval protocol.