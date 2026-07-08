# VL-JEPA for Compositional Zero-Shot Learning — Project Context

## Project Overview

Reproducing VL-JEPA (arXiv:2512.10942) and repurposing it for Compositional Zero-Shot Learning (CZSL), under Prof. Faisal Qureshi at Ontario Tech University. Long-term goal is integration with labmate Alan's continual learning work toward a Compositional Continual Zero-Shot Learning (CCZSL) system, targeting a workshop paper around September, with a possible top-tier conference submission for the combined CCZSL work.

The original VL-JEPA paper used DataComp for large-scale image-text pretraining with a predictive (non-contrastive) objective. We swapped to MIT-States because CZSL evaluation requires structured attribute-object pair data, which DataComp does not provide.

Open research question: no published work uses a predictive JEPA-style architecture for CZSL. Every existing strong baseline (CGE, CompCos, CAPE, Troika) uses a contrastive backbone (typically CLIP). This is the genuine novelty angle, regardless of whether our numbers beat baselines.

**Status update (June 2026):** Met with Prof. Qureshi, who is happy with progress and assessed the project at roughly 60% of the way to a publishable paper. He gave direct architectural guidance superseding the open decision points listed below — see "Architecture Overhaul" section. The backbone-swap question (V-JEPA 2 → CLIP) and the CCZSL multi-session reformatting question were not addressed and remain open for a future meeting; they are not being pursued right now.

---

## Architecture Overhaul (Post-Meeting, June 2026)

Prof. Qureshi reviewed the project history (Run 1–7, including the Run 7 classification head failure) and gave specific direction, summarized from the meeting transcript:

**Diagnosis of the core problem:** V-JEPA is trained unsupervised via masked image patches and representation-space loss, and treats attribute-object pairs as atomic — it never learns to disentangle attributes (e.g. color) from objects. This is why the architecture struggles with unseen compositions: nothing in training forces attribute and object information apart. The single Run 7 classification-head attempt confirmed this from a different angle — heads memorized seen attribute-object co-occurrences rather than learning transferable primitives.

**Directed changes:**

1. **Freeze the visual backbone entirely** (already the case for V-JEPA 2) — confirmed correct given the dataset size. 34,562 training images is not enough to safely fine-tune a large pretrained backbone; doing so risks overfitting. Keep V-JEPA 2 frozen, train only small heads on top.

2. **Replace the single LLaMA 3.2-1B predictor with three lightweight predictor heads:**
   - **Attribute head** — predicts an embedding representing just the attribute (e.g. "red"), invariant to object
   - **Object head** — predicts an embedding representing just the object (e.g. "chair"), invariant to attribute
   - **Composition head** — combines attribute + object head outputs into a full composition embedding (e.g. "red chair")

   Each head has its own loss, computed individually, rather than one shared loss over a single predictor output. This is intended to force disentangled representations directly, rather than hoping a single large predictor learns disentanglement implicitly.

   Model choice for the heads (finalized after further discussion, superseding the original MiniLM suggestion): **custom lightweight bidirectional transformer encoder, built from scratch with random initialization**, NOT a pretrained language model. Reasoning: the attr/obj heads consume raw visual patch tokens, not text, so a sentence-encoder's language pretraining doesn't transfer; a causal decoder LM (LLaMA, Qwen) would need the same bidirectional-attention retrofit that was the structural mismatch being fixed in the first place. Since the predictor was always trained from data regardless of starting point (encoders were frozen, only the predictor learned), pretraining benefit was never being leveraged here in a meaningful way. Composition head: simple MLP (concat fusion of attr+obj embeddings → 3-layer MLP) so any HM improvement is attributable to the batching strategy, not a more expressive fusion mechanism.

   Final specs: attr/obj heads each are a 4-layer bidirectional transformer encoder, hidden_dim=512, 8 attention heads, FFN dim 2048, mean-pooled patch tokens, output projected to 1536 — ~13.92M params each. Composition head is a 3-layer MLP with concat fusion (3072→1536→1536→1536) — ~9.44M params. Total ~37.28M trainable params across all three heads, replacing the ~400M+ trainable params from LLaMA layers 8–15.

3. **Replace random batch construction with structured primitive batching** — this is the core novelty Prof. Qureshi flagged as critical, and follows standard CZSL literature protocol:
   - **Attribute batches**: same attribute held constant, varied objects (e.g. red apple, red car, red shirt) — forces the attr head to learn attribute identity independent of object
   - **Object batches**: same object held constant, varied attributes (e.g. red chair, wooden chair, broken chair) — forces the obj head to learn object identity independent of attribute
   - **Composition batches**: standard random mixed pairs — trains the composition head
   - Batches cycle in **strict 1:1:1 alternation** across the three types
   - Faisal was explicit that without correct batch sampling and adherence to standard CZSL train/test split protocols, results are not fairly comparable to published baselines (CGE, CompCos, etc.)

4. **Use Claude Code for implementation, with conceptual understanding required** — Faisal plans to clone the repo and use Claude AI assistance directly for code improvements, but stressed that Tarun must understand every change deeply rather than treating it as a black box. This mirrors the project's existing working style.

**Implementation status:**
- `PrimitiveBatchSampler` and `RoutingDataLoader` — **implemented and tested**, in `data/primitive_sampler.py`. Groups dataset samples by attribute and by object index, yields batches in strict attr→obj→comp cycling order, with mode-appropriate text routing at the collate level (attr-batches carry just the attribute word, obj-batches carry just the object word, comp-batches carry the full phrase). Distinct primitive keys per batch guaranteed.
- Three primitive heads — **implemented and tested**, in `models/primitive_heads.py`. Custom bidirectional transformer encoders for attr/obj; MLP for composition. Composition head explicitly runs attr/obj forward passes under `torch.no_grad()` so comp-batch gradients are isolated to the composition head only (verified by smoke test).
- Training loop rewrite — **complete**, in `train.py`. Routes each batch type to its corresponding head, computes InfoNCE independently per head per step. Per-type val loss tracking.
- v2r1 architecture branch (`v2r1-primitive-heads`) — **first full training run completed (June 2026, 10 epochs, ~5.5 hours on Narval A100)**. See "v2r1: First Run Results" section below.
- Calibration bias sweep — **implemented** in `evaluate.py`. Standard generalized CZSL protocol: γ swept over 20 values [-2.0, 2.0] on val set, best γ applied once to test set. `--phase` flag added so val/test runs are explicitly separated. `--gamma` flag added for fixed single-value application.

**Branch convention going forward:** v2r2 series uses one-change-per-branch discipline (v2r2r1, v2r2r2, etc.) so each result is attributable to exactly one architectural change. This is required for the ablation table in the paper.

**Why this wasn't caught earlier:** Run 6's post-mortem had already diagnosed that the auxiliary cosine loss failed because same-primitive pairs rarely collide in a random batch of 32, and noted this would require a dedicated sampling strategy to be effective. That dedicated sampling strategy is exactly what Prof. Qureshi is now directing. The diagnosis existed before the meeting; the team instead pursued Run 7's classification heads and the (still pending) calibration bias sweep rather than acting on the batching insight directly.

---

## v2r1: First Run Results (June 2026)

**Official result (first properly-protocolled result in the project):**
Seen 10.17% / Unseen 12.07% / HM 11.04% (γ=0.11 from val sweep, applied once to test, three-branch λc=1.0/λa=0.5/λo=0.5). AUC=0.0265.

This is the first result following the standard generalized CZSL calibration protocol. All prior results (Runs 1-7) used direct test-set eval without calibration and are not directly comparable.

**Configuration:** 10 epochs, batch_size=32, lr=5e-5, AdamW (weight_decay=0.05). Branch: `v2r1-primitive-heads`. Checkpoint `step_0012960.pt` corresponds to epoch 4 (best val loss).

**Training dynamics (per-epoch val loss):**

| Epoch | val_loss | attr  | obj   | comp  |
|-------|----------|-------|-------|-------|
| 1     | 1.7022   | 2.19  | 1.46  | 1.46  |
| 2     | 1.5573   | 2.09  | 1.29  | 1.29  |
| 3     | 1.4744   | 2.05  | 1.21  | 1.16  |
| **4** | **1.4368** | **2.01** | **1.17** | **1.14** |
| 5     | 1.4445   | 2.07  | 1.17  | 1.10  |
| 6     | 1.4615   | 2.13  | 1.15  | 1.10  |
| 7     | 1.4962   | 2.25  | 1.13  | 1.11  |
| 8     | 1.5188   | 2.29  | 1.14  | 1.12  |
| 9     | 1.5640   | 2.38  | 1.21  | 1.10  |
| 10    | 1.6296   | 2.47  | 1.25  | 1.17  |

Best val_loss of 1.4368 at epoch 4 — the lowest val_loss observed across the entire project's history. Overfitting boundary at epoch 3-4 is consistent with every prior run.

**Eval results (epoch 4 checkpoint, step_0012960.pt):**

| Protocol | λc | λa | λo | γ | Seen | Unseen | HM |
|----------|----|----|-----|---|------|--------|-----|
| Uncalibrated (test) | 1.0 | 0.5 | 0.5 | 0 | 28.52% | 3.17% | 5.70% |
| Composition-only (test) | 1.0 | 0.0 | 0.0 | 0 | 19.25% | 3.41% | 5.79% |
| Calibrated sweep (val) | 1.0 | 0.5 | 0.5 | 0.11 | 12.13% | 14.94% | 13.39% |
| **Calibrated official (test)** | **1.0** | **0.5** | **0.5** | **0.11** | **10.17%** | **12.07%** | **11.04%** |

γ=0.11 found on val sweep, applied once to test. AUC=0.0265 (test), 0.0354 (val).

**Key findings:**

1. **Calibration is the single biggest inference-time lever.** Uncalibrated HM 5.70% → calibrated 11.04% with zero retraining. The model learned meaningful representations; the seen/unseen imbalance was an inference artifact correctable by γ.

2. **Composition head is the weak link, not the attr head.** Composition-only scoring (19.25% seen) is worse than three-branch (28.52% seen), meaning the composition head cannot recover signal from upstream noisy attr embeddings. The attr head's high val loss (2.47 vs obj 1.17) reflects genuine attribute hardness, but the composition head's blindness to raw visual features is the deeper architectural problem.

3. **The composition head sees no raw visual information.** It only receives attr+obj embeddings — if those are noisy, it has no fallback. CPF (2025) and Troika (CVPR 2024) both demonstrate that giving the composition path direct visual features alongside primitive embeddings fixes exactly this failure mode.

4. **The attr head is genuinely harder.** Attributes (sliced, wet, old) are visually context-dependent in ways objects are not. CANet (CVPR 2023) and CPF show object-conditioning on the attribute head closes this gap. This is v2r2r2's change.

5. **AUC of 0.0265 is low — the seen/unseen curve is a cliff, not a curve.** One γ step (−0.11 → 0.11) drops seen from 31.72% to 10.97% while unseen jumps from 0% to 14.94%. A smooth curve requires richer, more discriminative embeddings. v2r2 architecture changes are expected to smooth this.

**v2r2 plan (one change per run):**
- v2r2r1: raw visual tokens into composition head (Troika/CPF precedent)
- v2r2r2: object-conditioning on attribute head via FiLM (CANet/CPF/COT precedent)
- v2r2r3: upweight attribute loss to 1.5:1.0:1.0 ratio
- v2r2r4: CSP-style soft prompts on Y-encoder text targets

---

## Architecture

**Visual Encoder:** V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), frozen. Outputs patch tokens of shape (B, F, num_patches, 1024) — no CLS token. Single MIT-States images are duplicated to 2 frames to satisfy V-JEPA 2's video input format.

**Text Encoder (Y-Encoder):** EmbeddingGemma-300M (`google/embeddinggemma-300m`). Backbone frozen, projection head (768→1536) trainable at 0.05x base learning rate.

**Predictor (v2r1 onward):** Three lightweight heads replacing the old LLaMA 3.2-1B predictor:
- AttributeHead: 4-layer bidirectional transformer encoder, hidden=512, 8 heads, FFN=2048, mean-pool → Linear(512→1536) → L2-norm. ~13.92M params.
- ObjectHead: identical architecture. ~13.92M params.
- CompositionHead (v2r2r1 onward): 3-layer MLP, concat(attr_embed, obj_embed, visual_vec) → Linear(4608→1536) → GELU → Linear(1536→1536) → GELU → Linear(1536→1536) → L2-norm. visual_vec = mean-pool(patch_tokens, dims F+P) → L2-norm → (B, 1024). Runs attr/obj heads under torch.no_grad() so comp-batch gradients are isolated.
- Total: ~37.28M trainable params.

**Loss:** Bidirectional symmetric InfoNCE with learnable temperature (init τ=0.07, clamped). Computed independently per head per step — attr-batches only update the attr head, obj-batches only update obj head, comp-batches only update composition head.

---

## Dataset

**MIT-States CZSL split** (ExplainableML/czsl compositional-split-natural):
- 34,562 train / 18,375 val / 23,306 test images
- 115 attributes, 245 objects, 1,962 total pairs
- Test split: 1,262 seen pairs, 700 unseen pairs
- Custom `MITStatesDataset`, `__getitem__` returns `(clip, text, attr_idx, obj_idx, pair_idx)`
- Index mapping consistent across all splits (built from union of all three split files)

---

## Evaluation Protocol

**Metrics:** Seen accuracy / Unseen accuracy / Harmonic Mean (HM) / AUC. HM is the primary metric; AUC captures performance across all calibration operating points.

**Scoring:** Three-branch λ-weighted cosine similarity against pre-computed Y-encoder embedding banks (attr-only, obj-only, full composition). Default λc=1.0, λa=0.5, λo=0.5.

**Calibration (standard generalized CZSL protocol):** γ swept over 20 values on val set, best γ applied once to test set. Formula: `final_score(c|x) = score(c|x) + γ·𝟙[c∈unseen]`. All results from v2r1 onward follow this protocol. Prior results (Runs 1-7) are uncalibrated direct test-set evals and are not directly comparable.

---

## Branches and Runs — Full History

### Run 1 — `main` — Label Query Baseline
- **Change:** Predictor query was the label text itself (e.g. "wrinkled shirt")
- **Result:** Seen 1.88% / Unseen 0.20% / HM 0.36% (uncalibrated)
- **Diagnosis:** Text→text shortcut. Val loss misleadingly low (~2.98). Model never looked at the image.

### Run 2 — `main` — Neutral Query
- **Change:** Query changed to fixed `"a photo of"` for every sample
- **Result:** Seen 78.62% / Unseen 3.55% / HM 6.80% (uncalibrated)
- **Diagnosis:** Single biggest improvement (19x HM). Forced model to actually use visual tokens.

### Run 3 — `run3-regularization` — Layer Freezing
- **Change:** Froze LLaMA layers 8–11, trained only 12–15
- **Result:** Seen 77.80% / Unseen 3.33% / HM 6.38% (uncalibrated)
- **Diagnosis:** Slightly worse than Run 2. Freezing starved the model of needed cross-modal capacity.

### Run 4 — `run4-disentangle` — Separate Attr/Obj Encoding
- **Change:** Split label into attr+obj, encode separately, sum+normalize: `target = normalize(Y(attr) + Y(obj))`
- **Result:** Seen 73.03% / Unseen 3.70% / HM 7.05% (uncalibrated)
- **Diagnosis:** First principled compositional change. Trades seen for unseen, net positive HM.

### Run 5 — `run4-disentangle` (continued) — Auxiliary Cosine Loss
- **Change:** Added aux loss pulling same-primitive embeddings together (0.1x weight)
- **Result:** Seen 69.73% / Unseen 4.18% / HM 7.89% (uncalibrated, three-branch eval)
- **Critical bug:** `.item()` detached aux loss from autograd — zero gradient flowed. Improvement over Run 4 is noise.
- **Note:** This was the best uncalibrated result in the project pre-overhaul. Not directly comparable to v2r1 calibrated result.

### Run 6 — `run6-three-branch-eval` — Fixed Aux Loss + Grad Clipping
- **Change:** Fixed `.item()` bug, added grad clipping, built three-branch eval
- **Result:** Seen 73.80% / Unseen 3.88% / HM 7.37% (uncalibrated, three-branch)
- **Diagnosis:** Aux loss still never fired — batch size 32 too small for primitive collisions. Three-branch eval built here is the foundation of current evaluate.py.

### Run 7 — `run7-dual-head-primitives` — Dual Primitive Classifiers (Hybrid)
- **Change:** Added attr_head (Linear 2048→115) and obj_head (Linear 2048→245) on LLaMA predictor; joint InfoNCE + cross-entropy loss
- **Best result:** Seen 45.29% / Unseen 4.22% / HM 7.72% (uncalibrated, cls_weight=0.25, embedding mode)
- **Conclusion:** Negative result. Classification heads memorized seen co-occurrences, collapsed unseen to ~2% in classifier mode. InfoNCE embedding space remains more robust for unseen generalization.

### v2r1 — `v2r1-primitive-heads` — Three Lightweight Primitive Heads
- **Change:** Replaced LLaMA predictor entirely with three custom bidirectional transformer heads + PrimitiveBatchSampler
- **Official result:** Seen 10.17% / Unseen 12.07% / HM 11.04% (calibrated, γ=0.11 from val, AUC=0.0265)
- **Status:** First properly-protocolled result. Architecture established, v2r2 series underway.

### v2r2r1 — `v2r2r1-composition-visual-grounding` — Visual Tokens into Composition Head
- **Change:** Gave composition head direct access to mean-pooled V-JEPA patch tokens alongside attr+obj embeddings. Input dim: 3072 → 4608 (concat of attr_embed 1536 + obj_embed 1536 + visual_vec 1024). Motivated by CPF (2025) and Troika (CVPR 2024) which both show composition-path visual grounding fixes the "noisy upstream embeddings → weak composition" failure mode.
- **Training dynamics (per-epoch val loss):**

| Epoch | val_loss | attr  | obj   | comp  |
|-------|----------|-------|-------|-------|
| 1     | 1.6937   | 2.18  | 1.45  | 1.45  |
| 2     | 1.5593   | 2.11  | 1.29  | 1.28  |
| 3     | 1.4671   | 2.05  | 1.19  | 1.16  |
| **4** | **1.4316** | **2.02** | **1.16** | **1.12** |
| 5     | 1.4393   | 2.08  | 1.16  | 1.07  |
| 6     | 1.4532   | 2.15  | 1.14  | 1.07  |
| 7     | 1.5002   | 2.26  | 1.14  | 1.10  |
| 8     | 1.5236   | 2.31  | 1.15  | 1.11  |
| 9     | 1.5591   | 2.36  | 1.20  | 1.12  |
| 10    | 1.6271   | 2.50  | 1.22  | 1.16  |

- **Official result:** Seen 10.77% / Unseen 12.75% / HM 11.68% (calibrated, γ=0.11 from val, AUC=0.0344)
- **Val sweep result:** Best HM 13.54% at γ=0.11 (Seen=12.62%, Unseen=14.60%), AUC=0.0344
- **Diagnosis:** Small but real improvement over v2r1 (+0.64% HM, +0.68% unseen, AUC +0.0079). Val loss improved marginally for comp head (1.1354 → 1.1227 at epoch 4). The change is directionally correct but not a dramatic fix — the bottleneck is not purely composition-head blindness. Attr head val loss pattern is essentially unchanged (2.01 → 2.02 at epoch 4), confirming attr head quality is the next lever to pull.

---

## Key Findings (Cross-Run)

1. **Val loss is misleading for CZSL** — Run 1 had lowest val loss, worst CZSL. v2r1 broke this pattern with val_loss 1.44 (below InfoNCE floor of 3.47), confirming actual learning for the first time.

2. **Neutral query was the single highest-leverage change** (Run 1→2, 19x HM). Removing the text shortcut was necessary before any other change could matter.

3. **Calibration is mandatory** — uncalibrated v2r1 (5.70% HM) vs calibrated v2r1 (11.04% HM). All results must use the standard γ-sweep protocol for valid comparison.

4. **Disentanglement through data (batching) + architecture (separate heads) is the right direction.** Each run that forced primitive separation improved unseen accuracy.

5. **Auxiliary losses and classification heads both failed.** Aux loss never fired at batch size 32; classification heads memorized seen pairs. InfoNCE embedding space is the correct objective.

6. **V-JEPA features don't disentangle attribute from object.** This is the fundamental challenge — the frozen backbone treats pairs atomically. The three-head architecture + structured batching is the best available workaround without fine-tuning.

7. **Composition head blindness was a bottleneck but not the only one.** v2r2r1 gave it raw visual tokens — small improvement (+0.64% HM) but not dramatic. Attr head quality (val loss persistently ~2x higher than obj) is the next lever.

---

## Infrastructure Issues Encountered and Resolved

### Compute environments
- **Narval** (narval.alliancecan.ca): A100 40GB GPUs, allocation `def-fqureshi`. Primary compute. Project path `/lustre06/project/6001346/tarunm10/VL-JEPA-Implement/`. Modules: `python/3.10`, `cuda/12.2`.
- **Fir** (fir.alliancecan.ca): H100 GPUs. Abandoned due to persistent CUDA initialization failures on most nodes (node-specific driver issue). Home filesystem only 48GB — must set `CHECKPOINT_DIR` to `/scratch/tarunm10/vljepa_checkpoints`.

### Recurring issues and fixes
- `CHECKPOINT_DIR` must be set to scratch on both clusters — home fills up fast with 3GB checkpoints.
- sbatch scripts need `"$@"` to forward argparse arguments — several lost debugging sessions from this missing.
- HuggingFace gated models need `hf auth login` on login node, then `TRANSFORMERS_OFFLINE=1` on compute nodes.
- Optimizer state dicts are tied to parameter group structure — resuming after adding new model parameters requires fresh start, not resume.
- Concurrent training jobs need separate `--ckpt_dir` per job to avoid checkpoint race conditions.

---

## Current Status (as of June 2026)

**Ablation table (all calibrated, γ=0.11 from val, applied to test):**

| Run | Change | Seen | Unseen | HM | AUC |
|-----|--------|------|--------|-----|-----|
| v2r1 | three heads + primitive batching | 10.17% | 12.07% | 11.04% | 0.0265 |
| v2r2r1 | + visual tokens into composition | 10.77% | 12.75% | 11.68% | 0.0344 |
| v2r2r2 | + object-conditioning on attr head | TBD | TBD | TBD | TBD |
| v2r2r3 | + attr loss upweighting 1.5:1.0:1.0 | TBD | TBD | TBD | TBD |
| v2r2r4 | + CSP soft prompts on Y-encoder | TBD | TBD | TBD | TBD |

**Official best result:** v2r2r1 calibrated → Seen 10.77% / Unseen 12.75% / HM 11.68% / AUC 0.0344 (γ=0.11 from val, three-branch λc=1.0/λa=0.5/λo=0.5, test set).

**Baselines to beat (CLIP-free tier, calibrated HM):** CompCos ~16%, COT ~25%, CPF ~26%. Current gap: ~1.4x below weakest CLIP-free baseline. Note: previously quoted baselines (CoT ~17%, CompCos ~25%, CAPE ~33%) appear to be seen-accuracy numbers, not HM — corrected here.

**Active work:** v2r2r2 — object-conditioning on attribute head via FiLM (CANet/CPF/COT precedent). This is the highest-priority remaining change given attr head val loss is persistently ~2x higher than obj across all runs.

**Open questions (pending Faisal input):**
1. Backbone swap (V-JEPA 2 → CLIP) — not being pursued, remains open
2. CCZSL multi-session reformatting with Alan — not being pursued, current focus is single-session CZSL

**Relevant prior lab work:** PromptCCZSL (IJCAI 2024 + Dec 2025 follow-up) — soft prompt learning on frozen CLIP for continual compositional ZSL. Three-branch inference scoring in that paper directly inspired this project's eval protocol. Most likely template for eventual CCZSL integration with Alan's work.