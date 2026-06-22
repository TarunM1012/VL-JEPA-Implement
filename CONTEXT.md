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
- Calibration bias sweep — **still pending**, identified as a free, not-yet-implemented fix for the seen-heavy/unseen-collapsed pattern observed in every run to date. Now more relevant than ever given v2r1's seen accuracy drop.

**Branch convention going forward:** following the project's established pattern (e.g. `run4-disentangle`, `run3-regularization`), this overhaul should live on its own feature branch off `main`, separate from the old predictor's code, since the LLaMA predictor is being replaced rather than modified.

**Why this wasn't caught earlier:** Run 6's post-mortem had already diagnosed that the auxiliary cosine loss failed because same-primitive pairs rarely collide in a random batch of 32, and noted this would require a dedicated sampling strategy to be effective. That dedicated sampling strategy is exactly what Prof. Qureshi is now directing. The diagnosis existed before the meeting; the team instead pursued Run 7's classification heads and the (still pending) calibration bias sweep rather than acting on the batching insight directly.

---

## v2r1: First Run Results (June 2026)

**Configuration:** 10 epochs, batch_size=32, lr=5e-5, AdamW (weight_decay=0.05). Branch: `v2r1-primitive-heads` (cut from `main` after Run 4 merge). Checkpoint `step_0012960.pt` corresponds to epoch 4 (best val loss).

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

**Best val_loss of 1.4368 at epoch 4 — the lowest val_loss observed across the entire project's history.** Every prior run (1-7) sat at or near the InfoNCE random floor (log(32) ≈ 3.47) on val. Overfitting boundary is at the same epoch 3-4 mark seen in every prior run, suggesting it's a property of the dataset/regime, not the architecture.

**Eval results (epoch 4 checkpoint):**

| λc | λa | λo | Seen | Unseen | HM |
|----|----|----|------|--------|-----|
| 1.0 | 0.5 | 0.5 | 28.52% | 3.17% | 5.70% |
| 1.0 | 0.0 | 0.0 (composition-only) | 19.25% | 3.41% | 5.79% |

**Key findings:**

1. **Result is LOWER than Run 5's 7.95% HM best.** v2r1 has not beaten the pre-overhaul baseline yet.

2. **Seen accuracy collapse is the main story.** 28.52% vs Run 5's 68.84% — over 40 percentage points lost on seen pairs. Unseen accuracy (3.17%) is actually comparable to prior runs, so the new architecture has not destroyed generalization, it has degraded seen-pair recognition.

3. **Composition-only scoring is WORSE for seen than three-branch.** 19.25% vs 28.52%. This means the noisy attr and obj heads are actually *helping* seen accuracy when included, despite the attr head's high val loss. The composition head alone cannot recover seen accuracy.

4. **The composition head is the weak link, not the attr head.** Initial diagnosis was that the underperforming attr head (val loss 2.47 vs obj's 1.17) was dragging down the three-branch score, but composition-only eval falsifies that — composition is even weaker than the three-branch combination.

5. **Mechanistic explanation:** The composition head only sees attr+obj embeddings, never raw visual tokens. If those upstream embeddings are noisy (which the attr head's high val loss indicates they are), the composition head has no way to recover signal that was never extracted in the first place. The bottleneck is feature quality flowing into composition, not the composition fusion itself.

6. **The attr head is genuinely harder.** Across all epochs, attr val loss stays ~0.8-1.3 higher than obj. Attributes in MIT-States (sliced, old, wet, cooked) are genuinely more visually ambiguous and context-dependent than objects (chair, apple). This is not a bug, it's a property of the task — but it means the attr head likely needs more capacity, more epochs, or a different learning rate than obj/comp.

**Open questions for v2r2:**

- Should attr head have a higher learning rate or more capacity than obj/comp?
- Should the composition head get raw visual tokens in addition to attr/obj embeddings, so it has access to information the upstream heads might be discarding?
- Should training run longer than 10 epochs, with an attr-head-specific LR schedule that doesn't follow the obj/comp overfitting curve?
- Calibration bias sweep on epoch 4 checkpoint may still recover meaningful HM without retraining — still unimplemented, still potentially free.

**Status assessment:** v2r1 is a clean architectural overhaul that has not yet beaten baseline HM, but has produced the lowest val loss in project history and a clearer picture of where the bottleneck actually lives (composition-head input quality, specifically attribute features). This is a usable diagnostic result, not a failure — it points toward concrete v2r2 changes.

---

## Architecture

**Visual Encoder:** V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), frozen. Outputs patch tokens of shape (B, F, num_patches, 1024) — no CLS token, already flattened across frames. Single MIT-States images are duplicated to 2 frames to satisfy V-JEPA 2's video input format.

**Text Encoder (Y-Encoder):** EmbeddingGemma-300M (`google/embeddinggemma-300m`). Backbone frozen, projection head (768→1536) trainable at 0.05x base learning rate.

**Predictor:** LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`), layers 8–15 extracted, bidirectional attention (not causal). Visual projection 1024→2048, output head 2048→1536. All 8 layers trainable. As of Run 7, also has two auxiliary classification heads: `attr_head` (Linear 2048→115) and `obj_head` (Linear 2048→245).

**Loss:** Bidirectional symmetric InfoNCE with learnable temperature (init τ=0.07, stored as log-inverse-temperature, clamped to [log(1/100), log(1/0.01)]). As of Run 7, optionally combined with cross-entropy classification loss on attr/obj heads, weighted by `cls_weight`.

**Key insight:** The architecture predicts in embedding space rather than pixel/token space — the predictor's job is to produce an embedding close to the correct text embedding, not to reconstruct an image.

---

## Dataset

**MIT-States CZSL split** (ExplainableML/czsl compositional-split-natural):
- 34,562 train images / 18,375 val images / 23,306 test images
- 115 attributes, 245 objects, 1,962 total attribute-object pairs (not all 115×245=28,175 combinations exist — only visually sensible/photographed combinations)
- Test split: 1,262 seen pairs (appeared in training), 700 unseen pairs (held out compositions of known primitives)
- Custom `MITStatesDataset` class (not ExplainableML's `CompositionDataset`), `__getitem__` returns `(clip, text, attr_idx, obj_idx, pair_idx)`

**Index mapping verified:** `train_dataset._attr2idx == test_dataset._attr2idx` and same for `_obj2idx`. This is guaranteed by the dataset code, which always loads all three split files (train/val/test pairs) and takes `sorted(set(...))` before building the index dicts, regardless of which phase is requested. Confirmed via assertion before trusting Run 7 results.

---

## Evaluation Protocol

**Metric:** Seen accuracy / Unseen accuracy / Harmonic Mean (HM). HM is the primary metric because it punishes sacrificing one for the other — a model cannot hide poor unseen performance behind high seen accuracy.

**Three eval scoring modes** (`--score_mode` flag in `evaluate.py`):

1. **`embedding`** (single or three-branch): Precompute Y-encoder embeddings for all 1962 pairs. Score each test image via cosine similarity between predicted embedding and pair embeddings, argmax wins.
   - *Single-branch* (Runs 1–4): only composition embeddings (`attr_embed + obj_embed`, normalized).
   - *Three-branch* (Run 5 onward): adds separate attr-only and obj-only embedding banks. Final score = `λc·(pred·comp) + λa·(pred·attr) + λo·(pred·obj)`. Inspired by the inference protocol in Prof. Qureshi's own PromptCCZSL paper. This is a free improvement requiring no retraining — same checkpoint, better eval, +0.4% HM.

2. **`classifier`** (Run 7 onward): Uses the dual classification heads directly. `score = log_softmax(attr_logits)[pair_attr_idx] + log_softmax(obj_logits)[pair_obj_idx]` for each of the 1962 pairs, argmax wins. Pure primitive classification, never looks at composition as a whole.

3. **`combined`** (Run 7 onward): Sums the embedding-based score and classifier-based score.

**λ sweep results** (on Run 5 checkpoint, three-branch embedding mode):

| λc | λa | λo | Seen | Unseen | HM |
|----|----|----|------|--------|-----|
| 1.0 | 0.3 | 0.3 | 70.27% | 4.07% | 7.69% |
| 1.0 | 0.5 | 0.5 | 69.73% | 4.18% | 7.89% |
| 1.0 | 1.0 | 1.0 | 68.84% | 4.22% | 7.95% |

Trend: more weight on primitive branches consistently trades seen for unseen and slightly improves HM.

---

## Branches and Runs — Full History

### Run 1 — `main` — Label Query Baseline
- **Change:** Predictor query was the label text itself (e.g. "wrinkled shirt")
- **Result:** Seen 1.88% / Unseen 0.20% / HM 0.36%
- **Diagnosis:** The Y-encoder already knows exactly what "wrinkled shirt" maps to as a fixed embedding. The predictor learned a shortcut — match text to text — without ever needing to look at the image. Val loss was misleadingly low (~2.98, the lowest of any run) because the model was solving an easy degenerate version of the task (text→text) rather than the real one (image→text).

### Run 2 — `main` — Neutral Query
- **Change:** Predictor query changed to fixed neutral string `"a photo of"` for every sample, regardless of label
- **Result:** Seen 78.62% / Unseen 3.55% / HM 6.80%
- **Diagnosis:** This is the single biggest improvement across all experiments (19x HM). Removing the text shortcut forces the predictor to actually extract information from the visual patch tokens, since the query no longer hints at the answer. Val loss baseline shifted to InfoNCE's true random floor (~3.47 = log(32) at batch size 32) and the model now does meaningfully better than chance on seen pairs.

### Run 3 — `run3-regularization` — Layer Freezing
- **Change:** Froze LLaMA layers 8–11, trained only layers 12–15 (half the predictor)
- **Result:** Seen 77.80% / Unseen 3.33% / HM 6.38%
- **Diagnosis:** Slightly worse than Run 2. Hypothesis was overfitting reduction via fewer trainable params, but layers 8–11 likely contain cross-modal representations that need to adapt to the visual domain; freezing them starves the model of needed capacity. Regularization via freezing alone is not the lever that matters here.

### Run 4 — `run4-disentangle` — Separate Attr/Obj Encoding
- **Change:** Instead of encoding the full label text as one embedding, split into attribute and object, encode each separately via the Y-encoder, then sum and L2-normalize: `target = normalize(Y_encoder(attr) + Y_encoder(obj))`
- **Result:** Seen 73.03% / Unseen 3.70% / HM 7.05%
- **Diagnosis:** First architecturally principled compositional change. Trades seen accuracy for unseen accuracy, net positive for HM. The hypothesis — that decomposing the target embedding makes the space more compositional, so unseen pairs (built from seen primitives) become reachable — is directionally validated by the improved unseen accuracy.

### Run 5 — `run4-disentangle` (continued) — Auxiliary Cosine Loss
- **Change:** Added an auxiliary loss term: for any two samples in a batch sharing the same attribute, pull their attribute embeddings together via cosine similarity; same for shared objects. Weighted 0.1x added to the InfoNCE loss. Intent was to enforce that the Y-encoder produces consistent primitive representations regardless of what they're composed with.
- **Result:** Seen 71.06% / Unseen 3.97% / HM 7.51% (single-branch eval); **69.73% / 4.18% / 7.89% with three-branch eval on the same checkpoint**
- **Critical bug found:** The aux loss implementation called `.item()` on the cosine similarity tensor before accumulating: `aux_loss += 1 - F.cosine_similarity(...).item()`. This converts the value to a Python float and detaches it from the autograd graph. **No gradient ever flowed through the aux loss.** It was purely decorative — logged a number but contributed zero training signal. The small HM improvement over Run 4 is most likely noise or an artifact of the (still broken) total loss scale shifting optimizer dynamics slightly, not genuine learning from the aux objective.
- **This is the best embedding-mode result across the entire project: 7.89% HM.**

### Run 6 — `run6-three-branch-eval` — Fixed Aux Loss + Grad Clipping
- **Changes:**
  1. Fixed the `.item()` bug — removed all `.item()` calls from the aux loss computation so gradients flow correctly into `attr_embeds`/`obj_embeds` and back into the Y-encoder.
  2. Added gradient clipping (`torch.nn.utils.clip_grad_norm_`, max_norm=1.0) as standard stabilization practice.
  3. Split logging to show `infonce` and `aux` losses separately instead of one combined number.
  4. Also built the three-branch eval protocol in `evaluate.py` during this branch (separate attr/obj/comp embedding banks, λ-weighted combination).
- **Result:** Seen 73.80% / Unseen 3.88% / HM 7.37% (three-branch eval)
- **Diagnosis:** Despite fixing the gradient bug, aux loss still logged exactly 0 (or -0.0000) on every single training step. Root cause: with 115 attributes and 245 objects spread across a batch size of 32, the probability of two samples in the same batch sharing an attribute or object is low enough that it apparently never happened across the observed steps. The aux loss mechanism, even when correctly implemented, never actually fires given this batch composition. Result is statistically indistinguishable from Run 5 — slightly worse than Run 5's three-branch eval number, within noise.

### Run 7 — `run7-hybrid-classification-heads` — Dual Primitive Classifiers (Hybrid)
- **Strategic context:** After six runs of incremental tweaks yielding ~0.75 percentage points of total HM improvement, a deliberate decision was made to attempt a larger architectural change rather than continue micro-tuning the loss function. Two options were considered: (a) drop the InfoNCE/Y-encoder objective entirely and train pure primitive classifiers (highest expected HM gain, but abandons the "VL-JEPA" predictive-embedding premise of the project), or (b) a hybrid that keeps InfoNCE and Y-encoder intact while adding classification heads as an auxiliary objective (preserves the project's core identity, more conservative gain). Hybrid was chosen to keep the work aligned with the lab's stated direction pending prof input on strategy.
- **Changes:**
  1. Added two linear classification heads branching off the predictor's pooled 2048-dim representation: `attr_head = Linear(2048, 115)`, `obj_head = Linear(2048, 245)`.
  2. Predictor forward pass now returns `(embedding, attr_logits, obj_logits)` instead of just the embedding.
  3. Training loss: `total_loss = infonce_loss + cls_weight * (cross_entropy(attr_logits, attr_idx) + cross_entropy(obj_logits, obj_idx))`. `cls_weight` is a new argparse argument.
  4. Removed the broken aux cosine loss block entirely — superseded by the classification heads.
  5. `evaluate.py` gained the `--score_mode` flag (`embedding` / `classifier` / `combined`) described above.
  6. Old checkpoints (pre-Run 7) load with `strict=False` and a warning, since they lack the new head weights — these heads initialize randomly if an old checkpoint is loaded, making `classifier`/`combined` modes meaningless on old checkpoints but `embedding` mode still valid and reproducible.
- **Training dynamics observed:** For the first time across all seven runs, validation loss decreased rather than increased: epoch 1 → 9.04, epoch 2 → 8.48, epoch 3 → 7.90 (best), epoch 4 → 8.05 (overfitting begins), epoch 5 → 8.69. The cls_weight=1.0 run showed both InfoNCE and classification loss decreasing together on train and val simultaneously through epoch 3, indicating genuine generalization rather than memorization, before overfitting set in at the same epoch-3/4 boundary observed in every other run regardless of cls_weight.
- **cls_weight ablation** (all evaluated at epoch-3 checkpoint, embedding mode):

  | cls_weight | Seen | Unseen | HM |
  |---|---|---|---|
  | 0.10 | 42.27% | 4.09% | 7.46% |
  | **0.25** | **45.29%** | **4.22%** | **7.72%** |
  | 0.50 | 48.32% | 3.99% | 7.38% |
  | 1.00 | 45.10% | 3.89% | 7.16% |

  Best embedding-mode HM at cls_weight=0.25, still below Run 5's 7.89%.

- **Classifier/combined mode results** (cls_weight=0.25 checkpoint):

  | Mode | Seen | Unseen | HM |
  |---|---|---|---|
  | embedding | 45.29% | 4.22% | 7.72% |
  | classifier | 69.26% | 1.93% | 3.76% |
  | combined | 68.99% | 1.94% | 3.78% |

- **Conclusion — negative result, but a clear and useful one:** The classification heads, regardless of weight, hurt rather than help. In classifier/combined mode, seen accuracy is strong (~69%) but unseen accuracy collapses to ~2%, worse than any other run in the entire project. The heads learned to recognize attribute-object combinations that co-occurred during training but failed to generalize attribute and object recognition independently of composition — i.e., they memorized "wrinkled → shirt/paper/bag" associations rather than learning a transferable concept of "wrinkled." This directly demonstrates that primitive classification alone, at least as implemented here, does not solve CZSL, and that the InfoNCE embedding space (imperfect as it is) remains the more robust signal for unseen generalization. **Run 5 + three-branch eval (7.89% HM) remains the best result in the project.**

---

## Key Findings (Cross-Run)

1. **Val loss is a fundamentally misleading metric for CZSL.** Run 1 had the lowest val loss (~2.98) of any run and the worst CZSL scores (0.36% HM), because it was optimizing an easy degenerate sub-task (text→text matching) rather than the real one. Every subsequent run trained on pure InfoNCE showed val loss climbing monotonically from epoch 1 regardless of architectural changes — this looked like overfitting but was actually the model sitting near the InfoNCE random floor (log(32) ≈ 3.47) the entire time, with no real learning signal distinguishing seen from unseen compositions. The only Run where val loss meaningfully dropped during training was Run 7 (with classification heads added), and even then, overfitting set in at the same epoch 3–4 boundary.

2. **Neutral query was the single highest-leverage change in the entire project** (Run 1→2, 0.36%→6.80% HM, ~19x). Removing the text-based shortcut from the predictor's query input was necessary before any other change could matter.

3. **Disentangling attr/obj into separate target embeddings (Run 4) and three-branch eval (Run 5 onward) both reliably trade seen accuracy for unseen accuracy and net-improve HM.** This is the correct direction for CZSL — sacrificing memorization of seen compositions in favor of primitive-level generalization. Every successful improvement in this project has followed this exact pattern.

4. **Auxiliary losses and classification heads, across two separate implementation attempts (Run 5/6 aux loss, Run 7 classification heads), have not produced a net positive result over the three-branch InfoNCE baseline.** The aux loss never functioned correctly (gradient bug, then never-firing due to batch composition). The classification heads functioned correctly but actively hurt unseen generalization by memorizing seen attribute-object co-occurrences rather than learning transferable primitives.

5. **The fundamental bottleneck appears to be that V-JEPA 2's visual features do not disentangle attribute and object information well enough for either contrastive or classification objectives to generalize strongly to unseen compositions.** Every approach that has been tried operates on top of the same frozen visual features; none has broken meaningfully past ~8% HM, while baselines using CLIP-style contrastively-pretrained visual features achieve 17–33% HM on the same benchmark.

6. **Three-branch eval (PromptCCZSL-inspired) is a free win** — same checkpoint, no retraining, +0.4 to +0.5 percentage points of HM just from scoring against attribute and object embeddings independently in addition to the composition embedding.

---

## Infrastructure Issues Encountered and Resolved

### Compute environments
- **Narval** (narval.alliancecan.ca): A100 40GB GPUs, allocation `def-fqureshi`. Reliable throughout the project. Project path `/lustre06/project/6001346/tarunm10/VL-JEPA-Implement/`. Modules: `python/3.10`, `cuda/12.2` (no `StdEnv/2023` needed — adding it broke CUDA initialization on Fir, see below).
- **Fir** (fir.alliancecan.ca): H100 GPUs. Persistent and ultimately not fully resolved CUDA initialization issues:
  - Home filesystem is only 48GB and entirely separate from project/scratch space (unlike Narval, where home is on the same lustre06 filesystem as project space). This caused multiple checkpoint-save failures (`torch.save` filling the 48GB quota) until `CHECKPOINT_DIR` was redirected to `/scratch/tarunm10/vljepa_checkpoints` via an environment variable read by `train.py`.
  - GPU allocation via `--gres=gpu:h100:1` succeeds (confirmed via `nvidia-smi` showing the H100), but `torch.cuda.is_available()` intermittently returns `False` with a "Can't initialize NVML" warning, even with matching CUDA module versions and correct driver. This appeared to be node-specific: job 42179443 (node `fc10409`) successfully used CUDA throughout a full Run 6 training run; later jobs landing on different nodes (`fc10514`, `fc10516`, `fc10520`) consistently failed to initialize CUDA and silently fell back to CPU execution, producing valid but much slower results.
  - One GPU node (`fc10516`) was confirmed hardware-faulty — `nvidia-smi` showed `ERR!` status across all fields.
  - Net outcome: Fir was abandoned as the primary compute environment for this project after extensive debugging; Narval A100s used for all later runs (Run 7 cls_weight sweep) due to reliability, despite being nominally slower hardware.

### Checkpoint and storage bugs
- `save_checkpoint()` writes both a numbered `step_NNNNNNN.pt` and a `latest.pt` (via temp-file-then-rename) on every save — roughly 9.5GB × 2 per checkpoint event. Running multiple training jobs concurrently with the same default `checkpoints/` directory caused jobs to race and clobber each other's temp files, producing `FileNotFoundError` on the rename step. Fixed by adding a `--ckpt_dir` argparse argument so concurrent cls_weight sweep runs each write to isolated subdirectories (`checkpoints/run7_cls0.1`, etc.).
- `CKPT_DIR` was originally a relative path (`Path("checkpoints")`), which resolves differently depending on the cluster's home/project filesystem layout. Fixed by reading from a `CHECKPOINT_DIR` environment variable with the relative path as fallback, set explicitly per-cluster in sbatch scripts.
- Resuming training from an old checkpoint after adding new model parameters (Run 7's classification heads) raised `ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group` — optimizer state dicts are tied to the exact parameter group structure at save time. Resolved by moving old checkpoints aside and starting Run 7 fresh rather than attempting to resume.

### Other recurring issues
- HuggingFace gated repos (`google/embeddinggemma-300m`, `meta-llama/Llama-3.2-1B`) require both accepting the license on the model page and `hf auth login` with a token before first download; subsequent runs need `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` since compute nodes have no internet access.
- sbatch scripts repeatedly needed `"$@"` appended to the python invocation to forward command-line arguments through to `train.py`/`evaluate.py` — several debugging sessions were lost to scripts that silently ignored passed arguments (e.g., λ weights, `--cls_weight`, `--checkpoint` path) because this was missing.
- A stray `EOF` and duplicate `sbatch` line accidentally left at the end of a copy-pasted eval script caused confusing self-resubmission behavior.

---

## Current Status and Open Questions (as of last update, June 2026)

**Best result to date (pre-overhaul architecture):** Run 5 checkpoint, three-branch eval, λc=1.0/λa=1.0/λo=1.0 → Seen 68.84% / Unseen 4.22% / HM 7.95%. This is the number the new three-head architecture needs to beat.

**Baselines still to beat:** CoT ~17%, CompCos ~25%, CAPE ~33% HM. Current best is roughly 2.1x below the weakest baseline.

**Decision points requiring Prof. Qureshi's input (status as of June 2026 meeting):**
1. ~~Whether to run a controlled backbone-swap experiment (V-JEPA 2 → CLIP visual features)~~ — **not addressed in the meeting; remains open, not currently being pursued.**
2. **Resolved.** Faisal directed a third option not originally listed here: neither the hybrid InfoNCE+classification approach nor a pure primitive classifier, but three small dedicated predictor heads (attr/obj/composition) combined with structured primitive batch sampling. See "Architecture Overhaul" section above.
3. ~~Whether to begin reformatting MIT-States into a multi-session CCZSL setup now~~ — **not addressed in the meeting; remains open, not currently being pursued.** Current focus is strengthening the single-session CZSL result with the new architecture first.

**Relevant prior lab work:** PromptCCZSL (vclab.ca, Prof. Qureshi's lab, IJCAI 2024 + December 2025 follow-up) tackles continual compositional zero-shot learning using soft prompt learning on a frozen CLIP backbone, with cross-session knowledge distillation, cosine anchor loss, orthogonal projection loss, and intra-session diversity loss to prevent forgetting across sessions. Its three-branch inference scoring protocol directly inspired this project's three-branch eval. This paper is the most likely template for how this project's CZSL work and Alan's continual learning work are intended to combine.