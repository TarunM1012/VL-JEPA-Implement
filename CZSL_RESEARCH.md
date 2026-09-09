# CZSL Research Notes — Pathway to Beat SOTA

Deep-research findings on Compositional Zero-Shot Learning (CZSL), current
MIT-States SOTA, and what it would take for this repo's architecture to gain
+4-5 HM points. Compiled 2026-09-09.

## 1. What CZSL is, precisely

Compositional Zero-Shot Learning: recognize unseen **state-object**
(attribute-object) compositions at test time, where every individual state
and object appeared in training, just never in that combination.
MIT-States: 53,753 images, 115 states, 245 objects, 1,962 valid compositions.

- **Closed-world**: test candidates restricted to the compositions that are
  valid/labeled in the dataset (seen ∪ unseen sets, a few hundred pairs).
- **Open-world**: test candidates = full Cartesian product of all states ×
  all objects, including implausible/infeasible pairs (e.g. "wet fire").
  Roughly halves every method's HM/AUC vs closed-world (e.g. CDS-CZSL:
  39.2 HM closed → 22.1 HM open).
- **Seen acc**: best accuracy on seen compositions with the seen/unseen
  calibration bias → +∞. **Unseen acc**: same with bias → -∞. Sweeping that
  bias scalar from -∞ to +∞ traces a seen-vs-unseen accuracy curve. **HM**
  = harmonic mean at the best trade-off point on that curve. **AUC** = area
  under the whole curve (the primary metric — calibration-independent).

This repo targets **closed-world** MIT-States.

## 2. Verified current SOTA, MIT-States closed-world HM

| Method (arXiv) | Seen | Unseen | HM | AUC | Year |
|---|---|---|---|---|---|
| CSP | 46.6 | 49.9 | 36.3 | 19.4 | 2023 |
| DFSP | 47.1-47.4 | 52.4-52.8 | 37.3-37.7 | 20.7-20.8 | 2023 |
| GIPCOL | 48.5 | 49.6 | 36.6 | 19.9 | 2023 |
| Troika (2303.15230) | 49.0 | 53.0 | 39.3 | 22.1 | 2023 |
| CDS-CZSL | 50.3 | 52.9 | 39.2 | 22.4 | 2024 |
| MSCI (2505.10289) | 50.2 | 53.4 | 39.9 | 22.8 | 2025 |
| RAPR | 50.0 | 53.3 | 39.2 | 22.5 | 2025 |
| **EVA (2506.20986)** | 51.2 | 55.0 | **41.0** | 24.0 | 2025 |
| **H²EM (2512.20029)** | 52.4 | 54.1 | **41.3** | **24.2** | 2025 |

The "41%" figure quoted earlier in this conversation is **correct** — not a
closed/open-world mix-up. Current SOTA is genuinely ~41.0-41.3 HM, from two
independent 2025 papers landing within 0.3 of each other. Open-world numbers
are a different, much lower regime (~20-22 HM).

Note: PRPC (2607.05911, an MLLM-reasoning approach) only hits 29.2 HM on
MIT-States — reasoning-over-tokens approaches currently underperform
embedding-based CLIP methods here, so that direction isn't worth chasing.

## 3. What the gains are actually made of

- **CSP → DFSP/Troika (+3 HM)**: soft/learnable prompts per primitive
  instead of hand-written templates; Troika adds three explicit branches
  (state, object, composition) with inference-time fusion
  `p(comp) + p(state)·p(object)`, plus **LoRA/adapters on the vision
  encoder**. Their own ablation: "completely adapting both encoders
  mutually promotes alignment." Text encoder stays frozen; vision encoder
  does not.
- **Troika → EVA (+1.7 HM, 2025)**: replace fixed adapters with a
  **mixture-of-experts LoRA adapter** (token-routed, K=2 experts active)
  plus **semantic-variant alignment** (align to the most relevant of
  several expert-produced feature variants, not one fixed point). Ablation:
  MoE-adapter alone contributes far more (+6.8 AUC on C-GQA) than the
  alignment trick alone (+1.7 AUC).
- **Troika → H²EM (+2.0 HM, 2025)**: keeps Troika's architecture almost
  as-is but (a) maps embeddings into **hyperbolic (Lorentz) space** — alone
  +1.1 HM per ablation (39.3→40.4) — and (b) adds a **taxonomic entailment
  loss** forcing composition embeddings to sit inside their parent
  state/object "cones" (+0.8 HM more, 40.4→41.2), plus a hard-negative-mining
  InfoNCE variant. Text encoder stays frozen; only prompts/projections adapt.

**Common thread across every method above 39 HM**: the backbone is never
fully hands-off (at minimum learnable prompts, usually vision-side LoRA
too), and the composition branch is trained *jointly* with the primitive
branches, not isolated from them.

## 4. Binding constraints in this repo, ranked

1. **`CLIPEncoder.get_visual_features` discards CLIP's `ln_post`+`proj`**,
   handing from-scratch heads raw pre-projection 1024-d tokens and asking
   them to relearn an image→CLIP-text-space alignment from MIT-States-scale
   data alone. No SOTA method above throws this away — every one either
   keeps CLIP's own projection as the shared space or lightly adapts it
   (LoRA/prompts). Single largest gap vs. the table above.
2. **No hierarchy/entailment structure between attr/obj/composition.** The
   comp head (`CompositionHead`) is a flat MLP with no loss term expressing
   "composition embedding must be consistent with its own attr and obj
   embeddings" beyond feeding them in as inputs — Troika's explicit
   `p(c)+p(s)p(o)` fusion and H²EM's entailment loss are exactly this, worth
   ~2-3 HM combined per the ablations above.
3. **Full `no_grad()` detachment of attr/obj heads from the composition
   path.** No published top method isolates gradients this strictly — they
   jointly train all branches. This is a self-imposed research-cleanliness
   choice in this repo (isolating the routing contribution for the paper's
   own thesis), not a SOTA-motivated one.
4. **InfoNCE-only training, no primitive classification loss.**
   CSP/Troika/H²EM all train primitive prediction as cross-entropy over a
   fixed attribute/object vocabulary in addition to (or instead of)
   contrastive loss — also what makes the standard bias-sweep AUC metric
   natural to compute.

## 5. Ranked pathway to +4-5 HM

**A. Stop discarding CLIP's projection; add lightweight adaptation instead
of full pre-projection bypass.** Highest expected impact, medium cost —
requires editing `models/clip_encoder.py`. Either feed heads CLIP's
post-projection space, or initialize a new projection stage from CLIP's
actual `ln_post`+`proj` weights instead of random init, and/or add a small
LoRA adapter on the visual tower per Troika/EVA. Directly targets
constraint #1, which every SOTA jump above 39 HM has in common.

**B. Add Troika-style score fusion + a Euclidean (non-hyperbolic)
entailment loss on top of the existing three heads.** High impact, low
cost — stays entirely inside `models/primitive_heads.py`/`models/loss.py`,
no backbone changes. At inference, combine `comp_head` output with attr/obj
head scores rather than using comp_head alone; add a margin loss pulling
composition embeddings toward a function of their attr/obj embeddings.
H²EM's own ablation shows the *geometry* change (Euclidean→hyperbolic) is
worth +1.1 HM and the *entailment loss* is worth +0.8 HM on top of Troika's
baseline — a meaningful fraction of that is likely reachable without the
full Lorentz-model machinery.

**C. Loosen the `no_grad()` isolation on the composition path** (e.g., small
LR multiplier instead of full detach) to match how every top method jointly
optimizes branches. Moderate impact, low cost, but this is a direct
reversal of the current paper's stated research framing (isolating the
routing contribution) — a scientific-scope decision, not just an
engineering one.

**D. Feasibility/calibration term.** Deprioritize for the 41% closed-world
target — this is what separates closed- from open-world scores, not what
separates 39 from 41 HM within closed-world.

## 6. Is any of this actually novel?

No — pathways A, B, and C above are reproductions of what Troika/EVA/H²EM
already published (vision-side LoRA, inference-time score fusion,
entailment loss respectively). Implementing them well might get this
architecture toward **parity** with current SOTA (~41%), not past it by
another 4-5 points on their own — that would mean catching up to existing
techniques, not adding something new.

Two things found that genuinely nobody's done:

1. **EVA × H²EM combination.** EVA and H²EM are contemporaneous 2025 papers
   that never compare against or combine with each other — stacking EVA's
   MoE-LoRA adapter with H²EM's hyperbolic entailment loss is unpublished.
   Gains from two papers rarely just add; could land anywhere from ~42% to
   no better than either alone.
2. **Per-primitive routing itself.** This repo's actual point of
   difference — separate per-primitive heads with independent InfoNCE
   losses per batch type ("routing") — is structurally unlike every SOTA
   method's shared-prompt-classification framework. No paper found tries
   this. Real novel angle, but unproven — could just as easily underperform
   the prompt-based approach as beat it.

**Open question for the paper's actual thesis**: is the goal (a) a
semi-novel combination of published tricks with plausible-but-uncertain
upside, or (b) leaning further into the routing idea itself as the novel
contribution and treating "beat SOTA" as a stretch goal rather than the
paper's core claim?
