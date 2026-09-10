"""
CZSL evaluation for the vanilla CLIP+CSP ablation baseline on MIT-States.

Standalone counterpart to evaluate.py — duplicates only the logic needed to
score a train_vanilla.py checkpoint, so evaluate.py itself stays completely
untouched (it is wired to the per-head SoftPromptTextEncoder / v3-CLIP r2
checkpoint format, which the vanilla baseline does not produce).

Protocol (identical to evaluate.py, for direct comparability)
---------------------------------------------------------------
1. Pre-compute text embeddings for every (attr, obj) pair in the closed-world
   candidate set, in three banks: attribute-only, object-only, and full
   composition — all encoded through the SAME shared VanillaSoftPromptEncoder
   (there is no per-head prompt in this baseline; only the text differs).
2. For each test image, run CLIPEncoder's visual tower once, then the vanilla
   PrimitiveHeads (FrozenProjectionHead / PassThroughCompHead — zero learned
   params, just CLIP's own image projection) to produce attribute, object,
   and composition predictions. All three are numerically identical, since
   the vanilla baseline has no learned per-primitive differentiation on the
   image side — that is the entire point of this ablation.
3. Three-branch λ-weighted scoring against the three text banks, then the
   same γ-calibration sweep as evaluate.py.
4. Report Seen / Unseen / Harmonic Mean / AUC.

Checkpoint format expected (see train_vanilla.py::save_checkpoint):
    state["primitive_heads"]    — PrimitiveHeads(variant="vanilla") state_dict
    state["head_config"]        — {"variant": "vanilla"}
    state["soft_prompt"]        — VanillaSoftPromptEncoder state_dict
    state["soft_prompt_config"] — {"n_ctx": int}
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.mit_states import MITStates as MITStatesDataset, _load_pairs
from models.clip_encoder import CLIPEncoder
from models.primitive_heads import PrimitiveHeads
from models.soft_prompt_vanilla import VanillaSoftPromptEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_vanilla")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the vanilla CLIP+CSP baseline on MIT-States CZSL"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/latest.pt",
        help="path to a train_vanilla.py checkpoint (default: checkpoints/latest.pt)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_root",
        default="/scratch/tarunm10/datasets/release_dataset/images",
        help="root dir containing 'attr obj' image subdirectories",
    )
    parser.add_argument(
        "--split_root",
        default="/scratch/tarunm10/datasets/mit-states/compositional-split-natural",
        help="dir containing train/val/test_pairs.txt",
    )
    parser.add_argument(
        "--lambda_c", type=float, default=1.0,
        help="weight for composition branch scoring (default: 1.0)",
    )
    parser.add_argument(
        "--lambda_a", type=float, default=0.5,
        help="weight for attribute branch scoring (default: 0.5)",
    )
    parser.add_argument(
        "--lambda_o", type=float, default=0.5,
        help="weight for object branch scoring (default: 0.5)",
    )
    parser.add_argument(
        "--phase", choices=["train", "val", "test"], default="test",
        help="which dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--gamma_sweep", action=argparse.BooleanOptionalAction, default=True,
        help="sweep calibration bias γ over [-3, 3] and report best HM + AUC "
             "(disable with --no-gamma_sweep for single-pass eval)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="fixed calibration bias added to unseen pair scores at inference "
             "(ignored when --gamma_sweep is active)",
    )
    return parser.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(
    ckpt_path: Path, device: torch.device
) -> tuple[CLIPEncoder, PrimitiveHeads, VanillaSoftPromptEncoder]:
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    logger.info("Loading CLIPEncoder …")
    clip_encoder = CLIPEncoder.load_pretrained(device=device)
    clip_encoder.eval()

    logger.info("Building PrimitiveHeads …")
    head_config = state.get("head_config") or {"variant": "vanilla"}
    if head_config.get("variant") != "vanilla":
        raise ValueError(
            f"eval_vanilla.py expects a variant='vanilla' checkpoint, got "
            f"head_config={head_config!r}. Use evaluate.py for v3-CLIP checkpoints."
        )
    primitive_heads = PrimitiveHeads.build(device=device, **head_config)
    primitive_heads.load_state_dict(state["primitive_heads"])
    primitive_heads.eval()

    logger.info("Building VanillaSoftPromptEncoder …")
    n_ctx = state["soft_prompt_config"]["n_ctx"]
    soft_prompt = VanillaSoftPromptEncoder(clip_encoder.clip_model, n_ctx=n_ctx).to(device)
    soft_prompt.load_state_dict(state["soft_prompt"])
    soft_prompt.eval()

    return clip_encoder, primitive_heads, soft_prompt


# ── Pair embedding cache ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_all_pairs(
    soft_prompt: VanillaSoftPromptEncoder,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three embedding banks: attribute, object, and composition.

    Each is (num_pairs, 768) L2-normalised. All three banks are encoded
    through the SAME shared soft prompt (VanillaSoftPromptEncoder has no
    per-head context — that is exactly what this ablation removes); only the
    input text differs:
        attr bank : soft_prompt(attr)        — attribute word alone
        obj  bank : soft_prompt(obj)         — object word alone
        comp bank : soft_prompt("attr obj")  — full composition phrase
    """
    attrs   = [attr            for attr, obj in pairs]
    objs    = [obj             for attr, obj in pairs]
    phrases = [f"{attr} {obj}" for attr, obj in pairs]
    attr_chunks, obj_chunks, comp_chunks = [], [], []
    for i in range(0, len(attrs), batch_size):
        ae = soft_prompt(attrs[i   : i + batch_size])
        oe = soft_prompt(objs[i    : i + batch_size])
        ce = soft_prompt(phrases[i : i + batch_size])
        attr_chunks.append(ae.cpu())
        obj_chunks.append(oe.cpu())
        comp_chunks.append(ce.cpu())
    return (
        torch.cat(attr_chunks, dim=0),
        torch.cat(obj_chunks,  dim=0),
        torch.cat(comp_chunks, dim=0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Models ────────────────────────────────────────────────────────────────
    clip_encoder, primitive_heads, soft_prompt = load_models(
        Path(args.checkpoint), device
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    test_dataset = MITStatesDataset(
        transform=clip_encoder.preprocess,
        phase=args.phase,
        images_root=args.data_root,
        splits_root=args.split_root,
    )

    seen_pairs: set[tuple[str, str]] = set(
        _load_pairs(Path(args.split_root) / "train_pairs.txt")
    )
    phase_pairs: set[tuple[str, str]] = set(
        _load_pairs(Path(args.split_root) / f"{args.phase}_pairs.txt")
    )
    candidate_pairs = sorted(seen_pairs | phase_pairs)
    cand2idx = {p: i for i, p in enumerate(candidate_pairs)}

    _n_seen   = sum(1 for p in candidate_pairs if p in seen_pairs)
    _n_unseen = len(candidate_pairs) - _n_seen
    logger.info(
        "Candidate set: %d pairs (%d seen + %d unseen) for phase=%s",
        len(candidate_pairs), _n_seen, _n_unseen, args.phase,
    )
    _expected = {"test": 1662, "val": 1562}
    if args.phase in _expected and len(candidate_pairs) != _expected[args.phase]:
        raise RuntimeError(
            f"Candidate set size {len(candidate_pairs)} != expected "
            f"{_expected[args.phase]} for phase={args.phase}. "
            f"Verify split files at {args.split_root}."
        )

    logger.info("%s samples: %d", args.phase.capitalize(), len(test_dataset))

    # ── Pre-compute three embedding banks over the candidate set ──────────────
    logger.info("Encoding %d pair texts via vanilla soft-prompted CLIP text encoder …", len(candidate_pairs))
    attr_embeds_bank, obj_embeds_bank, comp_embeds_bank = encode_all_pairs(
        soft_prompt, candidate_pairs, args.batch_size
    )
    attr_embeds_bank = attr_embeds_bank.to(device)   # (num_candidates, 768)
    obj_embeds_bank  = obj_embeds_bank.to(device)
    comp_embeds_bank = comp_embeds_bank.to(device)

    logger.info(
        "Scoring weights: λc=%.2f  λa=%.2f  λo=%.2f",
        args.lambda_c, args.lambda_a, args.lambda_o,
    )

    seen_mask = torch.tensor(
        [p in seen_pairs for p in candidate_pairs],
        dtype=torch.bool, device=device,
    )                                        # (num_candidates,)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_scores:    list[torch.Tensor] = []   # each (B, num_candidates) on CPU
    all_pair_idxs: list[torch.Tensor] = []   # each (B,) on CPU — vocab-space

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    with torch.no_grad():
        for batch_idx, (images, _texts, _attr_idxs, _obj_idxs, pair_idxs) in enumerate(
            test_loader
        ):
            images    = images.to(device, non_blocking=True)  # (B, C, H, W)
            pair_idxs = pair_idxs.to(device)                  # (B,) vocab-space

            # Step 1 — Visual encoding (shared by all three "heads").
            patch_tokens = clip_encoder.get_visual_features(images)  # (B, P, 1024)

            # Step 2 — Per-primitive predictions. In the vanilla baseline
            # these are all identical (FrozenProjectionHead has no learned
            # differentiation), kept as three separate calls only to mirror
            # evaluate.py's scoring code exactly for comparability.
            attr_pred  = primitive_heads.forward_attribute(patch_tokens)  # (B, 768)
            obj_pred   = primitive_heads.forward_object(patch_tokens)     # (B, 768)
            visual_vec = F.normalize(patch_tokens.mean(dim=1), dim=-1)    # (B, 1024)
            comp_pred  = primitive_heads.compose(attr_pred, obj_pred, visual_vec)  # (B, 768)

            # Step 3 — Three-branch λ-weighted base scores against candidate set.
            sims = (
                args.lambda_c * (comp_pred @ comp_embeds_bank.T)
              + args.lambda_a * (attr_pred @ attr_embeds_bank.T)
              + args.lambda_o * (obj_pred  @ obj_embeds_bank.T)
            )                                                  # (B, num_candidates)

            all_scores.append(sims.cpu())
            all_pair_idxs.append(pair_idxs.cpu())

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    "  processed %d / %d batches",
                    batch_idx + 1, len(test_loader),
                )

    base_scores = torch.cat(all_scores,    dim=0)   # (N_test, num_candidates)
    gt_indices  = torch.cat(all_pair_idxs, dim=0)   # (N_test,) — vocab-space

    gt_cand_idx = torch.tensor(
        [cand2idx[test_dataset.pairs[i]] for i in gt_indices.tolist()],
        dtype=torch.long,
    )                                                # (N_test,) — candidate-space

    gt_is_seen = torch.tensor(
        [test_dataset.pairs[i] in seen_pairs for i in gt_indices.tolist()],
        dtype=torch.bool,
    )                                                # (N_test,)

    seen_mask_cpu = seen_mask.cpu()                  # (num_candidates,) — for γ offset

    # ── Helper: accuracy at a given score matrix ──────────────────────────────
    def _accuracy(scores: torch.Tensor) -> tuple[float, float]:
        if not args.gamma_sweep and args.gamma != 0.0:
            scores = scores + args.gamma * (~seen_mask_cpu).float().unsqueeze(0)
        preds   = scores.argmax(dim=1)               # (N_test,) — candidate-space
        correct = preds == gt_cand_idx               # (N_test,) — remapped GT
        s_acc = correct[gt_is_seen].float().mean().item()   if gt_is_seen.any()   else 0.0
        u_acc = correct[~gt_is_seen].float().mean().item()  if (~gt_is_seen).any() else 0.0
        return s_acc, u_acc

    def _hm(s: float, u: float) -> float:
        return 2 * s * u / (s + u) if (s + u) > 0 else 0.0

    # ── Single-pass results (γ = 0, original behavior) ───────────────────────
    seen_acc, unseen_acc = _accuracy(base_scores)
    hm = _hm(seen_acc, unseen_acc)

    if not args.gamma_sweep:
        print()
        print("=" * 52)
        print("  Vanilla CLIP+CSP  |  MIT-States CZSL Results")
        print("=" * 52)
        print(f"  Seen accuracy   :  {seen_acc * 100:6.2f}%")
        print(f"  Unseen accuracy :  {unseen_acc * 100:6.2f}%")
        print(f"  Harmonic mean   :  {hm * 100:6.2f}%")
        print(f"  λc={args.lambda_c}  λa={args.lambda_a}  λo={args.lambda_o}")
        print("=" * 52)
        print()
        return

    # ── Calibration γ sweep ───────────────────────────────────────────────────
    gammas = torch.linspace(-3.0, 3.0, 50).tolist()
    rows: list[tuple[float, float, float, float]] = []  # (γ, seen, unseen, hm)

    unseen_mask_cpu = ~seen_mask_cpu  # candidate columns to offset
    for gamma in gammas:
        scores = base_scores.clone()
        scores[:, unseen_mask_cpu] += gamma
        s, u = _accuracy(scores)
        rows.append((gamma, s, u, _hm(s, u)))

    best = max(rows, key=lambda r: r[3])

    pts = sorted(rows, key=lambda r: r[2])
    xs  = [r[2] for r in pts]   # unseen axis
    ys  = [r[1] for r in pts]   # seen axis
    auc = float(torch.trapezoid(torch.tensor(ys), torch.tensor(xs)).abs())

    # ── Print summary table ───────────────────────────────────────────────────
    W = 52
    print()
    print("=" * W)
    print("  Vanilla CLIP+CSP | MIT-States CZSL Calibration Sweep")
    print("=" * W)
    print(f"  {'γ':>6}   {'Seen':>8}   {'Unseen':>8}   {'HM':>8}")
    print("-" * W)
    for gamma, s, u, h in rows:
        marker = "  ← best" if (gamma, s, u, h) == best else ""
        print(f"  {gamma:>6.2f}   {s*100:>7.2f}%   {u*100:>7.2f}%   {h*100:>7.2f}%{marker}")
    print("=" * W)
    print(f"  AUC: {auc * 100:.2f}%")
    print(f"  Best HM: {best[3]*100:.2f}%  at γ={best[0]:.2f}")
    print(f"  (Seen={best[1]*100:.2f}%, Unseen={best[2]*100:.2f}%)")
    print(f"  λc={args.lambda_c}  λa={args.lambda_a}  λo={args.lambda_o}")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
