"""
CZSL evaluation for VL-JEPA on MIT-States.

Protocol
--------
1. Pre-compute CLIP text embeddings for every (attr, obj) pair in the
   closed-world candidate set, in three banks: attribute-only Y("attr"),
   object-only Y("obj"), and full-composition Y("attr obj").
2. For each test image, run CLIPEncoder's visual tower once, then the three
   primitive heads to produce attribute, object, and composition predictions
   in the shared embedding space.  The composition prediction fuses the
   attr/obj head outputs through the composition head.
3. Three-branch scoring: each prediction is scored against its matching bank,
   then combined with λ weights into a per-pair score.
4. Report seen accuracy, unseen accuracy, and harmonic mean (HM).

Seen / unseen split: pairs listed in train_pairs.txt are "seen"; all other
vocabulary pairs (val + test only) are "unseen".

Candidate set (closed-world): train_pairs ∪ phase_pairs (NOT the full 1962-pair
vocab). For test: 1262 seen + 400 unseen = 1662 pairs. For val: 1262 seen +
300 unseen = 1562 pairs. This matches the standard published protocol.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VL-JEPA on MIT-States CZSL")
    parser.add_argument(
        "--checkpoint", default="checkpoints/latest.pt",
        help="path to the checkpoint file (default: checkpoints/latest.pt)",
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
        help="sweep calibration bias γ over [-2, 2] and report best HM + AUC "
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
) -> tuple[CLIPEncoder, PrimitiveHeads]:
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    logger.info("Loading CLIPEncoder …")
    clip_encoder = CLIPEncoder.load_pretrained(device=device)
    clip_encoder.eval()

    logger.info("Building PrimitiveHeads …")
    head_config = state.get("head_config") or {}
    primitive_heads = PrimitiveHeads.build(device=device, **head_config)
    primitive_heads.load_state_dict(state["primitive_heads"])
    primitive_heads.eval()

    return clip_encoder, primitive_heads


# ── Pair embedding cache ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_all_pairs(
    clip_encoder: CLIPEncoder,
    pairs: list[tuple[str, str]],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three embedding banks: attribute, object, and composition.

    Each is (num_pairs, 768) L2-normalised, matching the targets each head was
    trained against:
        attr bank : Y("attr")        — attribute word alone
        obj  bank : Y("obj")         — object word alone
        comp bank : Y("attr obj")    — the full composition phrase
    """
    attrs   = [attr            for attr, obj in pairs]
    objs    = [obj             for attr, obj in pairs]
    phrases = [f"{attr} {obj}" for attr, obj in pairs]
    attr_chunks, obj_chunks, comp_chunks = [], [], []
    for i in range(0, len(attrs), batch_size):
        ae = F.normalize(clip_encoder.get_text_features(attrs[i   : i + batch_size]), dim=-1)
        oe = F.normalize(clip_encoder.get_text_features(objs[i    : i + batch_size]), dim=-1)
        ce = F.normalize(clip_encoder.get_text_features(phrases[i : i + batch_size]), dim=-1)
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
    clip_encoder, primitive_heads = load_models(
        Path(args.checkpoint), device
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Images are preprocessed with CLIP's own transform so pixel statistics
    # match what the frozen visual tower expects.
    test_dataset = MITStatesDataset(
        transform=clip_encoder.preprocess,
        phase=args.phase,
        images_root=args.data_root,
        splits_root=args.split_root,
    )

    # Seen pairs = those that appear in the training split file.
    seen_pairs: set[tuple[str, str]] = set(
        _load_pairs(Path(args.split_root) / "train_pairs.txt")
    )

    # Closed-world candidate set = seen (train) pairs + THIS phase's pairs.
    # NOT the full 1962-pair vocab — the other phase's unseen pairs are
    # distractors that inflate difficulty and break comparability with
    # published numbers.
    # test → 1262 seen + 400 unseen = 1662 pairs
    # val  → 1262 seen + 300 unseen = 1562 pairs
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
    logger.info("Encoding %d pair texts via CLIP text encoder …", len(candidate_pairs))
    attr_embeds_bank, obj_embeds_bank, comp_embeds_bank = encode_all_pairs(
        clip_encoder, candidate_pairs, args.batch_size, device
    )
    attr_embeds_bank = attr_embeds_bank.to(device)   # (num_candidates, 768)
    obj_embeds_bank  = obj_embeds_bank.to(device)
    comp_embeds_bank = comp_embeds_bank.to(device)

    logger.info(
        "Scoring weights: λc=%.2f  λa=%.2f  λo=%.2f",
        args.lambda_c, args.lambda_a, args.lambda_o,
    )

    # Boolean mask: True for candidate pairs that are seen (in train split).
    seen_mask = torch.tensor(
        [p in seen_pairs for p in candidate_pairs],
        dtype=torch.bool, device=device,
    )                                        # (num_candidates,)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    # Collect per-image ground-truth indices and base scores so the
    # γ sweep can reuse them without re-running the visual encoder.
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

            # Step 1 — Visual encoding (shared by all three heads).
            patch_tokens = clip_encoder.get_visual_features(images)  # (B, P, 1024)

            # Step 2 — Per-primitive predictions in the shared embedding space.
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

    # (N_test, num_candidates) base score matrix — built once, reused for γ sweep.
    base_scores = torch.cat(all_scores,    dim=0)   # (N_test, num_candidates)
    gt_indices  = torch.cat(all_pair_idxs, dim=0)   # (N_test,) — vocab-space

    # Remap ground-truth indices from full vocab-space into candidate-set space.
    # pair_idxs from the dataset indexes into the full 1962-pair vocab;
    # our score matrix is num_candidates wide, so we must remap before argmax.
    gt_cand_idx = torch.tensor(
        [cand2idx[test_dataset.pairs[i]] for i in gt_indices.tolist()],
        dtype=torch.long,
    )                                                # (N_test,) — candidate-space

    # Boolean GT membership: True when the ground-truth pair is seen.
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
        print("  VL-JEPA  |  MIT-States CZSL Results")
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

    # AUC: trapezoidal area under the seen-vs-unseen curve, sorted by unseen_acc.
    pts = sorted(rows, key=lambda r: r[2])
    xs  = [r[2] for r in pts]   # unseen axis
    ys  = [r[1] for r in pts]   # seen axis
    auc = float(torch.trapezoid(torch.tensor(ys), torch.tensor(xs)).abs())

    # ── Print summary table ───────────────────────────────────────────────────
    W = 52
    print()
    print("=" * W)
    print("  VL-JEPA | MIT-States CZSL Calibration Sweep")
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