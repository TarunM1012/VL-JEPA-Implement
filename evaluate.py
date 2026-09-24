"""
CZSL evaluation for VL-JEPA on MIT-States.

Protocol
--------
1. Pre-compute CLIP text embeddings for every (attr, obj) pair in the
   closed-world candidate set, in three banks: attribute-only Y("attr"),
   object-only Y("obj"), and full-composition Y("attr obj").
2. For each test image, run CLIPEncoder's visual tower once, then the three
   primitive heads to produce attribute, object, and composition predictions.
3. Three-branch scoring: each prediction is scored against its matching bank,
   then combined with λ weights into a per-pair score.
4. Calibration: a bias γ is added to every unseen candidate's score. Instead of
   sampling γ on a fixed grid, the exact seen/unseen accuracy curve is computed
   (see `exact_calibration_curve`). Reports best seen, best unseen, best HM and
   AUC, following the standard CZSL protocol (Purushwalkam et al., 2019;
   Naeem et al., 2021).

Seen / unseen split: pairs listed in train_pairs.txt are "seen"; all other
candidate pairs are "unseen".

Candidate set (closed-world): train_pairs ∪ phase_pairs. For test:
1262 seen + 400 unseen = 1662 pairs. For val: 1262 seen + 300 unseen = 1562.
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
    parser.add_argument("--lambda_c", type=float, default=1.0,
                        help="weight for composition branch scoring (default: 1.0)")
    parser.add_argument("--lambda_a", type=float, default=0.5,
                        help="weight for attribute branch scoring (default: 0.5)")
    parser.add_argument("--lambda_o", type=float, default=0.5,
                        help="weight for object branch scoring (default: 0.5)")
    parser.add_argument(
        "--phase", choices=["train", "val", "test"], default="test",
        help="which dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--gamma_sweep", action=argparse.BooleanOptionalAction, default=True,
        help="compute the exact calibration curve and report best seen / unseen / "
             "HM and AUC (disable with --no-gamma_sweep for a single fixed-γ pass)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="fixed bias added to unseen pair scores (only used with --no-gamma_sweep)",
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
    """Return three L2-normalised embedding banks (num_pairs, 768):
        attr bank : Y("attr")      obj bank : Y("obj")      comp bank : Y("attr obj")
    """
    attrs   = [attr            for attr, obj in pairs]
    objs    = [obj             for attr, obj in pairs]
    phrases = [f"{attr} {obj}" for attr, obj in pairs]
    attr_chunks, obj_chunks, comp_chunks = [], [], []
    for i in range(0, len(attrs), batch_size):
        attr_chunks.append(F.normalize(clip_encoder.get_text_features(attrs[i : i + batch_size]), dim=-1).cpu())
        obj_chunks.append(F.normalize(clip_encoder.get_text_features(objs[i : i + batch_size]), dim=-1).cpu())
        comp_chunks.append(F.normalize(clip_encoder.get_text_features(phrases[i : i + batch_size]), dim=-1).cpu())
    return torch.cat(attr_chunks), torch.cat(obj_chunks), torch.cat(comp_chunks)


# ── Calibration curve ─────────────────────────────────────────────────────────

def exact_calibration_curve(
    scores: torch.Tensor,       # (N, C) base scores, candidate-space
    gt: torch.Tensor,           # (N,)   ground-truth candidate index
    seen_mask: torch.Tensor,    # (C,)   True for seen candidate pairs
    gt_is_seen: torch.Tensor,   # (N,)   True when the ground-truth pair is seen
) -> list[tuple[float, float, float]]:
    """
    Exact seen/unseen accuracy curve over every calibration bias.

    A bias b is added to every unseen candidate's score. Each image predicts
    its best SEEN pair until b exceeds (best_seen_score - best_unseen_score),
    then flips to its best UNSEEN pair. Accuracy only changes at these flip
    points, so sorting them yields every distinct operating point exactly —
    no grid, nothing missed between samples.

    Returns a list of (bias, seen_acc, unseen_acc), from bias = -inf upward.
    Each row holds for biases just above its bias value.
    """
    neg_inf = torch.finfo(scores.dtype).min
    s_max, s_idx = scores.masked_fill(~seen_mask, neg_inf).max(dim=1)   # best seen pair
    u_max, u_idx = scores.masked_fill(seen_mask, neg_inf).max(dim=1)    # best unseen pair
    thr = s_max - u_max                    # image flips to its unseen pick once bias > thr

    ok_seen = (s_idx == gt)                # correct while predicting best seen
    ok_unseen = (u_idx == gt)              # correct after flipping to best unseen
    n_s = gt_is_seen.sum().item()
    n_u = (~gt_is_seen).sum().item()

    # Starting point (bias = -inf): every image predicts its best seen pair.
    cs0 = (ok_seen & gt_is_seen).sum().item()
    cu0 = (ok_seen & ~gt_is_seen).sum().item()

    # Walk the flip points in order; each flip changes one image's correctness.
    order = torch.argsort(thr)
    delta = (ok_unseen.long() - ok_seen.long())[order]
    flip_is_seen = gt_is_seen[order]
    cs = cs0 + torch.cumsum(torch.where(flip_is_seen, delta, 0), dim=0)
    cu = cu0 + torch.cumsum(torch.where(~flip_is_seen, delta, 0), dim=0)

    curve = [(float("-inf"), cs0 / n_s, cu0 / n_u)]
    curve += list(zip(thr[order].tolist(), (cs / n_s).tolist(), (cu / n_u).tolist()))
    return curve


def _hm(s: float, u: float) -> float:
    return 2 * s * u / (s + u) if (s + u) > 0 else 0.0


def summarize_curve(curve: list[tuple[float, float, float]]) -> dict:
    """Standard CZSL metrics from the calibration curve."""
    best = max(curve, key=lambda r: _hm(r[1], r[2]))
    unseen = torch.tensor([r[2] for r in curve])
    seen = torch.tensor([r[1] for r in curve])
    return {
        "best_seen":    max(r[1] for r in curve),
        "best_unseen":  max(r[2] for r in curve),
        "best_hm":      _hm(best[1], best[2]),
        "seen_at_hm":   best[1],
        "unseen_at_hm": best[2],
        "bias_at_hm":   best[0],
        "auc":          float(torch.trapezoid(seen, unseen).abs()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    clip_encoder, primitive_heads = load_models(Path(args.checkpoint), device)

    test_dataset = MITStatesDataset(
        transform=clip_encoder.preprocess,
        phase=args.phase,
        images_root=args.data_root,
        splits_root=args.split_root,
    )

    # ── Candidate set: seen (train) pairs + this phase's pairs ────────────────
    seen_pairs: set[tuple[str, str]] = set(_load_pairs(Path(args.split_root) / "train_pairs.txt"))
    phase_pairs: set[tuple[str, str]] = set(_load_pairs(Path(args.split_root) / f"{args.phase}_pairs.txt"))
    candidate_pairs = sorted(seen_pairs | phase_pairs)
    cand2idx = {p: i for i, p in enumerate(candidate_pairs)}

    n_seen = sum(1 for p in candidate_pairs if p in seen_pairs)
    logger.info(
        "Candidate set: %d pairs (%d seen + %d unseen) for phase=%s",
        len(candidate_pairs), n_seen, len(candidate_pairs) - n_seen, args.phase,
    )
    expected = {"test": 1662, "val": 1562}
    if args.phase in expected and len(candidate_pairs) != expected[args.phase]:
        raise RuntimeError(
            f"Candidate set size {len(candidate_pairs)} != expected {expected[args.phase]} "
            f"for phase={args.phase}. Verify split files at {args.split_root}."
        )
    logger.info("%s samples: %d", args.phase.capitalize(), len(test_dataset))

    # ── Text banks ────────────────────────────────────────────────────────────
    logger.info("Encoding %d pair texts via CLIP text encoder …", len(candidate_pairs))
    attr_bank, obj_bank, comp_bank = (
        b.to(device) for b in encode_all_pairs(clip_encoder, candidate_pairs, args.batch_size, device)
    )
    logger.info("Scoring weights: λc=%.2f  λa=%.2f  λo=%.2f", args.lambda_c, args.lambda_a, args.lambda_o)

    seen_mask = torch.tensor([p in seen_pairs for p in candidate_pairs], dtype=torch.bool)

    # ── Score every test image against every candidate ────────────────────────
    all_scores, all_pair_idxs = [], []
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    with torch.no_grad():
        for batch_idx, (images, _texts, _a, _o, pair_idxs) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            patch_tokens = clip_encoder.get_visual_features(images)          # (B, P, 1024)

            attr_pred  = primitive_heads.forward_attribute(patch_tokens)     # (B, 768)
            obj_pred   = primitive_heads.forward_object(patch_tokens)        # (B, 768)
            visual_vec = F.normalize(patch_tokens.mean(dim=1), dim=-1)       # (B, 1024)
            comp_pred  = primitive_heads.compose(attr_pred, obj_pred, visual_vec)

            sims = (
                args.lambda_c * (comp_pred @ comp_bank.T)
              + args.lambda_a * (attr_pred @ attr_bank.T)
              + args.lambda_o * (obj_pred  @ obj_bank.T)
            )                                                                # (B, num_candidates)
            all_scores.append(sims.cpu())
            all_pair_idxs.append(pair_idxs)

            if (batch_idx + 1) % 20 == 0:
                logger.info("  processed %d / %d batches", batch_idx + 1, len(test_loader))

    base_scores = torch.cat(all_scores)                                      # (N, num_candidates)
    gt_vocab = torch.cat(all_pair_idxs).tolist()                             # full-vocab indices
    gt_cand = torch.tensor([cand2idx[test_dataset.pairs[i]] for i in gt_vocab], dtype=torch.long)
    gt_is_seen = torch.tensor([test_dataset.pairs[i] in seen_pairs for i in gt_vocab], dtype=torch.bool)

    # ── Single fixed-γ pass ───────────────────────────────────────────────────
    if not args.gamma_sweep:
        scores = base_scores + args.gamma * (~seen_mask).float().unsqueeze(0)
        correct = scores.argmax(dim=1) == gt_cand
        s = correct[gt_is_seen].float().mean().item()
        u = correct[~gt_is_seen].float().mean().item()
        print()
        print("=" * 52)
        print(f"  VL-JEPA | MIT-States {args.phase} | fixed γ={args.gamma}")
        print("=" * 52)
        print(f"  Seen accuracy   :  {s * 100:6.2f}%")
        print(f"  Unseen accuracy :  {u * 100:6.2f}%")
        print(f"  Harmonic mean   :  {_hm(s, u) * 100:6.2f}%")
        print(f"  λc={args.lambda_c}  λa={args.lambda_a}  λo={args.lambda_o}")
        print("=" * 52)
        return

    # ── Exact calibration curve ───────────────────────────────────────────────
    curve = exact_calibration_curve(base_scores, gt_cand, seen_mask, gt_is_seen)
    m = summarize_curve(curve)

    # Print ~20 evenly spaced points so the trade-off is visible in the log.
    W = 52
    step = max(1, len(curve) // 20)
    print()
    print("=" * W)
    print(f"  VL-JEPA | MIT-States {args.phase} | calibration curve")
    print("=" * W)
    print(f"  {'bias >':>10}   {'Seen':>8}   {'Unseen':>8}   {'HM':>8}")
    print("-" * W)
    for b, s, u in curve[::step]:
        print(f"  {b:>10.4f}   {s*100:>7.2f}%   {u*100:>7.2f}%   {_hm(s, u)*100:>7.2f}%")
    print("=" * W)
    print(f"  Best seen   : {m['best_seen']*100:6.2f}%")
    print(f"  Best unseen : {m['best_unseen']*100:6.2f}%")
    print(f"  Best HM     : {m['best_hm']*100:6.2f}%   "
          f"(seen {m['seen_at_hm']*100:.2f}%, unseen {m['unseen_at_hm']*100:.2f}%, bias > {m['bias_at_hm']:.4f})")
    print(f"  AUC         : {m['auc']*100:6.2f}")
    print(f"  λc={args.lambda_c}  λa={args.lambda_a}  λo={args.lambda_o}   points on curve: {len(curve)}")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()