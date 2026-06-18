"""
CZSL evaluation for VL-JEPA on MIT-States.

Protocol
--------
1. Pre-compute y_encoder embeddings for every (attr, obj) pair in the vocabulary.
2. For each test image, run visual_encoder → predictor (empty text query) to
   produce a visual prediction in the shared embedding space.
3. Three-branch scoring following PromptCCZSL protocol: score against attr,
   obj, and composition embeddings separately, then combine.
4. Report seen accuracy, unseen accuracy, and harmonic mean (HM).

Seen / unseen split: pairs listed in train_pairs.txt are "seen"; all other
vocabulary pairs (val + test only) are "unseen".
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.mit_states import MITStates as MITStatesDataset, _load_pairs
from models.loss import InfoNCELoss
from models.predictor import Predictor
from models.visual_encoder import VisualEncoder
from models.y_encoder import YEncoder

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
    parser.add_argument("--num_frames", type=int, default=2,
                        help="clip length passed to VisualEncoder (must match training)")
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
    return parser.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(
    ckpt_path: Path, device: torch.device
) -> tuple[VisualEncoder, YEncoder, Predictor]:
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    logger.info("Building VisualEncoder …")
    visual_encoder = VisualEncoder.load_pretrained(is_frozen=True)
    visual_encoder.load_state_dict(state["visual_encoder"])
    visual_encoder.to(device).eval()

    logger.info("Building YEncoder …")
    y_encoder = YEncoder.load_pretrained(device=device).to(device)
    y_encoder.load_state_dict(state["y_encoder"])
    y_encoder.eval()

    logger.info("Building Predictor …")
    predictor = Predictor.load_pretrained(device=device).to(device)
    predictor.load_state_dict(state["predictor"])
    predictor.eval()

    return visual_encoder, y_encoder, predictor


# ── Pair embedding cache ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_all_pairs(
    y_encoder: YEncoder,
    pairs: list[tuple[str, str]],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three embedding banks: attr, obj, and composition (attr+obj summed).

    Each is (num_pairs, 1536) L2-normalised. Used for three-branch scoring
    at eval time, following PromptCCZSL inference protocol.
    """
    import torch.nn.functional as F
    attrs = [attr for attr, obj in pairs]
    objs  = [obj  for attr, obj in pairs]
    attr_chunks, obj_chunks, comp_chunks = [], [], []
    for i in range(0, len(attrs), batch_size):
        ae = y_encoder(attrs[i : i + batch_size])            # (B, 1536)
        oe = y_encoder(objs[i  : i + batch_size])            # (B, 1536)
        ce = F.normalize(ae + oe, dim=-1)                    # (B, 1536)
        attr_chunks.append(ae.cpu())
        obj_chunks.append(oe.cpu())
        comp_chunks.append(ce.cpu())
    return (
        torch.cat(attr_chunks, dim=0),   # (num_pairs, 1536)
        torch.cat(obj_chunks,  dim=0),
        torch.cat(comp_chunks, dim=0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    test_dataset = MITStatesDataset(
        phase="test",
        num_frames=args.num_frames,
        images_root=args.data_root,
        splits_root=args.split_root,
    )
    # Seen pairs = those that appear in the training split file.
    seen_pairs: set[tuple[str, str]] = set(
        _load_pairs(Path(args.split_root) / "train_pairs.txt")
    )

    num_seen_vocab   = sum(1 for p in test_dataset.pairs if p in seen_pairs)
    num_unseen_vocab = sum(1 for p in test_dataset.pairs if p not in seen_pairs)
    logger.info(
        "Test samples: %d | Vocabulary: %d pairs (%d seen, %d unseen)",
        len(test_dataset), len(test_dataset.pairs),
        num_seen_vocab, num_unseen_vocab,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    visual_encoder, y_encoder, predictor = load_models(
        Path(args.checkpoint), device
    )

    # ── Pre-compute three embedding banks ────────────────────────────────────
    logger.info("Encoding %d pair texts via y_encoder …", len(test_dataset.pairs))
    attr_embeds_bank, obj_embeds_bank, comp_embeds_bank = encode_all_pairs(
        y_encoder, test_dataset.pairs, args.batch_size, device
    )
    attr_embeds_bank = attr_embeds_bank.to(device)   # (num_pairs, 1536)
    obj_embeds_bank  = obj_embeds_bank.to(device)
    comp_embeds_bank = comp_embeds_bank.to(device)

    logger.info(
        "Scoring weights: λc=%.2f  λa=%.2f  λo=%.2f",
        args.lambda_c, args.lambda_a, args.lambda_o,
    )

    # Boolean mask: True for pairs present in the training split.
    seen_mask = torch.tensor(
        [p in seen_pairs for p in test_dataset.pairs],
        dtype=torch.bool, device=device,
    )                                        # (num_pairs,)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    total_seen = correct_seen = 0
    total_unseen = correct_unseen = 0

    with torch.no_grad():
        for batch_idx, (clips, _texts, _attr_idxs, _obj_idxs, pair_idxs) in enumerate(
            test_loader
        ):
            clips     = clips.to(device, non_blocking=True)   # (B, F, C, H, W)
            pair_idxs = pair_idxs.to(device)                  # (B,)

            # Step 1 — Visual encoding.
            patch_tokens = visual_encoder(clips)               # (B, F, P, 1024)

            # Step 2 — Visual prediction in the shared embedding space.
            empty_texts = ["a photo of"] * clips.size(0)
            pred_embeds = predictor(patch_tokens, empty_texts)  # (B, 1536), L2-normed

            # Step 3 — Three-branch scoring following PromptCCZSL protocol.
            # Score against attr, obj, and composition embeddings separately,
            # then combine. λ weights are argparse-tunable.
            sims = (
                args.lambda_c * (pred_embeds @ comp_embeds_bank.T)
              + args.lambda_a * (pred_embeds @ attr_embeds_bank.T)
              + args.lambda_o * (pred_embeds @ obj_embeds_bank.T)
            )                                                  # (B, num_pairs)
            preds = sims.argmax(dim=1)                        # (B,)

            for pred_idx, gt_idx in zip(preds.tolist(), pair_idxs.tolist()):
                gt_pair  = test_dataset.pairs[gt_idx]
                is_seen  = gt_pair in seen_pairs
                correct  = pred_idx == gt_idx
                if is_seen:
                    total_seen   += 1
                    correct_seen  += int(correct)
                else:
                    total_unseen   += 1
                    correct_unseen += int(correct)

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    "  processed %d / %d batches",
                    batch_idx + 1, len(test_loader),
                )

    # ── Results ───────────────────────────────────────────────────────────────
    seen_acc   = correct_seen   / total_seen   if total_seen   else 0.0
    unseen_acc = correct_unseen / total_unseen if total_unseen else 0.0
    hm = (
        2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
        if (seen_acc + unseen_acc) > 0
        else 0.0
    )

    print()
    print("=" * 52)
    print("  VL-JEPA  |  MIT-States CZSL Results")
    print("=" * 52)
    print(f"  Seen accuracy   :  {seen_acc * 100:6.2f}%"
          f"  ({correct_seen}/{total_seen})")
    print(f"  Unseen accuracy :  {unseen_acc * 100:6.2f}%"
          f"  ({correct_unseen}/{total_unseen})")
    print(f"  Harmonic mean   :  {hm * 100:6.2f}%")
    print(f"  λc={args.lambda_c}  λa={args.lambda_a}  λo={args.lambda_o}")
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()