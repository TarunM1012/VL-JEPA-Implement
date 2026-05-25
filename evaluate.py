"""
CZSL evaluation for VL-JEPA on MIT-States.

Protocol
--------
1. Pre-compute y_encoder embeddings for every (attr, obj) pair in the vocabulary.
2. For each test image, run visual_encoder → predictor (empty text query) to
   produce a visual prediction in the shared embedding space.
3. Rank all pair embeddings by cosine similarity; predict the top-1 pair.
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
    y_encoder = YEncoder.load_pretrained(device=device)
    y_encoder.load_state_dict(state["y_encoder"])
    y_encoder.eval()

    logger.info("Building Predictor …")
    predictor = Predictor.load_pretrained(device=device)
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
) -> torch.Tensor:
    """Return (num_pairs, 1536) L2-normalised embeddings for every pair text."""
    texts = [f"{attr} {obj}" for attr, obj in pairs]
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunk = y_encoder(texts[i : i + batch_size])   # (B, 1536), L2-normalised
        chunks.append(chunk.cpu())
    return torch.cat(chunks, dim=0)                    # (num_pairs, 1536)


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

    # ── Pre-compute pair embeddings ───────────────────────────────────────────
    logger.info("Encoding %d pair texts via y_encoder …", len(test_dataset.pairs))
    pair_embeds = encode_all_pairs(
        y_encoder, test_dataset.pairs, args.batch_size, device
    ).to(device)                             # (num_pairs, 1536)

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
            # An empty text query lets visual tokens dominate; the predictor
            # then produces an embedding that should be nearest to the correct
            # pair text embedding.
            empty_texts  = [""] * clips.size(0)
            pred_embeds  = predictor(patch_tokens, empty_texts)  # (B, 1536), L2-normed

            # Step 3 — Nearest-neighbour retrieval over all pair embeddings.
            # Both sides are L2-normalised so dot product == cosine similarity.
            sims  = pred_embeds @ pair_embeds.T               # (B, num_pairs)
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
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()
